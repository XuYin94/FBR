import torch
import torch.nn as nn
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
from region_utils import Bank_contrast_loss, get_contrast_loss,Compled_reco
import importlib

import voc12.dataloader
from misc import pyutils, imutils

import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm
from PIL import Image


def get_background_label(cam, label):
    cam_up = F.relu(cam[:, 1:]) * (label.unsqueeze(-1).unsqueeze(-1))
    # print(cam_up.shape)
    norm_cam = cam_up / (F.adaptive_max_pool2d(cam_up, (1, 1)) + 1e-5)
    norm_cam = torch.sum(norm_cam, dim=1)
    pseudo_bg_label = (norm_cam < 0.05).float()

    return pseudo_bg_label


def balanced_cross_entropy(logits, labels, one_hot_labels):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """

    N, C, H, W = logits.shape

    assert one_hot_labels.size(0) == N and one_hot_labels.size(
        1) == C, f'label tensor shape is {one_hot_labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    loss_structure = -torch.sum(log_logits * one_hot_labels, dim=1)  # (N)

    ignore_mask_bg = torch.zeros_like(labels)
    ignore_mask_fg = torch.zeros_like(labels)

    ignore_mask_bg[labels == 0] = 1
    ignore_mask_fg[(labels != 0) & (labels != 255)] = 1

    loss_bg = (loss_structure * ignore_mask_bg).sum() / ignore_mask_bg.sum()
    loss_fg = (loss_structure * ignore_mask_fg).sum() / ignore_mask_fg.sum()

    return (loss_bg + loss_fg) / 2


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def run(args):
    model = getattr(importlib.import_module(args.amn_network), 'Net')()

    train_dataset = voc12.dataloader.VOC12SegmentationDataset(args.train_list,
                                                              label_dir=args.ir_label_out_dir,
                                                              voc12_root=args.voc12_root,
                                                              hor_flip=True,
                                                              crop_size=args.amn_crop_size,
                                                              crop_method="random",
                                                              rescale=(0.5, 1.5)
                                                              )

    train_data_loader = DataLoader(train_dataset, batch_size=args.amn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = voc12.dataloader.VOC12SegmentationDataset(args.infer_list,
                                                            label_dir=args.ir_label_out_dir,
                                                            voc12_root=args.voc12_root,
                                                            crop_size=None,
                                                            crop_method="none",
                                                            )

    val_data_loader = DataLoader(val_dataset, batch_size=1,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    param_groups = model.trainable_parameters()

    optimizer = torch.optim.Adam(
        params=[
            {
                'params': param_groups[0],
                'lr': 5e-06,
                'weight_decay': 1.0e-4,
            },
            {
                'params': param_groups[1],
                'lr': 1e-04,
                'weight_decay': 1.0e-4,
            },
        ],
    )

    total_epochs = args.amn_num_epoches

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    reco_loss=Compled_reco()
    best_acc=0.0
    for ep in range(total_epochs):
        loader_iter = iter(train_data_loader)

        pbar = tqdm(
            range(1, len(train_data_loader) + 1),
            total=len(train_data_loader),
            dynamic_ncols=True,
        )

        for iteration, _ in enumerate(pbar):
            optimizer.zero_grad()
            try:
                pack = next(loader_iter)
            except:
                loader_iter = iter(train_data_loader)
                pack = next(loader_iter)

            img = pack['img'].cuda(non_blocking=True)
            label_amn = pack['label'].long().cuda(non_blocking=True)
            label_cls = pack['label_cls'].cuda(non_blocking=True)

            logit, loss_sim,fg_fea,bg_fea= model(img, label_cls)
            # loss_bg_seg = F.binary_cross_entropy(torch.sigmoid(bg_pre[:, 0]), get_background_label(logit, label_cls))

            B, C, H, W = logit.shape

            label_amn = resize_labels(label_amn.cpu(), size=logit.shape[-2:]).cuda()

            label_ = label_amn.clone()
            label_[label_amn == 255] = 0

            given_labels = torch.full(size=(B, C, H, W), fill_value=args.eps/(C-1)).cuda()
            given_labels.scatter_(dim=1, index=torch.unsqueeze(label_, dim=1), value=1-args.eps)

            loss_pcl = balanced_cross_entropy(logit, label_amn, given_labels)


            loss_fg_reco,loss_bg_reco = get_contrast_loss(logit, fg_fea,bg_fea,reco_loss)
            loss = loss_pcl + 0.1*loss_sim+loss_fg_reco+loss_bg_reco

            loss.backward()

            optimizer.step()

            avg_meter.add({'loss_pcl': loss_pcl.item()})
            avg_meter.add({'loss_sim': loss_sim.mean().item()})
            avg_meter.add({'loss_fg': loss_fg_reco.mean().item()})
            avg_meter.add({'loss_bg': loss_bg_reco .item()})
            # avg_meter.add({'loss_bg': loss_bg_seg.item()})
            # | BG_SEG: {avg_meter.pop('loss_bg'):.4f}
            pbar.set_description(f"[{ep + 1}/{total_epochs}] "
                                 f"PCL: [{avg_meter.pop('loss_pcl'):.4f}"
                                 f"|Sim: [{avg_meter.pop('loss_sim'):.4f}"
                                 f"|Fg_reco: {avg_meter.pop('loss_fg'):.4f}"
                                 f"|Bg_reco: {avg_meter.pop('loss_bg'):.4f}]")

        with torch.no_grad():
            model.eval()
            dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
            labels = []
            preds = []

            for i, pack in enumerate(tqdm(val_data_loader)):
                img_name = pack['name'][0]
                img = pack['img']
                label_cls = pack['label_cls'][0]

                img = img.cuda()

                logit, __,__,__= model(img, pack['label_cls'].cuda())

                size = img.shape[-2:]
                strided_up_size = imutils.get_strided_up_size(size, 16)

                valid_cat = torch.nonzero(label_cls)[:, 0]
                keys = np.pad(valid_cat + 1, (1, 0), mode='constant')

                logit_up = F.interpolate(logit, strided_up_size, mode='bilinear', align_corners=False)
                logit_up = logit_up[0, :, :size[0], :size[1]]

                logit_up = F.softmax(logit_up, dim=0)[keys].cpu().numpy()

                cls_labels = np.argmax(logit_up, axis=0)
                cls_labels = keys[cls_labels]

                preds.append(cls_labels.copy())

                gt_label = dataset.get_example_by_keys(i, (1,))[0]

                labels.append(gt_label.copy())

            confusion = calc_semantic_segmentation_confusion(preds, labels)

            gtj = confusion.sum(axis=1)
            resj = confusion.sum(axis=0)
            gtjresj = np.diag(confusion)
            denominator = gtj + resj - gtjresj
            iou = gtjresj / denominator

            print(f'[{ep + 1}/{total_epochs}] miou: {np.nanmean(iou):.4f}')

            model.train()
        if best_acc<=np.nanmean(iou):
            best_acc=np.nanmean(iou)
            torch.save(model.state_dict(), args.amn_weights_name + '.pth')
    torch.cuda.empty_cache()