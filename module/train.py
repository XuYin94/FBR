import os
import torch
from torch.nn import functional as F
from eps import get_eps_loss
from util import pyutils
from torchvision import transforms
from util import mytool
import util.transform as local_transform
from torchvision.utils import make_grid
from PIL import Image
import region_utils
from region_utils import Bank_contrast_loss
import random
import numpy as np
from eps import get_eps_loss


def get_reliable_pseudo_label(pseudo_label,scores,rate=0.6):

    batch,nbr_class,h,w=scores.shape

    cam_sure_region = torch.zeros_like(pseudo_label, dtype=bool)
    for i in range(batch):
        single_img_classes = torch.unique(pseudo_label[i])

        single_img_classes = single_img_classes[single_img_classes != 20]
        for class_i in single_img_classes:
            class_not_region = (pseudo_label[i] != class_i)
            cam_class = scores[i,class_i, :, :]
            cam_class[class_not_region] = 0
            cam_class_order = cam_class[cam_class > 0.05]
            cam_class_order ,__= torch.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] *rate)
            #print(cam_class_order.shape)
            #print(confidence_pos)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)
            cam_sure_region[i] = torch.logical_or(cam_sure_region[i], class_sure_region)
    cam_not_sure_region = ~cam_sure_region
    reliable_pseudo_label=pseudo_label.clone()
    reliable_pseudo_label[(cam_not_sure_region)&(pseudo_label!=20)]=255
    return reliable_pseudo_label

def get_pseudo_label(cam,label):
    #cam_bg=cam[:,0]
    cam_fg=cam[:,:-1]*label
    #print(F.adaptive_max_pool2d(cam_fg, (1, 1)).shape)
    cam_fg/=F.adaptive_max_pool2d(cam_fg, (1, 1)) + 1e-5
    cam_bg=torch.ones((cam.shape[0],1,cam.shape[-2],cam.shape[-1])).cuda()*0.20
    cam_up=torch.cat([cam_bg,cam_fg],dim=1)
    pseudo_label=torch.argmax(cam_up,dim=1)

    batch,nbr_class,h,w=cam.shape

    cam_sure_region = torch.zeros_like(pseudo_label, dtype=bool)
    for i in range(batch):
        single_img_classes = torch.unique(pseudo_label[i])
        single_img_classes = single_img_classes[single_img_classes != 0]
        for class_i in single_img_classes:
            if class_i != 0:
                class_not_region = (pseudo_label[i] != class_i)
                cam_class = cam_up[i,class_i, :, :]
                cam_class[class_not_region] = 0
                cam_class_order = cam_class[cam_class > 0.1]
                cam_class_order ,__= torch.sort(cam_class_order)
                #print(cam_class_order.shape)
                confidence_pos = int(cam_class_order.shape[0] *0.4)
                confidence_value = cam_class_order[confidence_pos]
                class_sure_region = (cam_class > confidence_value)
                cam_sure_region[i] = torch.logical_or(cam_sure_region[i], class_sure_region)
            else:
                class_not_region = (pseudo_label[i] != class_i)
                cam_class = cam_up[i,class_i, :, :]
                cam_class[class_not_region] = 0
                class_sure_region = (cam_class > 0.8)
                cam_sure_region[i] = torch.logical_or(cam_sure_region[i], class_sure_region)
    cam_not_sure_region = ~cam_sure_region
    reliable_pseudo_label=pseudo_label.clone()
    reliable_pseudo_label[cam_not_sure_region]=255
    return pseudo_label,reliable_pseudo_label



def get_background_label(cam,label):
    cam_up = F.relu(cam[:, :-1])*(label.unsqueeze(-1).unsqueeze(-1))
    #print(cam_up.shape)
    norm_cam = cam_up / (F.adaptive_max_pool2d(cam_up, (1, 1)) + 1e-5)
    norm_cam = torch.sum(norm_cam, dim=1)
    pseudo_bg_label = (norm_cam < 0.05).float()

    return pseudo_bg_label


def max_norm(p, e=1e-5):
    if p.dim() == 3:
        C, H, W = p.size()
        # p = F.relu(p)
        max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        # min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
        # p = F.relu(p-min_v-e)/(max_v-min_v+e)
        p = p / (max_v + e)
    elif p.dim() == 4:
        N, C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        p = F.relu(p - min_v - e) / (max_v - min_v + e)
        # p = p / (max_v + e)
    return p


def get_contrast_loss(bg_memobank1,bg_memobank2,cam1, cam2, f_proj1, f_proj2,b_proj1,b_proj2, label,gamma=0.05, bg_thres=0.05):
    n1, c1, h1, w1 = cam1.shape
    n2, c2, hw, w2 = cam2.shape
    assert n1 == n2

    bg_score = torch.ones((n1, 1)).cuda()
    label = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)
    f_proj1 = F.interpolate(f_proj1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
    b_proj1 = F.interpolate(b_proj1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
    cam1 = F.interpolate(cam1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)

    with torch.no_grad():

        f_fea1 = f_proj1.detach()
        #tmp_f_fea1=f_proj1.detach()
        b_fea1 = b_proj1.detach()

        c_fea1 = f_fea1.shape[1]
        cam_rv1_down = F.relu(cam1.detach())

        n1, c1, h1, w1 = cam_rv1_down.shape
        max1 = torch.max(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        min1 = torch.min(cam_rv1_down.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        cam_rv1_down[cam_rv1_down < min1 + 1e-5] = 0.
        norm_cam1 = (cam_rv1_down - min1 - 1e-5) / (max1 - min1 + 1e-5)
        # norm_cam1 = cam_rv1_down / (max1 + 1e-5)
        cam_rv1_down = norm_cam1
        cam_rv1_down[:, -1, :, :] = bg_thres
        scores1 = F.softmax(cam_rv1_down * label, dim=1)
        pseudo_label1 = scores1.argmax(dim=1, keepdim=True)

        reliable_pseudo_mask1=get_reliable_pseudo_label(pseudo_label1[:,0],cam_rv1_down,rate=0.6)

        n_sc1, c_sc1, h_sc1, w_sc1 = scores1.shape

        f_fea1 = f_fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)  # (nhw, 128)
        b_fea1 = b_fea1.permute(0, 2, 3, 1).reshape(-1, c_fea1)  # (nhw, 128)

        top_values, top_indices = torch.topk(cam_rv1_down.transpose(0, 1).reshape(c_sc1, -1),
                                             k=h_sc1 * w_sc1 // 16, dim=-1)
        prototypes1 = torch.zeros(c_sc1, c_fea1).cuda()  # [21, 128]
        for i in range(c_sc1):
            top_fea = f_fea1[top_indices[i]]
            prototypes1[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])

        tmp_prototpes1=prototypes1.detach()
        prototypes1 = F.normalize(prototypes1, dim=-1)
        # For target
        f_fea2 = f_proj2.detach()
        b_fea2=b_proj2.detach()
        c_fea2 = f_fea2.shape[1]

        cam_rv2_down = F.relu(cam2.detach())
        n2, c2, h2, w2 = cam_rv2_down.shape
        max2 = torch.max(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
        min2 = torch.min(cam_rv2_down.view(n2, c2, -1), dim=-1)[0].view(n2, c2, 1, 1)
        cam_rv2_down[cam_rv2_down < min2 + 1e-5] = 0.
        norm_cam2 = (cam_rv2_down - min2 - 1e-5) / (max2 - min2 + 1e-5)

        # max norm
        cam_rv2_down = norm_cam2
        cam_rv2_down[:, -1, :, :] = bg_thres

        scores2 = F.softmax(cam_rv2_down * label, dim=1)
        pseudo_label2 = scores2.argmax(dim=1, keepdim=True)

        reliable_pseudo_mask2=get_reliable_pseudo_label(pseudo_label2[:,0],scores2,rate=0.6)
        # pseudo_label2


        n_sc2, c_sc2, h_sc2, w_sc2 = scores2.shape
        #scores2 = scores2.transpose(0, 1)  # (21, N, H/8, W/8)
        f_fea2 = f_fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)  # (N*C*H*W)
        b_fea2=b_fea2.permute(0, 2, 3, 1).reshape(-1, c_fea2)  # (N*C*H*W)
        top_values2, top_indices2 = torch.topk(cam_rv2_down.transpose(0, 1).reshape(c_sc2, -1), k=h_sc2 * w_sc2 // 16,
                                               dim=-1)
        prototypes2 = torch.zeros(c_sc2, c_fea2).cuda()

        for i in range(c_sc2):
            top_fea2 = f_fea2[top_indices2[i]]
            prototypes2[i] = torch.sum(top_values2[i].unsqueeze(-1) * top_fea2, dim=0) / torch.sum(top_values2[i])

        tmp_prototypes2=prototypes2.detach()
        # L2 Norm
        prototypes2 = F.normalize(prototypes2, dim=-1)

    # Contrast Loss
    n_f, c_f, h_f, w_f = f_proj1.shape
    f_proj1 = f_proj1.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
    f_proj1 = F.normalize(f_proj1, dim=-1)
    pseudo_label1 = pseudo_label1.reshape(-1)
    positives1 = prototypes2[pseudo_label1]
    negitives1 = prototypes2
    n_f, c_f, h_f, w_f = f_proj2.shape
    f_proj2 = f_proj2.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
    f_proj2 = F.normalize(f_proj2, dim=-1)
    pseudo_label2 = pseudo_label2.reshape(-1)
    positives2 = prototypes1[pseudo_label2]  # (N, 128)
    negitives2 = prototypes1

    mask_s = (pseudo_label1!=20).bool()
    mask_t = (pseudo_label2!=20).bool()

    A1 = torch.exp(torch.sum(f_proj1 * positives1, dim=-1) / 0.1)
    A2 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives1.transpose(0, 1)) / 0.1), dim=-1)
    loss_nce1 = torch.mean((-1 * torch.log(A1 / A2)*mask_s))

    A3 = torch.exp(torch.sum(f_proj2 * positives2, dim=-1) / 0.1)
    A4 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives2.transpose(0, 1)) / 0.1), dim=-1)
    loss_nce2 = torch.mean((-1 * torch.log(A3 / A4)*mask_t))

    loss_cross_nce = gamma * (loss_nce1 + loss_nce2) / 2

    A1_view1 = torch.exp(torch.sum(f_proj1 * positives2, dim=-1) / 0.1)
    A2_view1 = torch.sum(torch.exp(torch.matmul(f_proj1, negitives2.transpose(0, 1)) / 0.1), dim=-1)
    loss_cross_nce2_1 = torch.mean((-1 * torch.log(A1_view1 / A2_view1)*mask_t))

    A3_view2 = torch.exp(torch.sum(f_proj2 * positives1, dim=-1) / 0.1)
    A4_view2 = torch.sum(torch.exp(torch.matmul(f_proj2, negitives1.transpose(0, 1)) / 0.1), dim=-1)

    loss_cross_nce2_2 = torch.mean((-1 * torch.log(A3_view2 / A4_view2)*mask_s))

    loss_cross_nce2 = gamma * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2

    positives_intra1 = prototypes1[pseudo_label1]
    negitives_intra1 = prototypes1
    similarity_intra1 = (torch.sum(f_proj1 * positives_intra1, dim=-1) + 1) / 2.

    A1_intra_view1 = torch.exp(torch.sum(f_proj1 * positives_intra1, dim=-1) / 0.1)
    neg_scores = torch.matmul(f_proj1, negitives_intra1.transpose(0, 1))[:,:-1]  # (n*h*w, 21)
    #with torch.no_grad():
        #random_indices = torch.tensor([random.sample(range(21), 10) for _ in range(n_f * h_f * w_f)]).long()

    with torch.no_grad():
        _, lower_indices = torch.topk(neg_scores, k=13, largest=True, dim=-1)
        lower_indices = lower_indices[:, 3:]

    negitives_intra1 = negitives_intra1.unsqueeze(0).repeat(n_f * h_f * w_f, 1, 1)
    #random_negitives_intra1 = negitives_intra1[torch.arange(n_f * h_f * w_f).unsqueeze(1), random_indices]
    lower_negitives_intra1 = negitives_intra1[torch.arange(n_f * h_f * w_f).unsqueeze(1), lower_indices]
    negitives_intra1 = torch.cat([positives_intra1.unsqueeze(1), lower_negitives_intra1], dim=1)
    A2_intra_view1 = torch.sum(
        torch.exp(torch.matmul(f_proj1.unsqueeze(1), negitives_intra1.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)

    loss_intra_nce1 = torch.zeros(1).cuda()
    C = 0
    exists = np.unique(pseudo_label1.cpu().numpy()).tolist()
    for i_ in range(20):  # for each class
        if not i_ in exists:
            continue
        C += 1
        A1_intra_view1_class = A1_intra_view1[pseudo_label1 == i_]
        A2_intra_view1_class = A2_intra_view1[pseudo_label1 == i_]
        similarity_intra1_class = similarity_intra1[pseudo_label1 == i_]

        len_class = A1_intra_view1_class.shape[0]
        if len_class < 2:
            continue

        with torch.no_grad():
            random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
        random_A1_intra_view1 = A1_intra_view1_class[random_indices]  # (n, hw//2)
        random_A2_intra_view1 = A2_intra_view1_class[random_indices]

        with torch.no_grad():
            _, lower_indices = torch.topk(similarity_intra1_class, k=int(len_class * 0.6), largest=False)
            lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

        lower_A1_intra_view1 = A1_intra_view1_class[lower_indices]
        lower_A2_intra_view1 = A2_intra_view1_class[lower_indices]

        A1_intra_view1_class = torch.cat([random_A1_intra_view1, lower_A1_intra_view1], dim=0)  # (hw)
        A2_intra_view1_class = torch.cat([random_A2_intra_view1, lower_A2_intra_view1], dim=0)
        A1_intra_view1_class = A1_intra_view1_class.reshape(-1)
        A2_intra_view1_class = A2_intra_view1_class.reshape(-1)
        loss_intra_nce1 += torch.mean(-1 * torch.log(A1_intra_view1_class / A2_intra_view1_class))



    # mea over classes
    loss_intra_nce1 = loss_intra_nce1 / C

    # for target
    positives_intra2 = prototypes2[pseudo_label2]
    negitives_intra2 = prototypes2
    similarity_intra2 = (torch.sum(f_proj2 * positives_intra2, dim=-1) + 1) / 2.

    A3_intra_view2 = torch.exp(torch.sum(f_proj2 * positives_intra2, dim=-1) / 0.1)
    neg_scores = torch.matmul(f_proj2, negitives_intra2.transpose(0, 1))[:,:-1]  # (n*h*w, 21)

    with torch.no_grad():
        random_indices = torch.tensor([random.sample(range(20), 10) for _ in range(n_f * h_f * w_f)]).long()

    with torch.no_grad():
        _, lower_indices = torch.topk(neg_scores, k=13, largest=True, dim=-1)
        lower_indices = lower_indices[:, 3:]

    negitives_intra2 = negitives_intra2.unsqueeze(0).repeat(n_f * h_f * w_f, 1, 1)
    random_negitives_intra2 = negitives_intra2[torch.arange(n_f * w_f * h_f).unsqueeze(1), random_indices]
    lower_negitives_intra2 = negitives_intra2[torch.arange(n_f * w_f * h_f).unsqueeze(1), lower_indices]
    negitives_intra2 = torch.cat([positives_intra2.unsqueeze(1), lower_negitives_intra2], dim=1)

    A4_intra_view2 = torch.sum(
        torch.exp(torch.matmul(f_proj2.unsqueeze(1), negitives_intra2.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)
    loss_intra_nce2 = torch.zeros(1).cuda()
    C = 0
    exists = np.unique(pseudo_label2.cpu().numpy()).tolist()
    for i_ in range(20):
        if not i_ in exists:
            continue
        C += 1
        A3_intra_view2_class = A3_intra_view2[pseudo_label2 == i_]
        A4_intra_view2_class = A4_intra_view2[pseudo_label2 == i_]
        similarity_intra2_class = similarity_intra2[pseudo_label2 == i_]
        len_class = A3_intra_view2_class.shape[0]

        if len_class < 2:
            continue

        with torch.no_grad():
            random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
        random_A3_intra_view2 = A3_intra_view2_class[random_indices]  # (n, hw//2)
        random_A4_intra_view2 = A4_intra_view2_class[random_indices]
        with torch.no_grad():  # lowest 50%
            _, lower_indices = torch.topk(similarity_intra2_class, k=int(len_class * 0.6), largest=False)
            lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

        lower_A3_intra_view2 = A3_intra_view2_class[lower_indices]
        lower_A4_intra_view2 = A4_intra_view2_class[lower_indices]
        A3_intra_view2_class = torch.cat([random_A3_intra_view2, lower_A3_intra_view2], dim=0)
        A4_intra_view2_class = torch.cat([random_A4_intra_view2, lower_A4_intra_view2], dim=0)
        A3_intra_view2_class = A3_intra_view2_class.reshape(-1)
        A4_intra_view2_class = A4_intra_view2_class.reshape(-1)

        loss_intra_nce2 += torch.mean(-1 * torch.log(A3_intra_view2_class / A4_intra_view2_class))

    loss_intra_nce2 = loss_intra_nce2 / C

    loss_intra_nce = gamma * (loss_intra_nce1 + loss_intra_nce2) / 2

    loss_nce = loss_cross_nce + loss_cross_nce2 + loss_intra_nce

    pseudo_label1=pseudo_label1.view(n1,-1)
    pseudo_label2=pseudo_label2.view(n2,-1)
    bg_mask1=(pseudo_label1==20).bool()
    bg_mask2=(pseudo_label2==20).bool()
    bg_fea1=b_fea1.view(n1,h_f*w_f,-1)
    bg_fea2=b_fea2.view(n2,h_f*w_f,-1)
    ##update the bg. memory bank
    #keys, pseudo_label, indicator, f_fea, prototypes

    reliable_mask1=reliable_pseudo_mask1.reshape(-1)
    reliable_mask2=reliable_pseudo_mask2.reshape(-1)
    fg_bg_reco_loss1=bg_memobank1(bg_fea1,bg_mask1,pseudo_label1,reliable_mask1,f_fea1,tmp_prototpes1)
    fg_bg_reco_loss2=bg_memobank2(bg_fea2,bg_mask2,pseudo_label2,reliable_mask2,f_fea2,tmp_prototypes2)
    # #print(fg_bg_reco_loss1.item())
    #print(bg_memobank2.memo_bank.shape[0])
    loss_fg_bg_reco=0.01*(fg_bg_reco_loss1+fg_bg_reco_loss2)/2
    return loss_nce,loss_fg_bg_reco


def get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label):
    ns, cs, hs, ws = cam2.size()
    cam1 = F.interpolate(max_norm(cam1), size=(hs, ws), mode='bilinear', align_corners=True) * label
    # cam1 = F.softmax(cam1, dim=1) * label
    # cam2 = F.softmax(cam2, dim=1) * label
    cam2 = max_norm(cam2) * label
    loss_er = torch.mean(torch.abs(cam1[:, :-1, :, :] - cam2[:, :-1, :, :]))

    cam1[:, -1, :, :] = 1 - torch.max(cam1[:, :-1, :, :], dim=1)[0]
    cam2[:, -1, :, :] = 1 - torch.max(cam2[:, :-1, :, :], dim=1)[0]
    cam_rv1 = F.interpolate(max_norm(cam_rv1), size=(hs, ws), mode='bilinear', align_corners=True) * label
    cam_rv2 = max_norm(cam_rv2) * label
    tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
    tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
    loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr = loss_ecr1 + loss_ecr2

    return loss_er, loss_ecr


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance,
    # but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


def train_cls(train_loader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, label = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                img_id, img, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred = model(img)

            # Classification loss
            loss = F.multilabel_soft_margin_loss(pred, label)
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


def train_eps(train_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred, cam = model(img)

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam,
                                                              saliency,
                                                              label,
                                                              args.tau,
                                                              args.alpha,
                                                              intermediate=True)
            loss = loss_cls + loss_sal

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'imps:%.1f' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


def train_contrast(train_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal', 'loss_bg', 'loss_sim', 'loss_fg_bg_reco','loss_nce', 'loss_er',
                                     'loss_ecr')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    gamma = 0.10
    # print(args)
    print('Using Gamma:', gamma)

    bg_bank1=Bank_contrast_loss()
    bg_bank2=Bank_contrast_loss()

    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            img2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
            saliency2 = F.interpolate(saliency, size=(128, 128), mode='bilinear', align_corners=True)

            loss_sim1, pred1, cam1, pred_rv1, cam_rv1, fg_feat1, bg_pre1,bg_fea1 = model(img)
            loss_sim2, pred2, cam2, pred_rv2, cam_rv2, fg_feat2, bg_pre2,bg_fea2 = model(img2)

            # Classification loss 1
            loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
            loss_sal, fg_map, bg_map1, sal_pred = get_eps_loss(cam1, saliency, label, args.tau, args.alpha,
                                                               intermediate=True)
            loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, saliency, label, args.tau, args.alpha, intermediate=True)

            # Classification loss 2
            loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)

            loss_sal2, fg_map2, bg_map2, sal_pred2 = get_eps_loss(cam2, saliency2, label, args.tau, args.alpha,
                                                                  intermediate=True)

            loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, saliency2, label, args.tau, args.alpha, intermediate=True)

            ## background segmentation loss
            # bg_pre1=F.interpolate(bg_pre1,size=(224,224),mode='bilinear',align_corners=True)
            # bg_pre2 = F.interpolate(bg_pre2, size=(128, 128), mode='bilinear', align_corners=True)
            loss_bg_seg1 = F.binary_cross_entropy(torch.sigmoid(bg_pre1[:, 0]), get_background_label(cam1,label))
            loss_bg_seg2 = F.binary_cross_entropy(torch.sigmoid(bg_pre2[:, 0]), get_background_label(cam2,label))

            bg_score = torch.ones((img.shape[0], 1)).cuda()
            label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
            loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
            loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])

            loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)

            loss_nce,loss_reco_fg_bg = get_contrast_loss(bg_bank1,bg_bank2,cam_rv1, cam_rv2, fg_feat1, fg_feat2,bg_fea1,bg_fea2, label,gamma=gamma, bg_thres=0.10)

            # loss cls = cam cls loss + cam_cv cls loss
            loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
            loss_bg_seg = (loss_bg_seg1 + loss_bg_seg2) / 2

            loss_sim = (loss_sim1 + loss_sim2) / 2

            loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.

            loss = loss_cls + loss_sal + loss_nce + loss_er + loss_ecr + 0.1 * loss_sim + 0.1 * loss_bg_seg+loss_reco_fg_bg

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item(),
                           'loss_bg': loss_bg_seg.item(),
                           'loss_sim': loss_sim.item(),
                           'loss_fg_bg_reco':loss_reco_fg_bg.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'Loss_Bg:%.4f' % (avg_meter.pop('loss_bg')),
                      'Loss_Sim:%.4f' % (avg_meter.pop('loss_sim')),
                      'Loss_reco:%.4f'%(avg_meter.pop('loss_fg_bg_reco')),
                      'Loss_Nce:%.4f' % (avg_meter.pop('loss_nce')),
                      'Loss_ER: %.4f' % (avg_meter.pop('loss_er')),
                      'Loss_ECR:%.4f' % (avg_meter.pop('loss_ecr')),
                      'imps:%.1f' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_contrast1.pth'))


def val_cam(cla_model,val_data_loader,writer):
    visual_boarder=[]
    palette=mytool.get_voc_palette(21)


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    restore_transform = transforms.Compose([
        local_transform.DeNormalize(mean, std),
        transforms.ToPILImage()])
    viz_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()])
    pseudo_intersection_meter = mytool.AverageMeter()
    pseudo_union_meter = mytool.AverageMeter()
    pseudo_target_meter = mytool.AverageMeter()
    cla_model.eval()
    with torch.no_grad():
        for iter, pack in enumerate(val_data_loader):
            #name = pack[0]
            images = pack[1].cuda(non_blocking=True)
            #salience=pack[2].cuda(non_blocking=True).float().unsqueeze(1)
            label = pack[-1]
            batch,__,img_w, img_h = images.shape
            label = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            seg_label = pack[3]
            cam=cla_model.forward_cam(images)


            pseudo_bg_label=get_background_label(cam)
            cam=F.softmax(cam,dim=1)
            cam=F.interpolate(cam,size=(img_w,img_h),mode='bilinear',align_corners=True)
            #print(pseudo_bg_label.shape)
            pseudo_bg_label=F.interpolate(pseudo_bg_label.unsqueeze(1),size=(img_w,img_h),mode='bilinear',align_corners=True)

            #print(fg_map.shape)
            pseudo_label,reliable_pseudo_label=get_pseudo_label(cam,label)
            bg_pre=cla_model.forward_bg_pre(images)
            seg_label=seg_label.cuda()


            pseudo_intersection, pseudo_union, pseudo_target = \
                mytool.intersectionAndUnionGPU(pseudo_label.clone().cuda(), seg_label, 21, 255)

            pseudo_intersection, pseudo_union, pseudo_target = \
                pseudo_intersection.cpu().numpy(), pseudo_union.cpu().numpy(), pseudo_target.cpu().numpy()
            pseudo_intersection_meter.update(pseudo_intersection)
            pseudo_union_meter.update(pseudo_union)
            pseudo_target_meter.update(pseudo_target)

            pseduo_accuracy = sum(pseudo_intersection_meter.val) / (sum(pseudo_target_meter.val) + 1e-10)

            if (iter % 50):
                print('Val | Iter:[{}/{}]'
                      'Pseudo_accuracy {pseudo_accuracy: .4f} '
                      .format(iter, len(val_data_loader),
                              pseudo_accuracy=pseduo_accuracy))
            if (torch.sum(label[0]) == 3 and len(visual_boarder) < 30):
                p_bg_label=pseudo_bg_label[0,0]
                p_bg_pre=bg_pre[0,0]
                p_bg_pre=(p_bg_pre-p_bg_pre.min()-1e-5)/(p_bg_pre.max()+1e-5)
                #threshold=torch.quantile(p_bg_map,0.2)

                #label=(p_bg_map>0.20).bool()
                visual_boarder.append(
                    [images[0].cpu(),p_bg_label.cpu().numpy(),p_bg_pre.cpu().numpy(),
                     pseudo_label[0].cpu().numpy(),reliable_pseudo_label[0].cpu().numpy(), seg_label[0].cpu().numpy()])

        val_img = []

        for rgb,p_bg_label,p_bg_pre,ps_label,r_ps_label, gt_label in visual_boarder:
            rgb_q1 = rgb.clone()
            #rgb_q2 = rgb.clone()
            # rgb_q3 = rgb.clone()
            #rgb_q4=rgb.clone()
            mask=(r_ps_label==255)
            rgb = restore_transform(rgb)
            [ps_label,r_ps_label, gt_label] = [mytool.colorize_mask(x, palette) for x in [ps_label, r_ps_label,gt_label]]
            #r_ps_label[:,mask>0]=255
            #bg_pre = Image.fromarray(p_bgr* 255)
            #print((np.asarray(ps_label)!=np.asarray(r_ps_label)).sum())
            p_bg_pre = Image.fromarray(p_bg_pre * 255)
            #
            #
            #
            rgb_q1[:, p_bg_label > 0] = 0
            # rgb_q2[:, qy2 > 0] = 0
            # rgb_q3[:, qy3 > 0] = 0
            #rgb_q4[:,p_bgr>0]=0
            rgb_q1 = restore_transform(rgb_q1)
            # rgb_q2 = restore_transform(rgb_q2)
            # rgb_q3 = restore_transform(rgb_q3)
           # rgb_q4 = restore_transform(rgb_q4)
            #
            [rgb, rgb_q1,p_bg_pre,ps_label,r_ps_label, gt_label] = [x.convert('RGB') for x in
                                                                      [rgb,rgb_q1,p_bg_pre,ps_label,r_ps_label, gt_label]]
            #assert (np.asarray(r_ps_label) != np.asarray(ps_label)).any()
            [rgb,rgb_q1,p_bg_pre,ps_label,r_ps_label, gt_label] = [viz_transform(x) for x in
                                                                      [rgb,rgb_q1,p_bg_pre,ps_label,r_ps_label, gt_label]]
            #
            r_ps_label[:,torch.from_numpy(mask)>0]=255
            val_img.extend([rgb,rgb_q1,p_bg_pre,ps_label,r_ps_label, gt_label])
        val_img = torch.stack(val_img, 0)
        #
        val_img = make_grid(val_img.cpu(), nrow=6, padding=5)
        writer.add_image(f'seg/val', val_img, 0)

        pseudo_iou_class = pseudo_intersection_meter.sum / (pseudo_union_meter.sum + 1e-10)
        pseudo_accuracy_class = pseudo_intersection_meter.sum / (pseudo_target_meter.sum + 1e-10)
        pseudo_mIoU = np.mean(pseudo_iou_class)
        pseudo_mAcc = np.mean(pseudo_accuracy_class)
        pseudo_allAcc = sum(pseudo_intersection_meter.sum) / (sum(pseudo_target_meter.sum) + 1e-10)

        # writer.add_scalar(f'val/seg_loss1',seg_loss1_meter.val,current_epoch)
        writer.add_scalar(f'val/mIoU_pseudo', pseudo_mIoU, 0)

        print('Val result for pseudo labels: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(pseudo_mIoU, pseudo_mAcc,
                                                                                            pseudo_allAcc))

        print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')