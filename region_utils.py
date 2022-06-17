import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import random
from sklearn.cluster import KMeans

class Bank_contrast_loss(nn.Module):

    def __init__(self,dim=128,queue_size=50000,nbr_query=128,nbr_negative=256,temp=0.5):
        super(Bank_contrast_loss, self).__init__()
        self.dim=dim
        self.maximum_limits=queue_size
        self.queue_ptr=0## pointer of the queue
        self.memo_bank=torch.zeros(0,dim)
        self.nbr_query=nbr_query
        self.nbr_negative=nbr_negative
        self.temp=temp

    @torch.no_grad()
    def dequeue_and_enqueue(self,feas,mask,clustering=True):
        # gather keys before updating queue
        feas =feas.detach().clone().cpu()
        mask=mask.detach().clone().cpu()
        batch,hw,dim=feas.shape
        if clustering:
            self.nbr_cluster=8
            keys=[]
            for i in range(batch):
                single_img_keys=feas[i,mask[i]]
                if single_img_keys.shape[0]>self.nbr_cluster:
                    cluster=KMeans(n_clusters=self.nbr_cluster).fit(single_img_keys.numpy())
                    tmp_keys=torch.from_numpy(cluster.cluster_centers_)
                    #print(tmp_keys.shape)
                else:## in some rare cases, the nbr. of bg.keys is less than the nbr of clusters
                    tmp_keys=single_img_keys.detach()
                keys.append(tmp_keys)
            keys=torch.cat(keys,dim=0)
        else:

            feas=feas.view(-1,dim)
            mask=mask.view(-1)
            keys=feas[mask]
        batch_size=keys.shape[0]
        self.memo_bank = torch.cat((self.memo_bank, keys.cpu()), dim=0)
        if self.memo_bank.shape[0] >= self.maximum_limits:
            self.memo_bank= self.memo_bank[-self.maximum_limits:, :]
            self.queue_ptr = self.maximum_limits
        else:
            self.queue_ptr = (self.queue_ptr + batch_size) % self.maximum_limits  # move pointer

    def forward(self,feas,mask,pseudo_label, indicator,f_fea,prototypes):
        if feas.shape[0]>0:
            self.dequeue_and_enqueue(feas,mask)
        pseudo_label=pseudo_label.view(-1)
        device = f_fea.device
        reco_loss = torch.tensor(0.0).to(device)
        if self.memo_bank.shape[0]>0:
            valid_classes=torch.unique(pseudo_label)
            valid_classes=valid_classes[valid_classes!=20]
        if len(valid_classes)==0:
            return torch.tensor(0.0).to(device)
        else:
            for i in valid_classes:
                seg_hard_mask=(pseudo_label==i) & (indicator==255)
                single_seg_hard_fea=f_fea[seg_hard_mask]
                if seg_hard_mask.sum():
                    negative_fea_all = self.memo_bank.to(device)
                else:
                    continue
                # if seg_hard_mask.sum()>=self.nbr_query:
                #     seg_hard_idx = torch.randint(
                #         seg_hard_mask.sum(), size=(self.nbr_query,))
                #     nbr_query=self.nbr_query
                # else:
                seg_hard_idx = torch.arange(seg_hard_mask.sum())
                nbr_query=seg_hard_mask.sum()
                #print(seg_hard_mask.sum())
                anchor_fea=single_seg_hard_fea[seg_hard_idx]  ##
                postive_fea = prototypes[i]
                with torch.no_grad():
                    negative_index=[]

                    for sample in range(nbr_query):
                        negative_index+=np.random.randint(low=0,high=negative_fea_all.shape[0],size=self.nbr_negative).tolist()
                    negative_fea=negative_fea_all[negative_index].reshape(nbr_query,self.nbr_negative,self.dim)
                    postive_fea=postive_fea.repeat(nbr_query,1,1)
                    all_feat=torch.cat((postive_fea,negative_fea),dim=1)
                seg_logits=torch.cosine_similarity(anchor_fea.unsqueeze(1),all_feat,dim=1)

                reco_loss=reco_loss+F.cross_entropy(seg_logits/self.temp,torch.zeros(nbr_query).long().cuda())
            return reco_loss/ (len(valid_classes))





def label_onehot(inputs, nbr_category):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    # print(inputs.shape)
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, nbr_category, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def compute_cam_up(cam, label, b, w, h):
    cam = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    cam_up = cam.clone()
    cam_up = cam_up * label.clone().view(b, 20, 1, 1)
    cam_up = cam_up.cpu().data.numpy()
    return cam_up


def build_pseudo_label(cam, ori_cam,label, height, width, un_rate):
    pos_cam_up = compute_cam_up(F.relu(cam), label, cam.shape[0], height, width, )
    neg_cam_up = compute_cam_up(F.relu(-ori_cam), label, cam.shape[0], height, width, )
    # indicator=-torch.sum(torch.softmax(cam_up,dim=1)*(torch.log(torch.softmax(cam_up,dim=1))+1e-10),dim=1).cuda()
    # cam_up=cam_up.data.cpu().numpy()
    pseudo_label = np.zeros((cam.shape[0], height, width))
    # print(pseudo_label.shape)
    #print(ori_cam.shape)
    no_sure_mask = np.zeros((cam.shape[0], height, width))
    pseudo_bgr_label = np.zeros((cam.shape[0], height, width))
    for i in range(cam.shape[0]):
        cam_up_single = pos_cam_up[i]
        cam_up_single[cam_up_single< 0] = 0
        cam_max = np.max(cam_up_single, (1,2), keepdims=True)
        cam_min = np.min(cam_up_single, (1,2), keepdims=True)
        cam_up_single[cam_up_single < cam_min+1e-5] = 0
        norm_cam = (cam_up_single-cam_min-1e-5) / (cam_max - cam_min + 1e-5)
        p_label, no_sure = compute_seg_label(norm_cam, un_rate)
        pseudo_label[i] = p_label
        no_sure_mask[i] = no_sure
        cam_up_single = neg_cam_up[i]
        norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)
        norm_cam = np.sum(norm_cam, axis=0)
        threshold = np.percentile(norm_cam.flatten(), 70)
        pseudo_bgr_label[i] = norm_cam>threshold
    pseudo_label = torch.from_numpy(pseudo_label).long().cuda()
    label_mask = label_onehot(pseudo_label, 21).long().cuda()
    no_sure_mask = torch.from_numpy(no_sure_mask).long().cuda()
    pseudo_bgr_label = torch.from_numpy(pseudo_bgr_label).float().cuda()
    return pseudo_label, label_mask, no_sure_mask, pseudo_bgr_label


def compute_seg_label(norm_cam, percentile=0.4):
    # fore_mask = np.zeros_like(norm_cam)
    #print(norm_cam.shape)
    bg_score = np.ones((1,norm_cam.shape[-2],norm_cam.shape[-1]))*0.26
    #bg_score = np.power(1 - np.max(norm_cam, 0), 32)
    #bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, norm_cam))
    #print(cam_all.shape)
    pseudo_label = np.argmax(cam_all, 0)
    single_img_classes = np.unique(pseudo_label)
    cam_sure_region = np.zeros_like(pseudo_label, dtype=bool)
    for class_i in single_img_classes:
        if class_i != 0:
            class_not_region = (pseudo_label != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            cam_class_order = cam_class[cam_class > 0.1]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * 0.4)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        else:
            class_not_region = (pseudo_label != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            # assert not np.isnan(cam_class).any()
            class_sure_region = (cam_class > 0.8)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region

    return pseudo_label, cam_not_sure_region


# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def compute_reco_loss(fore_rep,no_sure_mask, pseudo_labels, fore_mask, strong_threshold=0.80, temp=0.5,
                      num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h = fore_rep.shape
    device = fore_rep.device
    # pseudo_labels=torch.from_numpy(pseudo_labels).cuda(device)
    pre_class = torch.unique(pseudo_labels).long()[1:]

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    fore_rep = fore_rep.permute(0, 2, 3, 1)
    #back_rep = back_rep.permute(0, 2, 3, 1)
    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    # print(no_sure_mask.shape)
    # print(fore_mask.shape)

    # valid_pixel_seg = fore_mask[:,0].bool()  # select binary mask for i-th class
    # rep_mask_hard = no_sure_mask.bool() * valid_pixel_seg # select hard queries
    # seg_proto_list.append(torch.mean(back_rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
    # seg_feat_all_list.append(fore_rep[valid_pixel_seg.bool()])
    # seg_feat_hard_list.append(fore_rep[rep_mask_hard])
    # seg_num_list.append(int(valid_pixel_seg.sum().item()))

    for i in pre_class:
        valid_pixel_seg = fore_mask[:, i].bool()  # select binary mask for i-th class
        # if i==0:
        # rep_mask_hard =valid_pixel_seg

        # else:
        rep_mask_hard = no_sure_mask.bool() * valid_pixel_seg  # select hard queries
        # print(rep_mask_hard.flatten().sum())
        seg_proto_list.append(torch.mean(fore_rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
        seg_feat_all_list.append(fore_rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(fore_rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        reco_fg_loss=torch.tensor(0.0)
    else:
        reco_fg_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)

                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                # print(proto_prob)

                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i + 1:] + seg_num_list[:i]

                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i + 1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_fg_loss = reco_fg_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        reco_fg_loss=reco_fg_loss/valid_seg

    # bg_mask = (pseudo_labels == 0).bool()
    # if (no_sure_mask.bool() * bg_mask).sum()>0:
    #     bg_proto = torch.mean(back_rep[bg_mask], dim=0, keepdim=True)
    #
    #     bg_feat_hard = back_rep[no_sure_mask.bool() * bg_mask]
    #     seg_feat_all_list = torch.cat(seg_feat_all_list)  ## foregorund features
    #
    #     bg_seg_hard_idx = torch.randint(len(bg_feat_hard), size=(num_queries,))
    #     bg_anchor_feat_hard = bg_feat_hard[bg_seg_hard_idx]
    #     bg_anchor_feat = bg_anchor_feat_hard
    #     with torch.no_grad():
    #         negative_index = []
    #         for i in range(num_queries):
    #             #print(bg_feat_all.shape)
    #             negative_index += np.random.randint(low=0,
    #                                                 high=seg_feat_all_list.shape[0],
    #                                                 size=num_negatives).tolist()
    #         negative_fg_feat = seg_feat_all_list[negative_index].reshape(num_queries, num_negatives, num_feat)
    #         positive_bg_feat=bg_proto[0].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
    #         bg_all_feat=torch.cat((negative_fg_feat,positive_bg_feat),dim=1)
    #     seg_logits = torch.cosine_similarity(bg_anchor_feat.unsqueeze(1),bg_all_feat, dim=2)
    #     reco_fg_bg_loss = F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
    #
    # else:
    #     reco_fg_bg_loss=torch.tensor(0.0)
    return reco_fg_loss#, reco_fg_bg_loss


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j + 1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index


def query_visualization(no_sure_mask, pseudo_labels, fore_mask):
    pre_class = torch.unique(pseudo_labels)[1:]

    seg_feat_hard_list = []
    # threshold = torch.quantile(indicator.flatten(),0.10)
    # print(indicator.shape)
    seg_feat_hard_list.append((pseudo_labels == 0).bool())  # select hard querie
    for i in pre_class:
        # print(threshold)

        # valid_pixel_seg = fore_mask[i-1]  # select binary mask for i-th class
        rep_mask_hard = no_sure_mask * ((pseudo_labels == i).bool())  # select hard queries
        seg_feat_hard_list.append(rep_mask_hard)
    return seg_feat_hard_list






