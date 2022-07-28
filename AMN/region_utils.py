import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import random
from sklearn.cluster import KMeans


def get_background_label(cam, label):
    cam_up = F.relu(cam[:, 1:]) * (label.unsqueeze(-1).unsqueeze(-1))
    # print(cam_up.shape)
    norm_cam = cam_up / (F.adaptive_max_pool2d(cam_up, (1, 1)) + 1e-5)
    norm_cam = torch.sum(norm_cam, dim=1)
    pseudo_bg_label = (norm_cam < 0.05).float()

    return pseudo_bg_label


def get_reliable_pseudo_label(pseudo_label, norm_scores, prob, rate=0.8):
    batch, nbr_class, h, w = norm_scores.shape
    entropy_mask = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    cam_sure_region = torch.zeros_like(pseudo_label, dtype=bool)
    for i in range(batch):
        single_img_classes = torch.unique(pseudo_label[i])

        single_img_classes = single_img_classes[single_img_classes != 0]
        for class_i in single_img_classes:
            class_not_region = (pseudo_label[i] != class_i)
            cam_class = norm_scores[i, class_i, :, :]
            cam_class[class_not_region] = 0
            cam_class_order = cam_class[cam_class > 0.05]
            cam_class_order, __ = torch.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * rate)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)
            cam_sure_region[i] = torch.logical_or(cam_sure_region[i], class_sure_region)
    cam_not_sure_region = ~cam_sure_region
    reliable_pseudo_label = pseudo_label.clone()
    entropy_threshold=torch.quantile(entropy_mask.view(batch,-1),0.25, dim=1, keepdim=True)
    high_entropy_mask=(entropy_mask>entropy_threshold.unsqueeze(-1))
    reliable_pseudo_label[cam_not_sure_region&high_entropy_mask]=255
    return reliable_pseudo_label

def get_contrast_loss(logits, f_proj, b_proj,reco_loss):
    n1, c1, h1, w1 = logits.shape

    # bg_score = torch.ones((n1, 1)).cuda()
    with torch.no_grad():
        f_fea = f_proj.detach()
        b_fea = b_proj.detach()
        c_fea = f_fea.shape[1]

        logits = F.softmax(logits, dim=1)
        # print(up_logits.shape)
        max = torch.max(logits.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        min = torch.min(logits.view(n1, c1, -1), dim=-1)[0].view(n1, c1, 1, 1)
        # print(min.shape)
        # print(max.shape)
        norm_cam = (logits - min) / (max - min + 1e-5)

        pseudo_label = norm_cam.argmax(dim=1, keepdim=False)
        reliable_pseudo_mask = get_reliable_pseudo_label(pseudo_label, norm_cam, logits, rate=0.4)

        n_sc, c_sc, h_sc, w_sc = norm_cam.shape

        f_fea = f_fea.permute(0, 2, 3, 1).reshape(-1, c_fea)  # (nhw, 128)
        b_fea = b_fea.permute(0, 2, 3, 1).contiguous().reshape(n_sc, h_sc * w_sc, c_fea)  # (nhw, 128)
        top_values, top_indices = torch.topk(norm_cam.transpose(0, 1).reshape(c_sc, -1),
                                             k=h_sc * w_sc // 32, dim=-1)
        prototypes = torch.zeros(c_sc, c_fea).cuda()  # [21, 128]
        for i in range(c_sc):
            top_fea = f_fea[top_indices[i]]
            prototypes[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
    n_f, c_f, h_f, w_f = f_proj.shape
    #f_proj = f_proj.permute(0, 2, 3, 1).contiguous().reshape(n_f * h_f * w_f, c_f)
    #b_proj = b_proj.permute(0, 2, 3, 1).contiguous().reshape(n_f, h_f * w_f, c_f)

    pseudo_label=pseudo_label.view(n_f,-1)
    no_sure_mask = (reliable_pseudo_mask==255).bool()

    loss_fg_loss,loss_bg_loss=reco_loss(f_fea, no_sure_mask, pseudo_label, prototypes, b_fea)

    loss_fg_loss=0.1*loss_fg_loss
    loss_bg_loss=0.05*loss_bg_loss

    return loss_fg_loss,loss_bg_loss


class Compled_reco(nn.Module):
    def __init__(self, dim=128, nbr_negative=256, temp=0.5, queue_size=100000):
        super(Compled_reco, self).__init__()
        self.dim = dim
        if queue_size > 0:
            self.bg_maximum_limits = queue_size
            self.compute_bg = True
            self.queue_ptr = 0  ## pointer of the queue
            self.memo_bank = torch.zeros(0, dim)
        self.nbr_negative = nbr_negative
        self.temp = temp

    @torch.no_grad()  ## update the bg_bank
    def dequeue_and_enqueue(self, feas, mask, clustering=True):
        # gather keys before updating queue
        feas = feas.detach().clone().cpu()
        mask = mask.detach().clone().cpu()
        batch, hw, dim = feas.shape
        if clustering:
            self.nbr_cluster = 8
            keys = []
            for i in range(batch):
                single_img_keys = feas[i, mask[i]]
                if single_img_keys.shape[0] > self.nbr_cluster:
                    cluster = KMeans(n_clusters=self.nbr_cluster).fit(single_img_keys.numpy())
                    tmp_keys = torch.from_numpy(cluster.cluster_centers_)
                    # print(tmp_keys.shape)
                else:  ## in some rare cases, the nbr. of bg.keys is less than the nbr of clusters
                    tmp_keys = single_img_keys.detach()
                keys.append(tmp_keys)
            keys = torch.cat(keys, dim=0)
        else:

            feas = feas.view(-1, dim)
            mask = mask.view(-1)
            keys = feas[mask]
        batch_size = keys.shape[0]
        self.memo_bank = torch.cat((self.memo_bank, keys.cpu()), dim=0)
        if self.memo_bank.shape[0] >= self.bg_maximum_limits:
            self.memo_bank = self.memo_bank[-self.bg_maximum_limits:, :]
            self.queue_ptr = self.bg_maximum_limits
        else:
            self.queue_ptr = (self.queue_ptr + batch_size) % self.bg_maximum_limits  # move pointer

    @torch.no_grad()
    def bank_sampling(self,nbr_query, device):
        negative_index = []
        negative_bg_fea_all = self.memo_bank.to(device)
        #print(self.memo_bank.shape)
        for sample in range(nbr_query):
            negative_index += np.random.randint(low=0, high=negative_bg_fea_all.shape[0], size=self.nbr_negative).tolist()
        negative_bg_fea = negative_bg_fea_all[negative_index].reshape(nbr_query, self.nbr_negative, self.dim)
        return negative_bg_fea

    def forward(self,fore_rep, no_sure_mask, pseudo_labels, prototype_list, bg_feas=None):
        device = fore_rep.device
        fg_loss = torch.tensor(0.0).to(device)
        fg_bg_loss = torch.tensor(0.0).to(device)

        __, dim = fore_rep.shape
        no_sure_mask = no_sure_mask.view(-1)
        pre_class = torch.unique(pseudo_labels).long()  # get aviable classes in the current batch
        pre_class = pre_class[pre_class != 0]
        current_prototype_list = prototype_list[pre_class]
        seg_feat_all_list = []
        seg_feat_hard_list = []
        seg_num_list = []


        if bg_feas.shape[0] > 0:
            bg_mask = (pseudo_labels == 0).bool()
            #print(bg_mask.shape)
            self.dequeue_and_enqueue(bg_feas, bg_mask)

        pseudo_labels=pseudo_labels.view(-1)
        for i in pre_class:
            valid_pixel_seg = (pseudo_labels == i).bool()  # select binary mask for i-th class
            rep_mask_hard = no_sure_mask.bool() * valid_pixel_seg  # select hard queries
            seg_feat_all_list.append(fore_rep[valid_pixel_seg.bool()])
            seg_feat_hard_list.append(fore_rep[rep_mask_hard])
            seg_num_list.append(int(valid_pixel_seg.sum().item()))

        # compute regional contrastive loss
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                anchor_feat_hard = seg_feat_hard_list[i]
                nbr_query = anchor_feat_hard.shape[0]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(current_prototype_list[seg_mask[0]].unsqueeze(0),
                                                    current_prototype_list[seg_mask[1:]], dim=1)

                proto_prob = torch.softmax(proto_sim / self.temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                # print(proto_prob)

                samp_class = negative_dist.sample(sample_shape=[nbr_query, self.nbr_negative])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i + 1:] + seg_num_list[:i]

                negative_fg_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_fg_feat_all = torch.cat(seg_feat_all_list[i + 1:] + seg_feat_all_list[:i])
                negative_fg_feat = negative_fg_feat_all[negative_fg_index].reshape(nbr_query,self.nbr_negative, dim)

                # combine positive and negative keys
                positive_feat = current_prototype_list[i].unsqueeze(0).unsqueeze(0).repeat(nbr_query, 1, 1)
                all_fg_feat = torch.cat((positive_feat, negative_fg_feat), dim=1)

                if self.compute_bg:
                    sampled_bg_fea = self.bank_sampling(nbr_query, device)
                    all_fg_bg_fea = torch.cat((positive_feat, sampled_bg_fea), dim=1)
            fg_seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_fg_feat, dim=2)
            fg_loss = fg_loss + F.cross_entropy(fg_seg_logits / self.temp, torch.zeros(nbr_query).long().to(device))
            if self.compute_bg:
                bg_seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_fg_bg_fea, dim=1)
                fg_bg_loss = fg_bg_loss + F.cross_entropy(bg_seg_logits / self.temp, torch.zeros(nbr_query).long().to(device))
        fg_loss = fg_loss / valid_seg
        if self.compute_bg:
            fg_bg_loss = fg_bg_loss / valid_seg
        return fg_loss, fg_bg_loss


class Bank_contrast_loss(nn.Module):

    def __init__(self, dim=128, queue_size=50000, nbr_query=128, nbr_negative=256, temp=0.5):
        super(Bank_contrast_loss, self).__init__()
        self.dim = dim
        self.maximum_limits = queue_size
        self.queue_ptr = 0  ## pointer of the queue
        self.memo_bank = torch.zeros(0, dim)
        self.nbr_query = nbr_query
        self.nbr_negative = nbr_negative
        self.temp = temp

    @torch.no_grad()
    def dequeue_and_enqueue(self, feas, mask, clustering=True):
        # gather keys before updating queue
        feas = feas.detach().clone().cpu()
        mask = mask.detach().clone().cpu()
        batch, hw, dim = feas.shape
        if clustering:
            self.nbr_cluster = 8
            keys = []
            for i in range(batch):
                single_img_keys = feas[i, mask[i]]
                if single_img_keys.shape[0] > self.nbr_cluster:
                    cluster = KMeans(n_clusters=self.nbr_cluster).fit(single_img_keys.numpy())
                    tmp_keys = torch.from_numpy(cluster.cluster_centers_)
                    # print(tmp_keys.shape)
                else:  ## in some rare cases, the nbr. of bg.keys is less than the nbr of clusters
                    tmp_keys = single_img_keys.detach()
                keys.append(tmp_keys)
            keys = torch.cat(keys, dim=0)
        else:

            feas = feas.view(-1, dim)
            mask = mask.view(-1)
            keys = feas[mask]
        batch_size = keys.shape[0]
        self.memo_bank = torch.cat((self.memo_bank, keys.cpu()), dim=0)
        if self.memo_bank.shape[0] >= self.maximum_limits:
            self.memo_bank = self.memo_bank[-self.maximum_limits:, :]
            self.queue_ptr = self.maximum_limits
        else:
            self.queue_ptr = (self.queue_ptr + batch_size) % self.maximum_limits  # move pointer

    def forward(self, feas, mask, pseudo_label, indicator, f_fea, prototypes):
        if feas.shape[0] > 0:
            self.dequeue_and_enqueue(feas, mask)
        pseudo_label = pseudo_label.view(-1)
        device = f_fea.device
        reco_loss = torch.tensor(0.0).to(device)
        if self.memo_bank.shape[0] > 0:
            valid_classes = torch.unique(pseudo_label)
            valid_classes = valid_classes[valid_classes != 0]
        if len(valid_classes) == 0:
            return torch.tensor(0.0).to(device)
        else:
            for i in valid_classes:
                seg_hard_mask = (pseudo_label == i) & (indicator == 255)
                single_seg_hard_fea = f_fea[seg_hard_mask]
                if seg_hard_mask.sum():
                    negative_fea_all = self.memo_bank.to(device)
                else:
                    continue
                seg_hard_idx = torch.arange(seg_hard_mask.sum())
                nbr_query = seg_hard_mask.sum()
                # print(seg_hard_mask.sum())
                anchor_fea = single_seg_hard_fea[seg_hard_idx]  ##
                postive_fea = prototypes[i]
                with torch.no_grad():
                    negative_index = []

                    for sample in range(nbr_query):
                        negative_index += np.random.randint(low=0, high=negative_fea_all.shape[0],
                                                            size=self.nbr_negative).tolist()
                    negative_fea = negative_fea_all[negative_index].reshape(nbr_query, self.nbr_negative, self.dim)
                    postive_fea = postive_fea.repeat(nbr_query, 1, 1)
                    all_feat = torch.cat((postive_fea, negative_fea), dim=1)
                seg_logits = torch.cosine_similarity(anchor_fea.unsqueeze(1), all_feat, dim=1)

                reco_loss = reco_loss + F.cross_entropy(seg_logits / self.temp, torch.zeros(nbr_query).long().cuda())
            return reco_loss / (len(valid_classes))


# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def compute_fg_reco_loss(fore_rep, no_sure_mask, pseudo_labels, prototype_list, temp=0.5, num_negatives=256):
    device = fore_rep.device
    __, dim = fore_rep.shape
    no_sure_mask = no_sure_mask.view(-1)

    pre_class = torch.unique(pseudo_labels).long()  # get aviable classes in the current batch

    pre_class = pre_class[pre_class != 0]
    current_prototype_list = prototype_list[pre_class]
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    for i in pre_class:
        valid_pixel_seg = (pseudo_labels == i).bool()  # select binary mask for i-th class
        rep_mask_hard = no_sure_mask.bool() * valid_pixel_seg  # select hard queries
        seg_feat_all_list.append(fore_rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(fore_rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        reco_fg_loss = torch.tensor(0.0)
    else:
        reco_fg_loss = torch.tensor(0.0)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                anchor_feat_hard = seg_feat_hard_list[i]
                nbr_query = anchor_feat_hard.shape[0]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(current_prototype_list[seg_mask[0]].unsqueeze(0),
                                                    current_prototype_list[seg_mask[1:]], dim=1)

                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                # print(proto_prob)

                samp_class = negative_dist.sample(sample_shape=[nbr_query, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i + 1:] + seg_num_list[:i]

                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i + 1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(nbr_query, num_negatives, dim)

                # combine positive and negative keys
                positive_feat = current_prototype_list[i].unsqueeze(0).unsqueeze(0).repeat(nbr_query, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_fg_loss = reco_fg_loss + F.cross_entropy(seg_logits / temp, torch.zeros(nbr_query).long().to(device))
        reco_fg_loss = reco_fg_loss / valid_seg

    return reco_fg_loss


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j + 1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index



