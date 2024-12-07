# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 18:42
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : custom_loss.py

from .loss import LOSS_REGISTRY, LossBase

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from distutils.version import LooseVersion

@LOSS_REGISTRY.register()
class ContrastiveLoss(nn.Module):
    def __init__(self, cfg, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs):
        """
        Args:
            x1, x2: FloatTensors, (batch_size, num_features) - the feature vectors of two samples
            y: FloatTensor, (batch_size,) - the similarity labels (1 for similar, 0 for dissimilar)
        """
        x1 = outputs["pred_embedding"]
        x2 = outputs["edit_embedding"]
        y = outputs["sim_lable"]
        y = y.view(-1, 1)

        # Compute the Euclidean distance
        dist = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1)).view(-1, 1)

        # Compute the losses for similar and dissimilar samples
        loss_similar = torch.sum(y * dist ** 2, dim=0)
        loss_dissimilar = torch.sum((1 - y) * torch.relu(self.margin - dist) ** 2, dim=0)

        # Compute the final loss
        loss = 0.5 * (loss_similar + loss_dissimilar)

        return loss.squeeze()


@LOSS_REGISTRY.register()
class MaxMarginRankingLoss(LossBase):
    def __init__(self, cfg):
        super(MaxMarginRankingLoss, self).__init__(cfg)
        cfg = cfg.LOSS.MaxMarginRankingLoss

        self.margin = cfg.MARGIN
        self.n_pair = cfg.N_PAIR
        self.batch_size = cfg.BATCH_SIZE
        easy_negative_rate = 1 - cfg.HARD_NEGATIVE_RATE
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = cfg.NEGATIVE_WEIGHT
        if cfg.N_PAIR > 1 and cfg.BATCH_SIZE > 1:
            alpha = easy_negative_rate / ((cfg.BATCH_SIZE - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((cfg.N_PAIR, cfg.N_PAIR)))
            mm_mask = torch.tensor(mm_mask) * (cfg.BATCH_SIZE * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def _forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

    def forward(self, inputs, outputs):
        sim_matrix = outputs["sim_matrix"]
        return self._forward(sim_matrix)


@LOSS_REGISTRY.register()
class LabelSmoothingLoss(LossBase):
    def __init__(self, cfg):
        label_smoothing = 0.1
        self.tgt_vocab_size = 2
        ignore_index = 0
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(cfg)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (self.tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((self.tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, target, output):
        output = output["prediction_scores"]
        output = output.view(-1, self.tgt_vocab_size)
        target = target['input_labels'].reshape(-1).long()
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1).to(target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


@LOSS_REGISTRY.register()
class MILNCELoss(LossBase):
    def __init__(self, cfg):
        super(MILNCELoss, self).__init__(cfg)
        cfg = cfg.LOSS.MILNCELoss

        self.batch_size = cfg.BATCH_SIZE
        self.n_pair = cfg.N_PAIR
        torch_v = LooseVersion(torch.__version__)
        self.bool_dtype = torch.bool if torch_v >= LooseVersion("1.3") else torch.uint8

    def _forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair // 2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

    def forward(self, inputs, outputs):
        sim_matrix = outputs["sim_matrix"]
        return self._forward(sim_matrix)


@LOSS_REGISTRY.register()
class ReconstructLoss(LossBase):
    def forward(self, inputs, outputs):
        iframe_pred, motion_vector_pred, iframe_mask, motion_vector_mask, iframe_patched, motion_vector_patched = \
            outputs["iframe_pred"], outputs["motion_vector_pred"], \
                outputs["iframe_mask"], outputs["motion_vector_mask"], \
                outputs["iframe_patched"], outputs["motion_vector_patched"]

        iframe_loss = torch.mean((iframe_pred - iframe_patched) ** 2, dim=-1)
        iframe_loss = (iframe_loss * (1 - iframe_mask)).sum() / (1 - iframe_mask).sum()
        motion_vector_loss = torch.mean((motion_vector_pred - motion_vector_patched) ** 2, dim=-1)
        motion_vector_loss = (motion_vector_loss * (1 - motion_vector_mask)).sum() / (1 - motion_vector_mask).sum()

        return iframe_loss + motion_vector_loss


@LOSS_REGISTRY.register()
class VideoClassificationLoss(LossBase):
    def forward(self, inputs, outputs):
        logits = outputs["cls_logits"]
        labels = inputs["label"]
        return F.cross_entropy(input=logits, target=labels)


@LOSS_REGISTRY.register()
class ClassLabelSmoothingLoss(LossBase):
    def __init__(self, cfg):
        label_smoothing = 0.1
        self.tgt_vocab_size = 2
        assert 0.0 < label_smoothing <= 1.0
        super(ClassLabelSmoothingLoss, self).__init__(cfg)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (self.tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((self.tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs):
        logits = outputs["pred"]
        labels = outputs["sim_label"]
        logits = self.log_softmax(logits)
        model_prob = self.one_hot.repeat(labels.size(0), 1).to(labels.device)
        model_prob.scatter_(1, labels.unsqueeze(1), self.confidence)
        return F.kl_div(logits, model_prob, reduction="sum")


@LOSS_REGISTRY.register()
class EnhanceMotionLoss(LossBase):
    def forward(self, inputs, outputs):
        enhance_motion = torch.index_select(outputs["motion_feature"], dim=2, index=outputs["enhance_index"])
        return F.l1_loss(enhance_motion, outputs["bp_feature"])


@LOSS_REGISTRY.register()
class InfoNCELoss(nn.Module):
    def __init__(self, cfg, T=0.07, **kwargs):
        super(InfoNCELoss, self).__init__()
        self.T = T
        self.CE = nn.CrossEntropyLoss()

    def forward(self, outputs, labels=None):
        inputs_q = outputs["element_1"]
        inputs_k = outputs["element_2"]
        n = inputs_q.size(0)

        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())
        # print(sim_mat)
        # split the positive and negative pairs
        eyes_ = torch.eye(n).cuda()

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        # print(pos_sim.size(), neg_sim.size())
        # raise Exception

        l_pos = pos_sim.unsqueeze(dim=1).expand(n, 1)
        l_neg = neg_sim.reshape(n, n-1)

        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = self.CE(logits, labels)

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()

        return loss, mean_pos_sim, mean_neg_sim

@LOSS_REGISTRY.register()
class InfoNCELossv4(nn.Module):
    def __init__(self, cfg, T=0.07, **kwargs):
        super(InfoNCELossv4, self).__init__()
        self.T = T
        self.CE = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs, labels=None, mean_pos = None):
        inputs_q = outputs["element_1"]
        inputs_k = outputs["element_2"]
        n = inputs_q.size(0)

        normalized_inputs_q = F.normalize(inputs_q, p=2, dim=1) #/ torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = F.normalize(inputs_k, p=2, dim=1) #/ torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())
        sim_mat_2 = torch.matmul(normalized_inputs_q, normalized_inputs_q.t())
        sim_mat_3 = torch.matmul(normalized_inputs_k, normalized_inputs_k.t())
        # print(sim_mat[:10,:10], torch.triu(sim_mat_2,diagonal=1)[:10,:10], torch.triu(sim_mat_3,diagonal=1)[:10,:10])
        # split the positive and negative pairs
        eyes_ = torch.eye(n).cuda()
        n_eyes = 1. - eyes_

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        # print(pos_sim.size(), neg_sim.size())
        # raise Exception

        # l_pos = pos_sim.unsqueeze(dim=1).expand(n, 1)
        # l_neg = neg_sim.reshape(n, n-1)

        # logits = torch.cat([l_pos, l_neg], dim=1)
        logits_1 = torch.cat([sim_mat, sim_mat_2 * n_eyes], dim=1)
        logits_2 = torch.cat([sim_mat.t(), sim_mat_3 * n_eyes], dim=1)

        # apply temperature
        logits_1 /= self.T
        logits_2 /= self.T

        # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels = torch.arange(n, dtype=torch.long).cuda()
        if mean_pos is not None:
            for i in range(n):
                if sim_mat[i, labels[i]] < mean_pos:
                    labels[i] = -100

        loss = 0.5*self.CE(logits_1, labels) + 0.5*self.CE(logits_2, labels)

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()

        return loss, mean_pos_sim, mean_neg_sim



@LOSS_REGISTRY.register()
class InfoNCELoss_dyn(nn.Module):
    def __init__(self, cfg, temperature=0.07):
        super(InfoNCELoss_dyn, self).__init__()
        self.temperature = temperature

    def forward(self, outputs):
        if outputs["dyn_embedding"] == None:
            return 0, 0, 0

        labels = outputs["labels"]
        A = (outputs["element_1"] + outputs["element_2"]) / 2
        B = outputs["dyn_embedding"]

        # Normalize the features
        A = F.normalize(A, dim=1)
        B = F.normalize(B, dim=1)

        # Compute the dot product between A and B
        logits = torch.matmul(A, B.t()) / self.temperature

        # Compute the InfoNCE loss using the cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss, 0, 0
