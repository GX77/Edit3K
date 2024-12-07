# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 22:01
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : edit_trainer.py

import os
import logging
from tqdm import tqdm

import json
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler

from .build import TRAINER_REGISTRY, TrainerBase
from ..data.build import build_loader
from ..modeling.model import build_model
from ..modeling.optimizer import build_optimizer
from ..modeling.loss import build_loss


from ..modeling.bert.optimization import EMA
import itertools
from ..utils.train_utils import CudaPreFetcher
from ..utils.cal_sim import cosine_similarity_matrix
from ..utils.cal_top import  top_accuracy

logger = logging.getLogger(__name__)

def gather_object_multiple_gpu(list_object):
    """
    gather a list of something from multiple GPU
    :param list_object:
    """
    gathered_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_objects, list_object)
    return list(itertools.chain(*gathered_objects))

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def class_split(data, all_edit_name):
    edit = load_json(all_edit_name)
    type_list = []
    for name in data:
        id = name.split("_")[-1].split(".")[0]

        if id in set(edit['video_effect']):
            type_list.append(0)
        elif id in set(edit['transition']):
            type_list.append(1)
        elif id in set(edit['sticker']):
            type_list.append(2)
        elif id in set(edit['filter']):
            type_list.append(3)
        elif id in set(edit['video_animation']):
            type_list.append(4)
        elif id in set(edit['texts']):
            type_list.append(5)
        else:
            type_list.append(6)
            # print(id, "no edit category")

    return type_list
    
@TRAINER_REGISTRY.register()
class EditTrainer(TrainerBase):

    def __init__(self, cfg):
        super(EditTrainer, self).__init__(cfg)

    def build(self):
        cfg = self.cfg
        self.model = build_model(cfg)
        self.dataloader = build_loader(cfg, mode=("train", "test"))
        # update total step for optimizer
        gradient_accumulation_steps = cfg.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS
        epoch = cfg.TRAINER.TRAINER_BASE.EPOCH
        num_train_optimization_steps = (int(len(self.dataloader["train"]) + gradient_accumulation_steps - 1)
                                        / gradient_accumulation_steps) * epoch
        cfg.defrost()
        cfg.OPTIMIZER.BertAdam.t_total = num_train_optimization_steps
        cfg.freeze()
        self.optimizer, self.scheduler = self.prep_optimizer(
            cfg, self.model, coef_lr=1. if cfg.TRAINER.TRAINER_BASE.RESUME is not None else 0.1
        )
        self.ema = self.prep_ema(self.model)
        self.loss_func = build_loss(self.cfg)
        self.scaler = GradScaler() if self.enable_amp else None
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.debug("Model:\n%s", self.model)

    @staticmethod
    def prep_ema(model):
        ema_decay = 0.9999
        ema = EMA(ema_decay)
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema.register(name, p.data)
        return ema

    @staticmethod
    def prep_optimizer(cfg, model, coef_lr=1.):
        if hasattr(model, 'module'):
            model = model.module

        # https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if fpn.startswith("clip"):
                    # clip model will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("bias") or pn.endswith("embedding") or pn.endswith("attention_mask"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or pn.endswith('in_proj_weight')) \
                        and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        decay_param = [param_dict[pn] for pn in sorted(list(decay))]
        no_decay_vid_param = [param_dict[pn] for pn in sorted(list(no_decay)) if pn.startswith("vid")]
        no_decay_no_vid_param = [param_dict[pn] for pn in sorted(list(no_decay)) if not pn.startswith("vid")]
        logger.debug("Parameter group 0: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(decay))]))
        logger.debug("Parameter group 1: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(no_decay)) if pn.startswith("vid")]))
        logger.debug("Parameter group 2: %s",
                     "\n   " + "\n   ".join([pn for pn in sorted(list(no_decay)) if not pn.startswith("vid")]))

        optimizer_grouped_parameters = [
            {"params": decay_param},
            {"params": no_decay_vid_param, "weight_decay": 0.0, "lr": cfg.TRAINER.EditTrainer.VID_LR},
            {"params": no_decay_no_vid_param, "weight_decay": 0.0}
        ]

        warmup_epoch = int(cfg.OPTIMIZER.BertAdam.warmup * cfg.TRAINER.TRAINER_BASE.EPOCH)
        optimizer = build_optimizer(cfg, optimizer_grouped_parameters)
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 if epoch < warmup_epoch
            else cfg.TRAINER.EditTrainer.LR_DECAY_GAMMA ** (epoch - warmup_epoch)
        )

        return optimizer, scheduler

    @torch.no_grad()
    def _on_test_epoch(self):
        self.model.eval()
        dataloader = self.dataloader["test"]
        dataloader = tqdm(dataloader, desc=f"Eval epoch {self.epoch + 1}", dynamic_ncols=True,
                          disable=dist.is_initialized() and dist.get_rank() != 0)
        if torch.cuda.is_available():
            dataloader = CudaPreFetcher(dataloader)  # move to GPU
        mean_pos_sim = []
        mean_neg_sim = []
        row = []
        column = []
        edit_video_1 = []
        edit_video_2 = []
        for inputs in dataloader:
            outputs = self.model(inputs)
            loss_meta = self.loss_func(outputs)['InfoNCELossv4']
            mean_pos_sim.append(loss_meta[1])
            mean_neg_sim.append(loss_meta[2])
            row.append(outputs['element_1'].cpu())
            column.append(outputs['element_2'].cpu())
            edit_video_1 += inputs['v1_path']
            edit_video_2 += inputs['v2_path']

        if dist.is_initialized():
            row = gather_object_multiple_gpu(row)
            row = torch.cat(row, dim=0)
            column = gather_object_multiple_gpu(column)
            column = torch.cat(column, dim=0)
            edit_video_1 = gather_object_multiple_gpu(edit_video_1)
            edit_video_2 = gather_object_multiple_gpu(edit_video_2)

        if not dist.is_initialized() or dist.get_rank() == 0:
            gt_category_list = [i for i in range(len(row))]  # len(sim_matrix)
            sim_matrix = cosine_similarity_matrix(row, column)
            all_top1 = []
            all_top10 = []
            type_list = class_split(edit_video_1, "./edit_id2type.json")
            print("#### Result ####")
            type_name_list = ['video_effect', 'transition', 'sticker', 'filter', 'video_animation', 'texts']
            for ti in range(6):
                top1, top10 = top_accuracy(sim_matrix, gt_category_list,
                                            [int(t == ti) for t in type_list])
                print(type_name_list[ti], f"Top-1: {top1:.2%}", f"Top-10 accuracy: {top10:.2%}")
                all_top1.append(top1)
                all_top10.append(top10)

            print(f"Avg Top-1: {sum(all_top1)/len(all_top1):.2%}", f"Avg Top-10: {sum(all_top10)/len(all_top10):.2%}")