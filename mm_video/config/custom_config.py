# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 16:03
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : custom_config.py

import os
import torch
from fvcore.common.config import CfgNode

from ipaddress import ip_address, IPv4Address


def is_ipv6(ip: str) -> bool:
    try:
        return False if type(ip_address(ip)) is IPv4Address else True
    except ValueError:
        return False


def add_custom_default_config(cfg: CfgNode):
    # >>> add custom default configs here <<<
    pass

    cfg.INFO.PROJECT_NAME = "mm_video"

    # set arnold ddp variables
    cfg.SYS.INIT_METHOD = None
    cfg.SYS.NUM_SHARDS = int(os.getenv("ARNOLD_WORKER_NUM")) if os.getenv("ARNOLD_WORKER_NUM") is not None else 1
    cfg.SYS.SHARD_ID = int(os.getenv("ARNOLD_ID")) if os.getenv("ARNOLD_ID") is not None else 0
    master_address = os.getenv("METIS_WORKER_0_HOST") if \
        os.getenv("METIS_WORKER_0_HOST") is not None and os.getenv("WORKSPACE_ENVS_SET") is None else "localhost"
    master_port = int(os.getenv("METIS_WORKER_0_PORT").split(",")[0]) \
        if os.getenv("METIS_WORKER_0_PORT") is not None else 2222
    cfg.SYS.INIT_METHOD = f"tcp://{master_address}:{master_port}" if not is_ipv6(
        master_address) else f"tcp://[{master_address}]:{master_port}"

    # EDIT
    cfg.DATA.DATASET.EDIT = CfgNode()
    cfg.DATA.DATASET.EDIT.VIDEO_ROOT = ""
    cfg.DATA.DATASET.EDIT.ALL_EDIT = ""
    cfg.DATA.DATASET.EDIT.LENGTH = 1000
    cfg.DATA.DATASET.EDIT.TEST_MATERIAL = []
    cfg.DATA.DATASET.EDIT.MAX_FRAMES = 8
    cfg.DATA.DATASET.EDIT.TEST = None
    cfg.DATA.DATASET.EDIT.CLASSES = 4510 + 1
    cfg.DATA.DATASET.EDIT.SHAPE = (224, 224)
    cfg.DATA.DATASET.EDIT.RATIO = 1.0
    cfg.MODEL.Edit = CfgNode()
    # Edit architecture
    # cfg.MODEL.Edit.VID_BACKBONE = "clip"  # supported backbone: clip, clip4clip, vidswin
    cfg.MODEL.Edit.TASK_TYPE = ""

    # pretext opt
    cfg.MODEL.Edit.VISUAL_MASKED_MODELING = CfgNode()
    cfg.MODEL.Edit.VISUAL_MASKED_MODELING.MASK_PROB = 0.0
    cfg.MODEL.Edit.VISUAL_MASKED_MODELING.NUM_RECONSTRUCT_LAYERS = 4
    # task specific head
    cfg.MODEL.Edit.NUM_CLASSES = None  # video action recognition (classification)
    cfg.MODEL.Edit.LAYERS = 2
    cfg.MODEL.Edit.RAW = True

    cfg.OPTIMIZER.BertAdam = CfgNode()
    cfg.OPTIMIZER.BertAdam.lr = 1e-4
    cfg.OPTIMIZER.BertAdam.warmup = 0.1
    cfg.OPTIMIZER.BertAdam.schedule = "warmup_constant"
    cfg.OPTIMIZER.BertAdam.t_total = None
    cfg.OPTIMIZER.BertAdam.weight_decay = 0.01
    cfg.OPTIMIZER.BertAdam.max_grad_norm = 1.0

    cfg.LOSS.MILNCELoss = CfgNode()
    cfg.LOSS.MILNCELoss.BATCH_SIZE = None  # generate automatically, do not specify
    cfg.LOSS.MILNCELoss.N_PAIR = 1

    cfg.LOSS.MaxMarginRankingLoss = CfgNode()
    cfg.LOSS.MaxMarginRankingLoss.MARGIN = 0.1
    cfg.LOSS.MaxMarginRankingLoss.NEGATIVE_WEIGHT = 1
    cfg.LOSS.MaxMarginRankingLoss.BATCH_SIZE = None  # generate automatically, do not specify
    cfg.LOSS.MaxMarginRankingLoss.N_PAIR = 1
    cfg.LOSS.MaxMarginRankingLoss.HARD_NEGATIVE_RATE = 0.5

    cfg.TRAINER.EditTrainer = CfgNode()
    cfg.TRAINER.EditTrainer.TASK_TYPE = "pretrain"
    cfg.TRAINER.EditTrainer.VID_LR = 5e-6
    cfg.TRAINER.EditTrainer.LR_DECAY_GAMMA = 0.95
    cfg.TRAINER.EditTrainer.VID_BACKBONE = "clip"


def add_custom_check_config(cfg: CfgNode):
    # >>> add custom config check here <<<
    pass

    cfg.MODEL.Edit.BATCH_SIZE = cfg.DATA.LOADER.BATCH_SIZE
    cfg.LOSS.MILNCELoss.BATCH_SIZE = cfg.DATA.LOADER.BATCH_SIZE
    cfg.LOSS.MaxMarginRankingLoss.BATCH_SIZE = cfg.DATA.LOADER.BATCH_SIZE