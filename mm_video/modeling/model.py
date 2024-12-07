# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 21:57
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : meter.py

import logging

import torch.nn as nn
import torch.distributed as dist
from yacs.config import CfgNode

from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

logger = logging.getLogger(__name__)


def build_model(cfg: CfgNode):
    logger.debug("Building model...")
    assert cfg.MODEL.NAME is not None, "specify a model to load in config: MODEL.NAME"
    model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)
    # data parallel and distributed data parallel
    if cfg.SYS.MULTIPROCESS:
        # move to GPU
        device_id = cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU]
        logger.debug("Moving model to device: %s...", device_id)
        model.to(device_id)
        logger.debug("Model is moved to device: %s", device_id)
        logger.debug("Build DistributedDataParallel, check whether the program is hanging...")
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU]],
            output_device=cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU],
            find_unused_parameters=cfg.MODEL.DDP.FIND_UNUSED_PARAMETERS
        )
    elif cfg.SYS.NUM_GPU > 0:
        model = model.cuda()
        model = nn.parallel.DataParallel(model)
    logger.debug("Model build finished.")
    return model
