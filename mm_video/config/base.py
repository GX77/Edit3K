# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 16:02
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : meter.py

import argparse
import os

import torch
from fvcore.common.config import CfgNode
from .custom_config import add_custom_default_config, add_custom_check_config

_C = CfgNode()

# project info
_C.INFO = CfgNode()
_C.INFO.PROJECT_NAME = "untitled_project"
_C.INFO.EXPERIMENT_NAME = "untitled_experiment"

# system config
_C.SYS = CfgNode()
_C.SYS.MULTIPROCESS = False
_C.SYS.INIT_METHOD = "tcp://localhost:2222"
_C.SYS.NUM_GPU = torch.cuda.device_count()
_C.SYS.GPU_DEVICES = list(range(torch.cuda.device_count()))
_C.SYS.NUM_SHARDS = 1
_C.SYS.SHARD_ID = 0
_C.SYS.DETERMINISTIC = False
_C.SYS.SEED = 222

# log config
_C.LOG = CfgNode()
_C.LOG.DIR = None
_C.LOG.LOGGER_FILE = "logger.log"
_C.LOG.LOGGER_CONSOLE_LEVEL = "info"
_C.LOG.LOGGER_CONSOLE_COLORFUL = True

# build config for base dataloader
_C.DATA = CfgNode()
_C.DATA.DATASET = CfgNode()
_C.DATA.DATASET.NAME = None
_C.DATA.LOADER = CfgNode()
_C.DATA.LOADER.COLLATE_FN = None
_C.DATA.LOADER.BATCH_SIZE = 1
_C.DATA.LOADER.NUM_WORKERS = 0
_C.DATA.LOADER.SHUFFLE = False
_C.DATA.LOADER.PREFETCH_FACTOR = 2
_C.DATA.LOADER.MULTIPROCESSING_CONTEXT = "spawn"

# build config for base model
_C.MODEL = CfgNode()
_C.MODEL.NAME = None
_C.MODEL.DDP = CfgNode()
_C.MODEL.DDP.FIND_UNUSED_PARAMETERS = False

# optimizer
_C.OPTIMIZER = CfgNode()
_C.OPTIMIZER.NAME = "Adam"

# scheduler
_C.SCHEDULER = CfgNode()
_C.SCHEDULER.NAME = None

# build config for loss
_C.LOSS = CfgNode()
_C.LOSS.NAME = None
_C.LOSS.MultiObjectiveLoss = CfgNode()
_C.LOSS.MultiObjectiveLoss.LOSSES = []
_C.LOSS.MultiObjectiveLoss.WEIGHT = None

# build config for meter
_C.METER = CfgNode()
_C.METER.NAME = None

# trainer
_C.TRAINER = CfgNode()
_C.TRAINER.NAME = "TrainerBase"
# trainer base
_C.TRAINER.TRAINER_BASE = CfgNode()
_C.TRAINER.TRAINER_BASE.DEBUG = False
_C.TRAINER.TRAINER_BASE.TEST_ENABLE = True
_C.TRAINER.TRAINER_BASE.TRAIN_ENABLE = True
_C.TRAINER.TRAINER_BASE.EPOCH = 50
_C.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS = 1
_C.TRAINER.TRAINER_BASE.RESUME = None
_C.TRAINER.TRAINER_BASE.AUTO_RESUME = False
_C.TRAINER.TRAINER_BASE.CLIP_NORM = None
_C.TRAINER.TRAINER_BASE.SAVE_FREQ = 1
_C.TRAINER.TRAINER_BASE.LOG_FREQ = 1
_C.TRAINER.TRAINER_BASE.AMP = False
_C.TRAINER.TRAINER_BASE.DEBUG = False
_C.TRAINER.TRAINER_BASE.WRITE_HISTOGRAM = False
_C.TRAINER.TRAINER_BASE.WRITE_PROFILER = False

add_custom_default_config(_C)
add_custom_check_config(_C)
default_config = _C.clone()
default_config.freeze()


def check_config(cfg: CfgNode):
    # default check config
    if cfg.LOG.DIR is None:
        info = [i for i in [cfg.INFO.PROJECT_NAME, cfg.INFO.EXPERIMENT_NAME] if i]
        if info:  # not empty
            cfg.LOG.DIR = os.path.join("log", "_".join(info))
        else:
            cfg.LOG.DIR = os.path.join("log", "default")
    assert not cfg.SYS.MULTIPROCESS or cfg.SYS.NUM_GPU > 0, "At least 1 GPU is required to enable ddp."
    assert cfg.TRAINER.TRAINER_BASE.GRADIENT_ACCUMULATION_STEPS > 0, "gradient accumulation step should greater than 0."
    assert cfg.LOSS.MultiObjectiveLoss.WEIGHT is None or \
           len(cfg.LOSS.MultiObjectiveLoss.WEIGHT) == len(cfg.LOSS.MultiObjectiveLoss.LOSSES)


def get_config():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c",
                        help="path to the additional config file",
                        default="configs/edit_element.yaml",
                        type=str)
    parser.add_argument("--debug",
                        help="set trainer to debug mode",
                        action="store_true")
    parser.add_argument("opts",
                        help="see config/custom_config.py for all options",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # config
    cfg = default_config.clone()
    cfg.defrost()
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    if args.debug:
        cfg.TRAINER.TRAINER_BASE.DEBUG = True
    check_config(cfg)
    add_custom_check_config(cfg)
    cfg.freeze()
    return cfg


def dump(cfg: CfgNode, config_file: str):
    with open(config_file, "w") as f:
        f.write(cfg.dump())


if __name__ == '__main__':
    print(get_config())
