# -*- coding: utf-8 -*-
# @Time    : 2022/11/17 16:54
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : dataset_edit.py

import os
import torch
from torch.utils import data
import cv2
import random
from ..build import DATASET_REGISTRY
from PIL import Image
from .until_function import load_json, random_pick_two_elements, _transform
import numpy as np
from decord import VideoReader
from decord import cpu

@DATASET_REGISTRY.register()
class EditDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.mode = split
        self.video_root = cfg.DATA.DATASET.EDIT.VIDEO_ROOT
        self.all_edit_names = cfg.DATA.DATASET.EDIT.ALL_EDIT
        self.max_frames = cfg.DATA.DATASET.EDIT.MAX_FRAMES
        self.transform = _transform(cfg.DATA.DATASET.EDIT.SHAPE)
        
        all = load_json(self.all_edit_names)
        if split == "train":
            self.data = [i for i in range(len(all['train']))] * cfg.DATA.DATASET.EDIT.LENGTH
            self.train = all['train']
        else:
            self.data = [(k,v) for k,v in all['test'].items()]

    def __len__(self):
        return len(self.data)

    def load_frames(self, video_path, normalize=None):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_step = total_frames // self.max_frames
            frame_indices = np.arange(0, total_frames, frame_step)[:self.max_frames]
            frames = vr.get_batch(frame_indices).asnumpy()
            frame_list = [normalize(Image.fromarray(frame)) for frame in frames]
            frames = torch.stack(frame_list)
        except Exception as e:
            print(video_path, e)
            frames = torch.zeros(self.max_frames, 3, 224, 224)
        return frames

    def __getitem__(self, idx):
        if self.mode == "train":
            edit_dict = self.train[self.data[idx]]
            edit_id = random.sample(list(edit_dict.keys()), 1)[0]
            material_id_list = edit_dict[edit_id]
        else:
            edit_id, material_id_list = self.data[idx]

        video_name_1, video_name_2 = random_pick_two_elements(material_id_list)

        edit_video_1 = self.load_frames(self.video_root + video_name_1, self.transform)
        edit_video_2 = self.load_frames(self.video_root + video_name_2, self.transform)

        return {
            "edit_video_1": edit_video_1,
            "edit_video_2": edit_video_2,
            "v1_path": self.video_root + video_name_1,
            "v2_path": self.video_root + video_name_2,
            "edit_type": edit_id
        }