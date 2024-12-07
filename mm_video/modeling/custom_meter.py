# -*- coding: utf-8 -*-
# @Time    : 2022/11/21 18:44
# @Author  : Xin Gu
# @Project : EDIT_ELEMNET
# @File    : custom_meter.py
import json
import os
import torch
import torch.distributed as dist
import numpy as np
import einops
import logging
import datetime
from typing import *
from torch.utils.tensorboard import SummaryWriter
from fvcore.common.config import CfgNode
import flow_vis

from mm_video.utils.train_utils import gather_object_multiple_gpu

from .meter import METER_REGISTRY, MeterBase

logger = logging.getLogger(__name__)


@METER_REGISTRY.register()
class AccuracyMeter(MeterBase):
    """
    accuracy meter for video action recognition
    """

    def __init__(self, cfg: CfgNode, writer: SummaryWriter, mode: str):
        super(AccuracyMeter, self).__init__(cfg=cfg, writer=writer, mode=mode)

        self._top1 = []
        self._top5 = []
        self._rank = []
        self._id = []

        self._label = []
        self._predict = []

    @staticmethod
    def topk_accuracy(logits: torch.Tensor, target: torch.LongTensor, topk=(1, 5), average=False, verbose=False):
        assert len(logits.shape) == 2
        assert len(target.shape) == 1
        assert logits.shape[0] == target.shape[0]

        num_class = logits.size(1)

        _, pred = logits.topk(num_class, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = {}
        for k in topk:
            if average:
                cur_acc = correct[:k].float().sum(0).mean().cpu().numpy().item() * 100.0
            else:
                cur_acc = correct[:k].float().sum(0).cpu().numpy().tolist()
            ret[f"R@{k}"] = cur_acc

        _, rank = correct.float().topk(1, dim=0)
        rank = rank[0]
        if average:
            # res.append(torch.mean(rank.float()))
            ret["rank"] = torch.mean(rank.float()).cpu().numpy().item()
        else:
            ret["rank"] = rank.cpu().numpy().tolist()

        if verbose:  # debug log
            logger.debug("[pred, gt]: %s", list(zip(torch.argmax(logits, dim=-1).cpu().numpy(), target.cpu().numpy())))
            logger.debug("Accuracy: %s", ret)
        return ret

    @torch.no_grad()
    def _update(self, labels, outputs, global_step=None, idx=None):
        """
        update for each step
        :param labels: 1D LongTensor for classification
        :param outputs: 2D logits with shape (BATCH_SIZE, NUM_CLASSES)
        :param global_step: specify global step manually
        :param idx: unique id for each sample
        """
        assert len(labels.shape) == 1, "Label should be 1 dim tensor"
        assert len(outputs.shape) == 2, "Output should be 2 dim tensor"
        assert labels.size(0) == outputs.size(0), "Label and output should have same batch size"
        if idx is not None and isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        elif idx is None:
            idx = [None] * labels.size(0)
        accuracy_result = self.topk_accuracy(logits=outputs, target=labels, topk=(1, 5), average=False, verbose=True)
        self._top1.extend(accuracy_result["R@1"])
        self._top5.extend(accuracy_result["R@5"])
        self._rank.extend(accuracy_result["rank"])
        self._id.extend(idx)

        for cur_label in labels:
            self._label.append(cur_label.item())
        for cur_logits in outputs:
            assert len(cur_logits.shape) == 1, f"logits should be 1 dim tensor, received {cur_logits.shape}"
            pred = torch.argmax(cur_logits).item()
            self._predict.append(pred)
        assert len(self._label) == len(self._predict)

        # write tensorboard
        if self.mode in ["train"] and self.writer is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            self.writer.add_scalar(f"{self.mode}/R@1", np.mean(accuracy_result["R@1"]), global_step=global_step)
            self.writer.add_scalar(f"{self.mode}/R@5", np.mean(accuracy_result["R@5"]), global_step=global_step)
            self.writer.add_scalar(f"{self.mode}/rank", np.mean(accuracy_result["rank"]), global_step=global_step)

    @torch.no_grad()
    def update(self, inputs, outputs, global_step=None):
        self._update(labels=inputs["label"], outputs=outputs["cls_logits"], global_step=global_step,
                     idx=inputs["id"] if "id" in inputs else None)

    @torch.no_grad()
    def summary(self, epoch):
        acc_top1 = gather_object_multiple_gpu(self._top1) if dist.is_initialized() else self._top1
        acc_top5 = gather_object_multiple_gpu(self._top5) if dist.is_initialized() else self._top5
        rank = gather_object_multiple_gpu(self._rank) if dist.is_initialized() else self._rank
        idx = gather_object_multiple_gpu(self._id) if dist.is_initialized() else self._id
        assert len(acc_top1) == len(acc_top5) == len(rank) == len(idx)

        all_metric = {}
        reassign_idx = 0
        for r1, r5, r, i in zip(acc_top1, acc_top5, rank, idx):
            if i is None:
                while reassign_idx in all_metric:
                    reassign_idx += 1
                i = reassign_idx
            all_metric[i] = {"R@1": r1, "R@5": r5, "Rank": r}
        with open(os.path.join(self.cfg.LOG.DIR, f"accuracy_epoch_{epoch}_{self.mode}.json"), "w") as f:
            json.dump(all_metric, f)
        acc_top1 = [x["R@1"] for i, x in all_metric.items()]
        acc_top5 = [x["R@5"] for i, x in all_metric.items()]
        rank = [x["Rank"] for i, x in all_metric.items()]
        avg_acc_top1 = np.mean(acc_top1)
        avg_acc_top5 = np.mean(acc_top5)
        avg_rank = np.mean(rank)
        mid_rank = np.median(rank)

        if dist.is_initialized():
            labels = np.array(gather_object_multiple_gpu(self._label))
            predicts = np.array(gather_object_multiple_gpu(self._predict))
        else:
            labels = np.array(self._label)
            predicts = np.array(self._predict)
        correct = (labels == predicts).astype(int)
        individual_labels = list(set(labels))
        accuracy_per_class = {}
        for cur_label in individual_labels:
            correct_selected = correct[labels == cur_label]
            accuracy_per_class[cur_label] = correct_selected.sum() / len(correct_selected)
        logger.debug("accuracy for each class: %s", accuracy_per_class)
        if not dist.is_initialized() or dist.get_rank() == 0:
            img = self.visualize_per_class_accuracy(accuracy_per_class, avg_acc_top1)
            self.writer.add_image(f"{self.mode}/accuracy_epoch", img, global_step=epoch, dataformats="HWC")

        logger.debug("Accuracy meter summary got {} samples".format(len(acc_top1)))
        if self.writer is not None and (not dist.is_initialized() or dist.get_rank() == 0) and epoch is not None:
            self.writer.add_scalar(f"{self.mode}/R@1_epoch", avg_acc_top1, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/R@5_epoch", avg_acc_top5, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/MeanR_epoch", avg_rank, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/MedianR_epoch", mid_rank, global_step=epoch)
        if self.mode in ["val", "test"] and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(f">>> Epoch {epoch} ({self.mode}): R@1: {avg_acc_top1} - R@5: {avg_acc_top5}"
                        f" - MeanR: {avg_rank} - MedianR: {mid_rank}")

        return {"R@1": avg_acc_top1,
                "R@5": avg_acc_top5,
                "MeanR": avg_rank,
                "MedianR": mid_rank}

    def reset(self):
        self._top1.clear()
        self._top5.clear()
        self._label.clear()
        self._predict.clear()
        self._rank.clear()
        self._id.clear()

    def visualize_per_class_accuracy(self, accuracy_dict: Dict[Any, str], average_acc: float):
        import matplotlib.pyplot as plt

        labels = [k for k, v in accuracy_dict.items()]
        acc = [v * 100 for k, v in accuracy_dict.items()]

        save_path = os.path.join(self.cfg.LOG.DIR,
                                 f"Accuracy-{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.png")

        ax: plt.Axes = plt.gca()
        ax.bar(labels, acc)
        ax.set_xlabel("Label")
        ax.set_ylabel("Accuracy(%)")
        ax.set_ylim(0.0, 100.0)
        ax.axhline(average_acc, color="r", label="average", linestyle=":")
        plt.savefig(save_path)
        logger.debug("Accuracy for each class is saved to %s", save_path)
        plt.close()
        img = plt.imread(save_path)
        return img


@METER_REGISTRY.register()
class ReconstructVisualizeMeter(MeterBase):
    show_freq = 100
    curr_frames = 8
    img_size = (224, 224)
    patch_size = (16, 16)

    def reset(self):
        pass

    @staticmethod
    def motion_vector_to_rgb(motion_vector):
        # motion_vector: b n c h w
        B, N, C, H, W = motion_vector.shape
        flow_vis.flow_to_color(motion_vector[0][0].permute(1, 2, 0).cpu().numpy())

        flow = np.zeros((B, N, 3, H, W), dtype=np.uint8)
        for b in range(B):
            for n in range(N):
                flow[b][n] = np.transpose(flow_vis.flow_to_color(motion_vector[b][n].permute(1, 2, 0).cpu().numpy()),
                                          (2, 0, 1))
        return flow

    @torch.no_grad()
    def update(self, inputs, outputs, n=None, global_step=None):
        if global_step % self.show_freq == 0:
            iframe_target = outputs["iframe_patched"]
            motion_vector_target = outputs["motion_vector_patched"]
            iframe_pred = outputs["iframe_pred"]
            motion_vector_pred = outputs["motion_vector_pred"]
            iframe_mask = outputs["iframe_mask"]
            motion_vector_mask = outputs["motion_vector_mask"]

            iframe_pred = einops.rearrange(
                iframe_pred,
                "b (n h_p w_p) (p_h p_w c)->b n c (h_p p_h) (w_p p_w)",
                n=self.curr_frames, c=3,
                h_p=self.img_size[0] // self.patch_size[0],
                w_p=self.img_size[1] // self.patch_size[1],
                p_h=self.patch_size[0], p_w=self.patch_size[1]
            )
            iframe_masked = einops.rearrange(
                iframe_target * iframe_mask.unsqueeze(-1),
                "b (n h_p w_p) (p_h p_w c)->b n c (h_p p_h) (w_p p_w)",
                n=self.curr_frames, c=3,
                h_p=self.img_size[0] // self.patch_size[0],
                w_p=self.img_size[1] // self.patch_size[1],
                p_h=self.patch_size[0], p_w=self.patch_size[1]
            )
            iframe_target = einops.rearrange(
                iframe_target,
                "b (n h_p w_p) (p_h p_w c)->b n c (h_p p_h) (w_p p_w)",
                n=self.curr_frames, c=3,
                h_p=self.img_size[0] // self.patch_size[0],
                w_p=self.img_size[1] // self.patch_size[1],
                p_h=self.patch_size[0], p_w=self.patch_size[1]
            )
            motion_vector_pred = einops.rearrange(
                motion_vector_pred,
                "b (n h_p w_p) (p_h p_w c)->b n c (h_p p_h) (w_p p_w)",
                n=self.curr_frames, c=2,
                h_p=self.img_size[0] // self.patch_size[0],
                w_p=self.img_size[1] // self.patch_size[1],
                p_h=self.patch_size[0], p_w=self.patch_size[1]
            )
            motion_vector_pred = self.motion_vector_to_rgb(motion_vector_pred)
            motion_vector_masked = einops.rearrange(
                motion_vector_target * motion_vector_mask.unsqueeze(-1),
                "b (n h_p w_p) (p_h p_w c)->b n c (h_p p_h) (w_p p_w)",
                n=self.curr_frames, c=2,
                h_p=self.img_size[0] // self.patch_size[0],
                w_p=self.img_size[1] // self.patch_size[1],
                p_h=self.patch_size[0], p_w=self.patch_size[1]
            )
            motion_vector_masked = self.motion_vector_to_rgb(motion_vector_masked)
            motion_vector_target = einops.rearrange(
                motion_vector_target,
                "b (n h_p w_p) (p_h p_w c)->b n c (h_p p_h) (w_p p_w)",
                n=self.curr_frames, c=2,
                h_p=self.img_size[0] // self.patch_size[0],
                w_p=self.img_size[1] // self.patch_size[1],
                p_h=self.patch_size[0], p_w=self.patch_size[1]
            )
            motion_vector_target = self.motion_vector_to_rgb(motion_vector_target)

            self.writer.add_images("iframe/target",
                                   np.clip(iframe_target[0][:2], a_min=0., a_max=1.), global_step=global_step)
            self.writer.add_images("iframe/pred",
                                   np.clip(iframe_pred[0][:2], a_min=0., a_max=1.), global_step=global_step)
            self.writer.add_images("iframe/masked",
                                   np.clip(iframe_masked[0][:2], a_min=0., a_max=1.), global_step=global_step)
            self.writer.add_images("motion_vector/target", motion_vector_target[0][:2], global_step=global_step)
            self.writer.add_images("motion_vector/pred", motion_vector_pred[0][:2], global_step=global_step)
            self.writer.add_images("motion_vector/masked", motion_vector_masked[0][:2], global_step=global_step)

    def summary(self, epoch):
        pass
