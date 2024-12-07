"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from typing import *
import torch
from torch import nn
from mm_video.modeling.model import MODEL_REGISTRY
from mm_video.modeling.bert.modules import BertEncoder, BertDecoder
from mm_video.modeling.clip.model import build_model
from mm_video.data.datasets.until_function import load_json
from sklearn.cluster import KMeans

from easydict import EasyDict as EDict

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register()
class Edit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        clip_pretrained_model = torch.jit.load("ViT-B-16.pt", map_location="cpu").state_dict()  # hard coded model path
        self.clip_backbone = build_model(clip_pretrained_model).float()
        self.hidden_size = 512

        config = EDict(
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            hidden_dropout_prob=0.1,
            intermediate_size=self.hidden_size,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            max_frame=cfg.DATA.DATASET.EDIT.MAX_FRAMES
        )
        self.encoder = BertEncoder(config, cfg.MODEL.Edit.LAYERS)
        self.decoder = BertDecoder(config, cfg.MODEL.Edit.LAYERS)

        self.temp_embedding = {}
        self.dyn_embedding = None
        self.guide_query = None
        self.linear = nn.Linear(self.hidden_size, 768)

    def extract_frame_features(self, data, guide_query):
        B = len(data)
        data = torch.flatten(data, start_dim=0, end_dim=1)
        clip_features = self.clip_backbone.encode_image(data, guide_query, output_all_features=True)[0]
        clip_features = clip_features.reshape(B, -1, clip_features.size(-1))
        return clip_features

    def generate_embedding(self, edit_video, guide_query):
        feature_1 = self.extract_frame_features(edit_video, self.linear(guide_query))
        fuse_1 = self.encoder(feature_1)
        query = torch.zeros(fuse_1.size(0), 1, fuse_1.size(-1)).to(fuse_1.device)
        video_embedding = self.decoder(query, fuse_1, guide_query).squeeze(1)
        return video_embedding

    def keep_dynamic_embedding(self, edit_id_list, element_1, element_2):
        for i, edit_id in enumerate(edit_id_list):
            if edit_id in self.temp_embedding.keys():
                self.temp_embedding[edit_id].append((element_1[i]+element_2[i])/2)
                self.temp_embedding[edit_id] = self.temp_embedding[edit_id][-5:]
            else:
                self.temp_embedding[edit_id] = [(element_1[i]+element_2[i])/2]

        dyn_embedding = []
        for k, v in self.temp_embedding.items():
            dyn_embedding.append(torch.stack(v).mean(0))
        self.dyn_embedding = torch.stack(dyn_embedding)

    def generate_guide_query(self, name_list, embedding):
        embedding_np = embedding.cpu().numpy()
        n_clusters = 6  # Number of categories. It can be modified according to the actual situation.
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embedding_np)
        labels = kmeans.labels_
        temp_guided_query = []
        for i in range(n_clusters):
            cluster_embeddings = embedding[labels == i]
            cluster_mean = cluster_embeddings.mean(dim=0)
            temp_guided_query.append(cluster_mean)
        self.guide_query = torch.stack(temp_guided_query)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        feature_1 = self.extract_frame_features(inputs["edit_video_1"], self.linear(self.guide_query) if self.guide_query != None else None)
        feature_2 = self.extract_frame_features(inputs["edit_video_2"], self.linear(self.guide_query) if self.guide_query != None else None)
        fuse_1 = self.encoder(feature_1)
        fuse_2 = self.encoder(feature_2)

        query = torch.zeros(fuse_1.size(0), 1, fuse_1.size(-1)).to(fuse_1.device)
        element_1 = self.decoder(query, fuse_1, self.guide_query).squeeze(1)
        element_2 = self.decoder(query, fuse_2, self.guide_query).squeeze(1)

        if self.training:
            with torch.no_grad():
                self.keep_dynamic_embedding(inputs['edit_type'], element_1, element_2)
            dyn_list = [k for k in self.temp_embedding.keys()]
            if self.dyn_embedding != None and self.dyn_embedding.size(0)>=6:
                self.generate_guide_query(dyn_list, self.dyn_embedding)
            labels = torch.LongTensor([index for value in inputs['edit_type'] for index, element in enumerate(dyn_list) if element == value]).to(element_1.device)

            return {"element_1": element_1, "element_2": element_2, "dyn_embedding": self.dyn_embedding, "labels": labels}
        else:
            return {"element_1": element_1, "element_2": element_2, "dyn_embedding": None, "labels": None}