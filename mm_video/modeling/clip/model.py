from collections import OrderedDict
from typing import Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

import logging

logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        out, attn = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, key_padding_mask=padding_mask)
        return out

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 in_channels: int = 3, n_cls: int = 1):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.n_cls = n_cls
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size,
                               bias=False)

        scale = width ** -0.5
        if n_cls == 1:
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
        else:
            self.class_embedding = nn.Parameter(scale * torch.randn(n_cls, width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + self.n_cls, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width * n_cls, output_dim))

    def forward(self, x: torch.Tensor, guide_query, output_all_features: bool = False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], self.n_cls, x.shape[-1],
                                                            dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if guide_query != None:
            guide_query = guide_query.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat((x, guide_query), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.n_cls == 1:
            cls_feature = self.ln_post(x[:, 0, :]) @ self.proj
        else:
            cls_feature = einops.rearrange(self.ln_post(x[:, :self.n_cls, :]), "b n_cls c->b (n_cls c)") @ self.proj
        if output_all_features:
            return cls_feature, x[:, 1:, :]
        else:
            return cls_feature


class CrossResidualAttentionBlock(ResidualAttentionBlock):
    """modified version of ResidualAttentionBlock to support the encoder-decoder attention between I-frame tokens and
    motion vector/residual"""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 enc_dec_attn_mask: torch.Tensor = None):
        super().__init__(d_model=d_model, n_head=n_head, attn_mask=attn_mask)
        self.attn2 = nn.MultiheadAttention(d_model, n_head)
        self.ln_3 = LayerNorm(d_model)
        self.ln_4 = LayerNorm(d_model)
        self.enc_dec_attn_mask = enc_dec_attn_mask

    def enc_dec_attention(self, highway: torch.Tensor, iframe: torch.Tensor):
        self.enc_dec_attn_mask = self.enc_dec_attn_mask.to(dtype=highway.dtype,
                                                           device=highway.device) if self.enc_dec_attn_mask is not None else None
        return self.attn2(highway, iframe, iframe, need_weights=False, attn_mask=self.enc_dec_attn_mask)[0]

    def forward(self, x: [torch.Tensor, torch.Tensor, torch.LongTensor]):
        highway, iframe, self_mask = x
        # self-attention
        x = highway + self.attention(self.ln_1(highway), padding_mask=self_mask)
        # enc-dec attention
        if iframe is not None:
            x = x + self.enc_dec_attention(self.ln_3(x), self.ln_4(iframe))
        # mlp
        x = x + self.mlp(self.ln_2(x))
        return [x, iframe, self_mask]


class TransformerDec(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[CrossResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        # add self attention

    def forward(self, highway: torch.Tensor, iframe: torch.Tensor = None, self_mask: torch.Tensor = None):
        return self.resblocks([highway, iframe, self_mask])


class MotionTransformer(nn.Module):
    def __init__(self, input_resolution: int, in_channels: int, num_mv: int, num_res: int, patch_size: int,
                 enc_width: int,
                 dec_width: int, iframe_width: int,
                 enc_layers: int, dec_layers: int, heads: int, output_dim: int, with_type_embedding: bool):
        """
        Encode motion vector and use decoder structure to query context info from I-frame feature
        :param input_resolution: resolution of motion vector
        :param in_channels: channel of motion vector
        :param num_mv: number of motion vector in each GOP
        :param num_res: number of residual in each GOP
        :param patch_size: patch size for motion extractor (VisionTransformer)
        :param enc_width: with for motion extractor
        :param dec_width: width for decoder
        :param iframe_width: I-frame width
        :param enc_layers: layer of motion extractor
        :param dec_layers: layer of decoder
        :param heads:
        :param output_dim:
        :param with_type_embedding:
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.with_type_embedding = with_type_embedding

        self.iframe_proj = nn.Parameter(dec_width ** -0.5 * torch.randn(iframe_width, dec_width))
        # TODO: replace the linear projection of CLS token with concat of multiple CLS token
        self.motion_extractor = VisionTransformer(input_resolution=input_resolution, patch_size=patch_size,
                                                  width=enc_width, layers=enc_layers, heads=heads,
                                                  output_dim=dec_width, in_channels=in_channels, n_cls=1)

        scale = dec_width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_mv, dec_width))
        if self.with_type_embedding:
            self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=dec_width)
            self.type_embedding.weight = nn.Parameter(dec_width ** -0.5 * torch.randn(3, dec_width))  # init like PE
        self.iframe_residual_embedding = nn.Parameter(scale * torch.randn(1 + num_res, dec_width))
        self.ln_pre = LayerNorm(dec_width)

        # no self-attention, iframe only use CLS
        self.transformer = TransformerDec(dec_width, dec_layers, heads)

        self.ln_post = LayerNorm(dec_width)
        self.proj = nn.Parameter(scale * torch.randn(dec_width, output_dim))

    def forward(self, motion_vector: torch.Tensor, iframe: torch.Tensor = None, residual: torch.Tensor = None,
                type_ids_mv: torch.LongTensor = None, input_mask_mv: torch.LongTensor = None):
        """
        :param motion_vector: motion vector with shape (b n_gop) n_mv c h w
        :param iframe: I-frame features with shape (b n_gop) n_patch c
        :param residual: residual features with shape (b n_gop) n_res c
        :param type_ids_mv: frame type of each mv
        :param input_mask_mv: shape is (b n_gop) n_mv
        :return: motion feature with shape: b n_bp c
        """
        # avoid NAN caused by empty GOP
        input_mask_mv[:, 0] = 0

        b, n_mv = motion_vector.shape[:2]
        assert iframe is not None
        assert residual is not None
        ref_feature = iframe @ self.iframe_proj  # project to same width

        motion_vector = einops.rearrange(motion_vector, "b n c h w->(b n) c h w")
        motion_vector = einops.rearrange(self.motion_extractor(motion_vector, output_all_features=False),
                                         "(b n) c->b n c", b=b, n=n_mv)  # (b n_gop), n_mv, c
        assert motion_vector.size(1) == residual.size(1), "n_res != n_mv"
        fused_feature = motion_vector + residual + self.positional_embedding.to(motion_vector.dtype)
        if self.with_type_embedding:  # add type embedding when frame type is provided
            fused_feature = fused_feature + self.type_embedding(type_ids_mv).to(motion_vector.dtype)
        fused_feature = self.ln_pre(fused_feature)

        fused_feature = fused_feature.permute(1, 0, 2)  # NLD -> LND
        fused_feature = self.transformer(fused_feature, ref_feature.permute(1, 0, 2), input_mask_mv)[0]
        fused_feature = fused_feature.permute(1, 0, 2)  # LND -> NLD
        return self.ln_post(fused_feature) @ self.proj


class FuseTransformer(nn.Module):
    def __init__(self, n_gop: int, n_frame: int, width: int, layers: int, heads: int, fuse_motion: bool = True):
        super().__init__()
        self.n_gop = n_gop
        self.n_frame = n_frame
        self.fuse_motion = fuse_motion

        scale = torch.tensor(width ** -0.5)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.type_iframe_embedding = nn.Parameter(scale * torch.randn(width))
        self.type_motion_embedding = nn.Parameter(scale * torch.randn(width))
        self.position_embedding = nn.Parameter(scale * torch.randn(n_gop, width))
        self.gop_embedding = nn.Parameter(scale * torch.randn(n_gop, 1, width))
        self.ln_pre = LayerNorm(width)

        # hardcode config for motion attention mask
        with_attention_mask = False
        mask_ratio = 20

        self.fuse_attention_mask = nn.Parameter(self.build_attention_mask(mask_ratio=mask_ratio)) \
            if self.fuse_motion and with_attention_mask else None
        self.transformer = Transformer(width, layers, heads, attn_mask=self.fuse_attention_mask)

    def build_attention_mask(self, mask_ratio):
        mask = (torch.eye(1 + self.n_gop * (1 + self.n_frame)) - 1) * mask_ratio
        mask[0, [1 + i * (1 + self.n_frame) for i in range(self.n_gop)]] = 0
        return mask

    def forward(self, iframe_features: torch.Tensor, motion_features: torch.Tensor):
        """
        fuse
        :param iframe_features: batch_size, n_gop, c
        :param motion_features: batch_size, n_gop, c
        """
        assert iframe_features is not None or self.fuse_motion, "Motion must be enabled when iframe feature is None"
        # embedding
        if iframe_features is not None:
            iframe_features = iframe_features
            iframe_features = iframe_features + self.type_iframe_embedding
        if self.fuse_motion:
            motion_features = motion_features + self.type_motion_embedding
            if iframe_features is not None:
                x = torch.concat((iframe_features, motion_features), dim=1)
                x = x + torch.concat([self.position_embedding, self.position_embedding], dim=0)
            else:
                raise ValueError
                # x = motion_features
                # x = x + self.position_embedding[1:, :] + self.gop_embedding
        else:
            raise ValueError
            # x = iframe_features
            # x = x + self.position_embedding[:1, :] + self.gop_embedding
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.fuse_attention_mask is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            torch.set_printoptions(edgeitems=5, linewidth=150, threshold=10_000)
            logger.debug("Fuse attention mask: %s", self.fuse_attention_mask)
            torch.set_printoptions(profile="full", linewidth=99999)  # reset
            logger.debug("Fuse attention mask ([CLS]): %s", self.fuse_attention_mask[0, :])
            torch.set_printoptions(profile="default")  # reset

        return x[:, 0, :]


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, guide_query, output_all_features=False):
        return self.visual(image.type(self.dtype), guide_query, output_all_features)

    def encode_text(self, text, output_all_features=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if output_all_features:
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
