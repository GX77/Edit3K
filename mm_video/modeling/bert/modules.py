


from collections import defaultdict
import copy
import logging
import math
from typing import List, Optional, Tuple

from easydict import EasyDict
import torch
from torch import LongTensor, Tensor
import torch.nn as nn
import torch.nn.functional as F

#from .albert import BertEmbedding as AlbertEmbedding, Block as AlbertBlock
logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing: float, tgt_vocab_size: int, ignore_index: int = -100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


def gelu(x: Tensor) -> Tensor:
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters: int = 128, max_len: int = 1024):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x: Tensor) -> Tensor:
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class BertSelfAttention(nn.Module):
    def __init__(self, config: EasyDict):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states: Tensor, key_states: Tensor,
                value_states: Tensor) -> Tensor:
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config: EasyDict):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: EasyDict):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        self_output = self.self(input_tensor, input_tensor, input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config: EasyDict):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: EasyDict):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def make_shifted_mask(input_mask: Tensor, max_len_context: int,
                      max_len_causal: int, memory_len: int = 0) -> Tensor:
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_len_context: int, the first `max_len_context` is for video and its padding, the length
            of the rest of the bits is `max_len_causal`. We have L = `max_len_context` + `max_len_causal`.
            Note max_len_context may also include the memory len (M), thus max_len_context += M
        max_len_causal: int
        memory_len: int, M
    Returns:

    >>> max_len_context = 2; max_len_causal=3; input_mask = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask, max_len_context, max_len_causal)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_len_context + max_len_causal + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(bsz, max_len_context + max_len_causal, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len+max_len_context] = 1
    shifted_mask[:, max_len_context:, memory_len+max_len_context:] = torch.tril(
        input_mask.new_ones(max_len_causal, max_len_causal), diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask: Tensor, max_len_context: int,
                          max_len_causal: int, memory_len: int = 0) -> Tensor:
    """input_mask: (N, L), """
    shifted_mask = make_shifted_mask(input_mask, max_len_context, max_len_causal, memory_len=memory_len)
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    pad_shifted_mask = (1 - pad_shifted_mask.unsqueeze(1)) * -10000.
    return pad_shifted_mask


def make_video_only_mask(input_mask: Tensor, max_v_len: int) -> Tensor:
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
    return target_config


class BertLayer(nn.Module):
    def __init__(self, config: EasyDict):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention_output = self.attention(hidden_states)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config: EasyDict, num_layers):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(num_layers)])
        # self.layer = nn.ModuleList([AlbertLayer(config) for _ in range(config.num_hidden_layers)]) # no gain so far
        width = config.hidden_size
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(config.max_frame, width))

    def forward(self, x: Tensor) -> List:
        x = x + self.positional_embedding.to(x.dtype)
        for layer_idx, layer_module in enumerate(self.layer):
            x = layer_module(x)
        return x

    # def load_pretrained_encoder(self, pretrained_model: str):
    #     model_path_dict = {
    #         'albert-pretrained': 'pretrained_weights/albert-pretrained-encoder.pth'
    #     }
    #     albert_state_dict = torch.load(model_path_dict[pretrained_model], map_location='cpu')
    #     self.layer.load_state_dict(albert_state_dict, strict=False)


class BertEmbeddingsWithVideo(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """
    def __init__(self, config: EasyDict, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        self.modality_count = defaultdict(int, config.modality_count)

        # video embeddding #
        if self.modality_count['video'] > 0:
            self.video_embeddings = nn.Sequential(  # 2048->768
                nn.LayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.video_feature_size, config.hidden_size),
                nn.ReLU(True),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )

        # word embeddding #
        self.albert_word_embeddings = AlbertEmbedding(albert_config)

        # timestamp embeddding #
        if self.modality_count['timestamp'] > 0:
            self.timestamp_embeddings = nn.Sequential(  # 2->768
                nn.Linear(config.timestamp_feature_size, config.hidden_size, bias=False),
                nn.ReLU(True),
            )

        # positional embeddding #
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=2*(config.max_len_context+config.max_len_causal))

        # token type embeddding #
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.video_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def load_pretrained_embedding(self, pretrained_model: str):
        model_path_dict = {
            'albert-pretrained': '/mnt/bn/gxnas/guang.chen/pretrained_weights/albert-pretrained-embedding.pth'
        }
        albert_state_dict = torch.load(model_path_dict[pretrained_model], map_location='cpu')
        self.albert_word_embeddings.load_state_dict(albert_state_dict)

    def forward(self, input_ids: LongTensor, video_features: Optional[Tensor], token_type_ids: LongTensor) -> Tensor:

        # word embedding #
        assert(input_ids is not None)
        words_embeddings = self.albert_word_embeddings(input_ids)

        # video embedding #
        video_embeddings = 0
        if self.modality_count['video'] > 0:
            video_embeddings = self.video_embeddings(video_features)

        # token type embedding #
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # video type embedding
        video_type_ids = torch.cat((torch.zeros(token_type_ids.size(0), int(token_type_ids.size(1)/2)),torch.ones(token_type_ids.size(0), int(token_type_ids.size(1)/2))),dim = -1).int().cuda()
        video_type_embeddings = self.video_type_embeddings(video_type_ids)
        embeddings = words_embeddings + video_embeddings + token_type_embeddings + video_type_embeddings

        # positional embedding
        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (N, L, D)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-12):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)
        self.decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


def contrastive_loss(hidden1: torch.Tensor,
                              hidden2: torch.Tensor,
                              hidden_norm: bool = True,
                              temperature: float = 1.0):
    """
    hidden1: (batch_size, dim)
    hidden2: (batch_size, dim)
    """
    batch_size, hidden_dim = hidden1.shape

    if hidden_norm:
        hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0, batch_size).to(device=hidden1.device)
    masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(
        device=hidden1.device, dtype=torch.float)

    logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
    logits_aa = logits_aa - masks * 1e9
    logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
    logits_bb = logits_bb - masks * 1e9
    logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
    logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)

    loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
    loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
    loss = loss_a + loss_b
    return loss

def cosine_loss(out_A, out_B, input_labels):
    sim = torch.cosine_similarity(out_A, out_B, dim=1)
    res = torch.where(input_labels == 1, 1 - sim, sim)
    caption_loss = sum(res * input_labels)
    return caption_loss, sim

class BertAttention_cross(nn.Module):
    def __init__(self, config):
        super(BertAttention_cross, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q, kv):
        self_output = self.self(q, kv, kv)
        attention_output = self.output(self_output, q)
        return attention_output


class BertLayerNoMemory_Cross(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory_Cross, self).__init__()
        self.config = config
        self.attention = BertAttention_cross(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, q, kv):
        attention_output = self.attention(q, kv)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertDecoder(nn.Module):
    def __init__(self, config, layers):
        super(BertDecoder, self).__init__()
        self.layers = layers
        self.layer_ca = nn.ModuleList([BertLayerNoMemory_Cross(config) for _ in range(self.layers * 2)])

    def forward(self, hidden_states, hidden_states2, guide_states):
        for i in range(self.layers):
            if guide_states != None:
                hidden_states = self.layer_ca[i](hidden_states, guide_states.unsqueeze(0).repeat(hidden_states.size(0), 1, 1))
            hidden_states = self.layer_ca[i+2](hidden_states, hidden_states2)
        return hidden_states