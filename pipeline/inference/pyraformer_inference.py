"""
Pyraformer (ICLR 2022) implemented from the project outline
`long_sequence_forecasting_transformers_outlines.md`, section "## 4. Pyraformer".

Pyraformer replaces full self-attention with pyramidal attention (PAM) over a
multi-resolution C-ary tree built on top of the input sequence. A coarser-scale
construction module (CSCM) builds shorter "summary" sequences level by level via
strided convolution; all levels are concatenated into one flat sequence of
`N_nodes` and a single fixed [N_nodes, N_nodes] mask restricts every node to its
`A` intra-scale neighbours, its `C` children, and its 1 parent. Prediction uses
the default "FC" head: gather every finest-scale node's ancestor features across
all scales, take the last position (the appended zero "predict token"), and map
it through one big linear layer to all future steps at once.

Only stdlib + torch/numpy are used; no CSV / dataset loading here (inference-only,
in-memory tensors).
"""

import math
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Embeddings
# ---------------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding, registered as a non-trainable buffer."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B, L, *]
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Value embedding via circular Conv1d (kernel=3), as inherited from Informer."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                    padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):  # [B, L, c_in]
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # [B, L, d_model]


class DataEmbedding(nn.Module):
    """Value + position + time-feature embedding used for the ETT-style default path."""

    def __init__(self, c_in, d_model, dropout=0.05):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = nn.Linear(4, d_model)  # d_inp = 4 (month/day/weekday/hour)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):  # x: [B, L, c_in], x_mark: [B, L, 4]
        x = (self.value_embedding(x) + self.position_embedding(x)
             + self.temporal_embedding(x_mark))
        return self.dropout(x)


class CustomEmbedding(nn.Module):
    """Value + position + time-feature + series-id embedding (electricity / app-flow / synthetic)."""

    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.05):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):  # x_mark: [B, L, temporal_size + 1], last col = series id
        x = (self.value_embedding(x) + self.position_embedding(x)
             + self.temporal_embedding(x_mark[:, :, :-1])
             + self.seqid_embedding(x_mark[:, :, -1].long()))
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. CSCM (Coarser-Scale Construction Module)
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super().__init__()
        self.downConv = nn.Conv1d(c_in, c_in, kernel_size=window_size, stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):  # [B, c_in, T] -> [B, c_in, floor(T / window_size)]
        return self.activation(self.norm(self.downConv(x)))


class Bottleneck_Construct(nn.Module):
    """Default CSCM: project to a bottleneck dim, build the pyramid there, project back up."""

    def __init__(self, d_model, window_size, d_inner):
        super().__init__()
        if not isinstance(window_size, (list, tuple)):
            window_size = [window_size]
        self.conv_layers = nn.ModuleList([ConvLayer(d_inner, w) for w in window_size])
        self.down = nn.Linear(d_model, d_inner)
        self.up = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):  # [B, L, d_model]
        temp = self.down(enc_input).permute(0, 2, 1)  # [B, d_inner, L]
        all_inputs = []
        for conv in self.conv_layers:
            temp = conv(temp)
            all_inputs.append(temp)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)  # [B, sum(coarser lens), d_inner]
        all_inputs = self.up(all_inputs)  # [B, sum(coarser lens), d_model]
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)  # [B, N_nodes, d_model]
        return self.norm(all_inputs)


class Conv_Construct(nn.Module):
    """CSCM variant: no bottleneck, conv stack applied directly at d_model width."""

    def __init__(self, d_model, window_size, d_inner=None):
        super().__init__()
        if not isinstance(window_size, (list, tuple)):
            window_size = [window_size]
        self.conv_layers = nn.ModuleList([ConvLayer(d_model, w) for w in window_size])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):  # [B, L, d_model]
        temp = enc_input.permute(0, 2, 1)  # [B, d_model, L]
        all_inputs = []
        for conv in self.conv_layers:
            temp = conv(temp)
            all_inputs.append(temp)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        return self.norm(all_inputs)


class MaxPooling_Construct(nn.Module):
    """CSCM variant: zero-parameter max pooling per level."""

    def __init__(self, d_model, window_size, d_inner=None):
        super().__init__()
        if not isinstance(window_size, (list, tuple)):
            window_size = [window_size]
        self.pools = nn.ModuleList([nn.MaxPool1d(kernel_size=w) for w in window_size])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp = enc_input.permute(0, 2, 1)
        all_inputs = []
        for pool in self.pools:
            temp = pool(temp)
            all_inputs.append(temp)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        return self.norm(all_inputs)


class AvgPooling_Construct(nn.Module):
    """CSCM variant: zero-parameter average pooling per level."""

    def __init__(self, d_model, window_size, d_inner=None):
        super().__init__()
        if not isinstance(window_size, (list, tuple)):
            window_size = [window_size]
        self.pools = nn.ModuleList([nn.AvgPool1d(kernel_size=w) for w in window_size])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp = enc_input.permute(0, 2, 1)
        all_inputs = []
        for pool in self.pools:
            temp = pool(temp)
            all_inputs.append(temp)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        return self.norm(all_inputs)


CSCM_REGISTRY = {
    "Bottleneck_Construct": Bottleneck_Construct,
    "Conv_Construct": Conv_Construct,
    "MaxPooling_Construct": MaxPooling_Construct,
    "AvgPooling_Construct": AvgPooling_Construct,
}


# ---------------------------------------------------------------------------
# 3. Pyramidal-graph index construction
# ---------------------------------------------------------------------------

def get_mask(input_size, window_size, inner_size, device="cpu"):
    """
    Build the fixed pyramidal-attention mask.

    input_size : length of the finest scale (== L, or L+1 for the FC head)
    window_size: list[int] of length S-1, e.g. [4, 4, 4]
    inner_size : A, the intra-scale neighbourhood size (odd, incl. self)

    Returns:
      mask     : BoolTensor [N_nodes, N_nodes], True == masked out (blocked)
      all_size : list[int] of length S, per-level node counts
    """
    all_size = [input_size]
    for i in range(len(window_size)):
        all_size.append(math.floor(all_size[i] / window_size[i]))
    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)  # 1 == allowed

    # intra-scale: A adjacent nodes (incl. self), clipped at level boundaries
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left = max(i - inner_window, start)
            right = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left:right] = 1

    # inter-scale: children (parents follow automatically by symmetrizing)
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            child_base = start - all_size[layer_idx - 1]
            left = child_base + (i - start) * window_size[layer_idx - 1]
            if i == start + all_size[layer_idx] - 1:  # last node of this level absorbs the tail
                right = start
            else:
                right = child_base + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left:right] = 1
            mask[left:right, i] = 1

    mask = (1 - mask).bool()  # invert: True == MASKED OUT
    return mask, all_size


def refer_points(all_sizes, window_size, device="cpu"):
    """
    For every finest-scale position i, return its flat index at every scale
    (itself, parent, grandparent, ...), used by the FC prediction head.

    Returns LongTensor [1, input_size, S, 1].
    """
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)
    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index
    return indexes.unsqueeze(0).unsqueeze(3).long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """
    Causal mask for the second decoder layer (keys = [encoder_out ; decoder_out]).
    Returns BoolTensor [1, predict_step, K], True == masked out.
    """
    if truncate:
        k_hist = input_size
    else:
        all_size = [input_size]
        for w in window_size:
            all_size.append(math.floor(all_size[-1] / w))
        k_hist = sum(all_size)
    mask = torch.zeros(predict_step, k_hist + predict_step)
    for i in range(predict_step):
        mask[i][: k_hist + i + 1] = 1
    return (1 - mask).bool().unsqueeze(0)


# ---------------------------------------------------------------------------
# 4. Attention / feed-forward building blocks
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q, k: [B, H, Lq, d_k]; v: [B, H, Lv, d_v]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [B, H, Lq, Lk]
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)  # mask: [B, 1, Lq, Lk] bool, True == blocked
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)  # [B, H, Lq, d_v]
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.normalize_before = normalize_before

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        B, Lq = q.size(0), q.size(1)
        Lk, Lv = k.size(1), v.size(1)
        residual = q

        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q).view(B, Lq, n_head, d_k).transpose(1, 2)  # [B, H, Lq, d_k]
        k = self.w_ks(k).view(B, Lk, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(B, Lv, n_head, d_v).transpose(1, 2)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, Lq, Lk]

        out, attn = self.attention(q, k, v, mask=mask)  # [B, H, Lq, d_v]
        out = out.transpose(1, 2).contiguous().view(B, Lq, n_head * d_v)
        out = self.dropout(self.fc(out)) + residual
        if not self.normalize_before:
            out = self.layer_norm(out)
        return out, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, T, d_in]
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = self.dropout(F.gelu(self.w_1(x)))
        x = self.dropout(self.w_2(x)) + residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# 5. Encoder / Decoder layers
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, normalize_before)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout, normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        x, attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        return self.pos_ffn(x), attn


class DecoderLayer(nn.Module):
    """Same internals as EncoderLayer, but Q may differ from K/V (cross-attention)."""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, normalize_before)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout, normalize_before)

    def forward(self, q_input, k_input, v_input, slf_attn_mask=None):
        x, attn = self.slf_attn(q_input, k_input, v_input, mask=slf_attn_mask)
        return self.pos_ffn(x), attn


class Predictor(nn.Module):
    def __init__(self, dim, num_types):
        super().__init__()
        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        return self.linear(data)


# ---------------------------------------------------------------------------
# 6. Encoder (embedding + CSCM + stacked masked attention layers)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.decoder_type = opt.decoder
        self.truncate = opt.truncate

        # NOTE the +1 for the FC head's predict token (Common Pitfall #3)
        size = opt.input_size if opt.decoder == "attention" else opt.input_size + 1
        mask, self.all_size = get_mask(size, opt.window_size, opt.inner_size, opt.device)
        self.register_buffer("mask", mask)

        if opt.decoder == "FC":
            indexes = refer_points(self.all_size, opt.window_size, opt.device)
            self.register_buffer("indexes", indexes)

        self.layers = nn.ModuleList([
            EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v,
                         dropout=opt.dropout, normalize_before=False)
            for _ in range(opt.n_layer)
        ])

        self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

        cscm_cls = CSCM_REGISTRY[opt.cscm]
        self.conv_layers = cscm_cls(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        B = x_enc.size(0)
        d_model = self.enc_embedding.temporal_embedding.out_features

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)  # [B, L(+1), d_model]
        mask = self.mask.unsqueeze(0).expand(B, -1, -1)  # [B, N, N] bool

        seq_enc = self.conv_layers(seq_enc)  # [B, N_nodes, d_model]
        for layer in self.layers:
            seq_enc, _ = layer(seq_enc, mask)  # same mask every layer

        if self.decoder_type == "FC":
            idx = self.indexes.repeat(B, 1, 1, d_model).view(B, -1, d_model)  # [B, (L+1)*S, d_model]
            all_enc = torch.gather(seq_enc, 1, idx)
            seq_enc = all_enc.view(B, self.all_size[0], -1)  # [B, L+1, S*d_model]
        elif self.decoder_type == "attention" and self.truncate:
            seq_enc = seq_enc[:, : self.all_size[0]]

        return seq_enc


class Decoder(nn.Module):
    """Optional 2-layer cross-attention decoder ("prediction module 2")."""

    def __init__(self, opt, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.layers = nn.ModuleList([
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v,
                         dropout=opt.dropout, normalize_before=False),
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v,
                         dropout=opt.dropout, normalize_before=False),
        ])
        self.dec_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

    def forward(self, x_dec, x_mark_dec, refer):
        # x_dec: zeros [B, M, enc_in] ("prediction tokens"); refer: encoder out [B, N, d_model]
        B = x_dec.size(0)
        F_p = self.dec_embedding(x_dec, x_mark_dec)  # [B, M, d_model]
        F_d1, _ = self.layers[0](F_p, refer, refer)  # no mask: free cross-attention to full encoder
        refer_enc = torch.cat([refer, F_d1], dim=1)  # [B, N+M, d_model]
        m = self.mask.expand(B, -1, -1)  # [B, M, N+M]
        F_d2, _ = self.layers[1](F_d1, refer_enc, refer_enc, slf_attn_mask=m)
        return F_d2  # [B, M, d_model]


# ---------------------------------------------------------------------------
# 7. Full model
# ---------------------------------------------------------------------------

class Pyraformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.predict_step = opt.predict_step
        self.decoder_type = opt.decoder
        self.input_size = opt.input_size

        self.encoder = Encoder(opt)

        if opt.decoder == "attention":
            mask = get_subsequent_mask(opt.input_size, opt.window_size, opt.predict_step, opt.truncate)
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.c_out)
        else:  # 'FC' (default)
            S = len(opt.window_size) + 1  # == 4
            self.predictor = Predictor(S * opt.d_model, opt.predict_step * opt.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain=False):
        # x_enc:      [B, L(+1), enc_in]   (already includes the +1 predict token for the FC head)
        # x_mark_enc: [B, L(+1), n_cov]
        # x_dec:      [B, M, enc_in]       (zeros)
        # x_mark_dec: [B, M, n_cov]
        if self.decoder_type == "attention":
            enc_out = self.encoder(x_enc, x_mark_enc)  # [B, N_nodes, d_model]
            dec_out = self.decoder(x_dec, x_mark_dec, enc_out)  # [B, M, d_model]
            if pretrain:
                dec_out = torch.cat([enc_out[:, : self.input_size], dec_out], dim=1)
            pred = self.predictor(dec_out)  # [B, (L+)M, c_out]
        else:  # 'FC'
            enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]  # [B, S*d_model] (the predict token)
            pred = self.predictor(enc_out).view(enc_out.size(0), self.predict_step, -1)  # [B, M, c_out]
        return pred


# ---------------------------------------------------------------------------
# 8. Builder with the outline's Default Hyperparameters
# ---------------------------------------------------------------------------

def build_pyraformer(**overrides):
    """
    Construct a Pyraformer model using the outline's "Default Hyperparameters" table:
    d_model=512, d_inner_hid=512, d_bottleneck=128, d_k=d_v=128, n_head=4 (CLI default),
    n_layer=4, window_size=[4,4,4] (S=4), inner_size=3, CSCM=Bottleneck_Construct,
    decoder='FC', dropout=0.05, truncate=False. `enc_in`/`c_out`=7 (ETT default) and
    `input_size`/`predict_step` are taken from the Notation table's typical values
    (L=168, L_pred=96). Any of these can be overridden via keyword arguments.
    """
    defaults = dict(
        d_model=512,
        d_inner_hid=512,
        d_bottleneck=128,
        d_k=128,
        d_v=128,
        n_head=4,
        n_layer=4,
        window_size=[4, 4, 4],
        inner_size=3,
        cscm="Bottleneck_Construct",
        decoder="FC",
        dropout=0.05,
        truncate=False,
        enc_in=7,
        c_out=7,
        input_size=168,
        predict_step=96,
        device="cpu",
    )
    defaults.update(overrides)
    opt = types.SimpleNamespace(**defaults)
    return Pyraformer(opt)


# ---------------------------------------------------------------------------
# 9. Reproducible inference entry point
# ---------------------------------------------------------------------------

def run_inference(x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                   model=None, seed=42, device="cpu", **model_overrides):
    """
    Build (if needed) a Pyraformer model and run a single deterministic forward pass.

    x_enc      : [B, L, enc_in]        historical observations; random if None
    x_mark_enc : [B, L, 4]             time-feature covariates for the history; random if None
    x_dec      : [B, L_pred, enc_in]   zero "prediction tokens"; zeros if None
    x_mark_dec : [B, L_pred, 4]        future time-feature covariates; random if None

    Returns the forecast tensor of shape [B, L_pred, c_out].
    """
    torch.manual_seed(seed)

    if model is None:
        model = build_pyraformer(device=device, **model_overrides)
    model = model.to(device)
    model.eval()

    input_size = model.input_size
    predict_step = model.predict_step
    enc_in = model.encoder.enc_embedding.value_embedding.tokenConv.in_channels
    n_cov = model.encoder.enc_embedding.temporal_embedding.in_features

    B = 2
    if x_enc is None:
        x_enc = torch.randn(B, input_size, enc_in, device=device)
    if x_mark_enc is None:
        x_mark_enc = torch.rand(B, input_size, n_cov, device=device) - 0.5
    if x_dec is None:
        x_dec = torch.zeros(B, predict_step, enc_in, device=device)
    if x_mark_dec is None:
        x_mark_dec = torch.rand(B, predict_step, n_cov, device=device) - 0.5

    if model.decoder_type == "FC":
        # Caller-side "+1 predict token" step (see outline's Training Loop section):
        # the FC head's encoder expects input_size = L + 1.
        predict_token = torch.zeros(x_enc.size(0), 1, x_enc.size(-1), device=device)
        x_enc = torch.cat([x_enc, predict_token], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, 0:1, :]], dim=1)

    with torch.no_grad():
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec, False)

    return pred


if __name__ == "__main__":
    out = run_inference()
    print(out.shape)
