"""
FEDformer implementation for inference, following the FEDformer (Fourier version) spec
from `long_sequence_forecasting_transformers_outlines.md`, section 3.

Includes the Autoformer-style decomposition modules (moving_avg, series_decomp,
series_decomp_multi, my_Layernorm), the frequency-domain attention blocks
(FourierBlock, FourierCrossAttention), the shared encoder/decoder stack, and the
full FEDformer model. The `version='Wavelets'` variant is not implemented (marked
optional/advanced in the outline); only `version='Fourier'` is supported here.

Random frequency-mode selection (`get_frequency_modes`) is done exactly once per
module, inside `__init__`, and frozen for the model's lifetime as required by the
outline's "Common Pitfalls" section.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Autoformer-style decomposition modules
# ---------------------------------------------------------------------------

class moving_avg(nn.Module):
    """Trend extractor via 1-D average pooling with replicate padding (output length == input length)."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - ((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Splits x into (seasonal residual, moving-average trend) using a single kernel size."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """Combines several moving averages via a learnable, input-dependent softmax gate."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.ModuleList([moving_avg(k, stride=1) for k in kernel_size])
        self.layer = nn.Linear(1, len(kernel_size))

    def forward(self, x):
        # x: [B, L, C]
        means = [f(x).unsqueeze(-1) for f in self.moving_avg]           # each [B, L, C, 1]
        moving_mean = torch.cat(means, dim=-1)                          # [B, L, C, K]
        w = F.softmax(self.layer(x.unsqueeze(-1)), dim=-1)              # [B, L, C, K]
        moving_mean = torch.sum(moving_mean * w, dim=-1)                # [B, L, C]
        res = x - moving_mean
        return res, moving_mean


class my_Layernorm(nn.Module):
    """LayerNorm followed by removal of the per-channel temporal mean."""

    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, L, d_model]
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1, keepdim=True).repeat(1, x.shape[1], 1)
        return x_hat - bias


# ---------------------------------------------------------------------------
# Embeddings (no positional encoding, per DataEmbedding_wo_pos)
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """Conv1d value embedding with circular padding and kaiming-normal init."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                    padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [B, L, c_in]
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class TimeFeatureEmbedding(nn.Module):
    """Linear projection of continuous time features (embed_type='timeF')."""

    FREQ_MAP = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

    def __init__(self, d_model, freq='h'):
        super().__init__()
        d_inp = self.FREQ_MAP[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    """Value embedding + temporal embedding, no positional encoding."""

    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.05):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        # x: [B, L, c_in], x_mark: [B, L, d_mark]
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Frequency-domain attention (Fourier version)
# ---------------------------------------------------------------------------

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """Returns a sorted list of `modes` rfft bin indices in [0, seq_len // 2)."""
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:  # 'low'
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(nn.Module):
    """Frequency Enhanced Block (FEB-f) -- used as encoder and decoder self-attention."""

    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random', n_heads=8):
        super().__init__()
        # Random mode selection happens once here, at construction time, and is frozen thereafter.
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        self.n_heads = n_heads
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index), dtype=torch.cfloat))

    def forward(self, q, k, v, mask=None):
        # q, k, v: [B, L, H, E]; k and v are ignored (per reference implementation).
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)                                   # [B, H, E, L]
        x_ft = torch.fft.rfft(x, dim=-1)                            # [B, H, E, L//2+1]
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            # NOTE: the selected bin `i` is written to the compact output bin `wi`
            # (not scattered back to `i`) -- reproduces the released code verbatim.
            out_ft[:, :, :, wi] = torch.einsum("bhi,hio->bho", x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1))                   # [B, H, E, L]
        return (x, None)


class FourierCrossAttention(nn.Module):
    """Frequency Enhanced Attention (FEA-f) -- used as decoder cross-attention."""

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv,
                 modes=64, mode_select_method='random', activation='tanh', n_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.index_q = get_frequency_modes(seq_len_q, modes, mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes, mode_select_method)
        self.n_heads = n_heads
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index_q), dtype=torch.cfloat))

    def forward(self, q, k, v, mask=None):
        # q: [B, L_q, H, E], k/v: [B, L_kv, H, E]. `v` is unused (per reference implementation).
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)                                  # [B, H, E, L_q]
        xk = k.permute(0, 2, 3, 1)                                  # [B, H, E, L_kv]

        xq_ft = torch.fft.rfft(xq, dim=-1)                          # [B, H, E, L_q//2+1]
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft = torch.fft.rfft(xk, dim=-1)                          # [B, H, E, L_kv//2+1]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xk.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)    # [B, H, M_q, M_kv]
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise ValueError(f"unknown activation '{self.activation}'")

        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)   # [B, H, E, M_q] (uses xk_ft_, not xv)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)  # [B, H, E, M_q]
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]                  # scatter back to original bins
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))  # [B, H, E, L_q]
        return (out, None)


class AutoCorrelationLayer(nn.Module):
    """Multi-head wrapper around a frequency-domain attention block."""

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(q, k, v, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ---------------------------------------------------------------------------
# Encoder / Decoder stacks
# ---------------------------------------------------------------------------

def _make_decomp(moving_avg_cfg):
    """Returns a factory producing a fresh decomposition module (never share instances)."""
    if isinstance(moving_avg_cfg, list):
        return lambda: series_decomp_multi(moving_avg_cfg)
    return lambda: series_decomp(moving_avg_cfg)


class EncoderLayer(nn.Module):
    """Self-attention + FFN, seasonal-only residual stream (trend is discarded)."""

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        make_decomp = _make_decomp(moving_avg)
        self.decomp1 = make_decomp()
        self.decomp2 = make_decomp()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, attn_mask=None):
        # x: [B, L_enc, d_model]
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    """Self-attention + cross-attention + FFN, accumulating a trend increment from all three sub-layers."""

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        make_decomp = _make_decomp(moving_avg)
        self.decomp1 = make_decomp()
        self.decomp2 = make_decomp()
        self.decomp3 = make_decomp()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=3, stride=1, padding=1,
                                     padding_mode='circular', bias=False)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x: [B, L_dec, d_model]; cross (encoder output): [B, L_enc, d_model]
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class FEDformer(nn.Module):
    """FEDformer forecasting model (Fourier version)."""

    def __init__(self, enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, pred_len=96,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 moving_avg=[24], modes=64, mode_select='random', version='Fourier',
                 dropout=0.05, embed='timeF', freq='h', activation='gelu',
                 cross_activation='tanh', output_attention=False):
        super().__init__()
        if version != 'Fourier':
            raise NotImplementedError("only version='Fourier' is implemented; 'Wavelets' is optional/advanced")

        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        ks = moving_avg
        self.decomp = series_decomp_multi(ks) if isinstance(ks, list) else series_decomp(ks)

        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        encoder_self_att = FourierBlock(in_channels=d_model, out_channels=d_model,
                                         seq_len=seq_len, modes=modes,
                                         mode_select_method=mode_select, n_heads=n_heads)
        decoder_self_att = FourierBlock(in_channels=d_model, out_channels=d_model,
                                         seq_len=seq_len // 2 + pred_len, modes=modes,
                                         mode_select_method=mode_select, n_heads=n_heads)
        decoder_cross_att = FourierCrossAttention(in_channels=d_model, out_channels=d_model,
                                                   seq_len_q=seq_len // 2 + pred_len,
                                                   seq_len_kv=seq_len, modes=modes,
                                                   mode_select_method=mode_select,
                                                   activation=cross_activation, n_heads=n_heads)

        self.encoder = Encoder(
            [EncoderLayer(AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
                          d_model, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation)
             for _ in range(e_layers)],
            norm_layer=my_Layernorm(d_model))

        self.decoder = Decoder(
            [DecoderLayer(AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                          AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
                          d_model, c_out, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation)
             for _ in range(d_layers)],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [B, L_enc, enc_in], x_dec: [B, L_label+L_pred, dec_in]
        mean = x_enc.mean(1).unsqueeze(1).repeat(1, self.pred_len, 1)         # [B, L_pred, enc_in]
        seasonal_init, trend_init = self.decomp(x_enc)                       # each [B, L_enc, enc_in]

        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)          # [B, L_dec, enc_in]
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))  # [B, L_dec, enc_in]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)                      # [B, L_enc, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)              # [B, L_dec, d_model]
        seasonal_part, trend_part = self.decoder(dec_out, enc_out,
                                                   x_mask=dec_self_mask,
                                                   cross_mask=dec_enc_mask,
                                                   trend=trend_init)
        dec_out = trend_part + seasonal_part                                 # [B, L_dec, c_out]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]                                # [B, L_pred, c_out]


# ---------------------------------------------------------------------------
# Builder + inference helpers
# ---------------------------------------------------------------------------

def build_fedformer(**overrides):
    """
    Builds a FEDformer model using the outline's Default Hyperparameters
    (Fourier version, ETT-style multivariate config). Any keyword can be overridden.
    """
    defaults = dict(
        enc_in=7,
        dec_in=7,
        c_out=7,
        seq_len=96,
        label_len=48,
        pred_len=96,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=[24],
        modes=64,
        mode_select='random',
        version='Fourier',
        dropout=0.05,
        embed='timeF',
        freq='h',
        activation='gelu',
        cross_activation='tanh',
        output_attention=False,
    )
    defaults.update(overrides)
    return FEDformer(**defaults)


def run_inference(x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                   model=None, seed=42, device="cpu", **model_overrides):
    """
    Runs a single deterministic forward pass through FEDformer and returns the
    [B, L_pred, c_out] forecast tensor. Missing inputs are generated as random
    tensors with shapes matching the model's default hyperparameters.
    """
    torch.manual_seed(seed)

    if model is None:
        model = build_fedformer(**model_overrides)
    model = model.to(device)
    model.eval()

    batch_size = 32
    seq_len = model.seq_len
    label_len = model.label_len
    pred_len = model.pred_len
    enc_in = model.enc_embedding.value_embedding.tokenConv.in_channels
    dec_in = model.dec_embedding.value_embedding.tokenConv.in_channels
    d_mark = model.enc_embedding.temporal_embedding.embed.in_features

    if x_enc is None:
        x_enc = torch.randn(batch_size, seq_len, enc_in, device=device)
    if x_mark_enc is None:
        x_mark_enc = torch.randn(batch_size, seq_len, d_mark, device=device)
    if x_dec is None:
        x_dec = torch.randn(batch_size, label_len + pred_len, dec_in, device=device)
    if x_mark_dec is None:
        x_mark_dec = torch.randn(batch_size, label_len + pred_len, d_mark, device=device)

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    return output


if __name__ == "__main__":
    out = run_inference()
    print(out.shape)
