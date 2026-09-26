"""
Autoformer (arXiv:2106.13008) implemented in PyTorch, following the module
specs in `long_sequence_forecasting_transformers_outlines.md` section 2.

Self-contained, inference-only: builds a default Autoformer and runs a
single forward pass on random (or supplied) tensors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. moving_avg / series_decomp / my_Layernorm
# ---------------------------------------------------------------------------

class moving_avg(nn.Module):
    """Moving average block, edge-padded so the trend keeps the input length."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x: [B, L, D]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Splits x into (seasonal, trend). Order matters: seasonal first."""

    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):  # x: [B, L, D]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  # (seasonal, trend)


class my_Layernorm(nn.Module):
    """LayerNorm specialized for the seasonal part: removes the temporal mean."""

    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):  # x: [B, L, d_model]
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


# ---------------------------------------------------------------------------
# 2. Embedding (no positional encoding)
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                    padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):  # x: [B, L, c_in]
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # [B, L, d_model]


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super().__init__()
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):  # x: [B, L, d_inp]
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    """Value (token conv) + temporal embedding, no positional encoding."""

    def __init__(self, c_in, d_model, embed_type="timeF", freq="h", dropout=0.05):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        # Only timeF is supported here (matches this repo's default configuration).
        self.temporal_embedding = TimeFeatureEmbedding(d_model, embed_type, freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):  # x: [B, L, c_in], x_mark: [B, L, d_mark]
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. AutoCorrelation + AutoCorrelationLayer
# ---------------------------------------------------------------------------

class AutoCorrelation(nn.Module):
    """
    FFT-based Auto-Correlation mechanism, replacing self-attention.
    `mask_flag`, `scale`, `attention_dropout` are stored but unused in forward
    (kept only for API compatibility with the reference implementation).
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        # values: [B, H, D, L]   corr: [B, H, E, L]
        head, channel, length = values.shape[1], values.shape[2], values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [B, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)  # [B, top_k]
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                .repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch, head, channel, length = values.shape
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        weights, delay = torch.topk(mean_value, top_k, dim=-1)  # [B, top_k], [B, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                .repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                .repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        batch, head, channel, length = values.shape
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)  # [B, H, E, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # FFT must run on the time axis: permute [B, L, H, E] -> [B, H, E, L] first.
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)  # [B, H, E, L]

        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
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

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ---------------------------------------------------------------------------
# 4. Encoder / Decoder
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1,
                 activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):  # x: [B, L_enc, d_model]
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)  # keep seasonal, discard trend
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)  # keep seasonal, discard trend
        return res, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):  # x: [B, L_enc, d_model]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode="circular", bias=False)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x: [B, L_dec, d_model], cross: [B, L_enc, d_model]
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend  # x: [B, L_dec, d_model], residual_trend: [B, L_dec, c_out]


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # x: [B, L_dec, d_model], cross: [B, L_enc, d_model], trend: [B, L_dec, c_out]
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend  # (seasonal_part, trend_part)


# ---------------------------------------------------------------------------
# 5. Full model assembly
# ---------------------------------------------------------------------------

class AutoformerConfig:
    """Plain container for Autoformer hyperparameters."""

    def __init__(self, seq_len=96, label_len=48, pred_len=96, output_attention=False,
                 moving_avg=25, enc_in=7, dec_in=7, c_out=7, d_model=512, n_heads=8,
                 e_layers=2, d_layers=1, d_ff=2048, factor=1, dropout=0.05,
                 activation="gelu", embed="timeF", freq="h"):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.moving_avg = moving_avg
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.embed = embed
        self.freq = freq


class Autoformer(nn.Module):
    """Encoder-decoder Transformer with series decomposition and Auto-Correlation."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.decomp = series_decomp(configs.moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model,
                                                   configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model,
                                                   configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor,
                                     attention_dropout=configs.dropout,
                                     output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                moving_avg=configs.moving_avg, dropout=configs.dropout,
                activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=my_Layernorm(configs.d_model))

        self.decoder = Decoder(
            [DecoderLayer(
                AutoCorrelationLayer(  # self
                    AutoCorrelation(True, configs.factor,
                                     attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                AutoCorrelationLayer(  # cross
                    AutoCorrelation(False, configs.factor,
                                     attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.c_out, configs.d_ff,
                moving_avg=configs.moving_avg, dropout=configs.dropout,
                activation=configs.activation)
             for _ in range(configs.d_layers)],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [B, L_enc, enc_in], x_dec: [B, L_label+L_pred, dec_in] (values unused; shape used)
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)

        dec_out = trend_part + seasonal_part
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]


# Alias matching the reference repo's `models/Autoformer.py::Model`.
Model = Autoformer


# ---------------------------------------------------------------------------
# 6. Builder + inference helper
# ---------------------------------------------------------------------------

def build_autoformer(**overrides):
    """
    Construct an Autoformer with the paper/repo default hyperparameters,
    overridable via keyword args (e.g. build_autoformer(enc_in=8, factor=3)).
    """
    defaults = dict(
        seq_len=96, label_len=48, pred_len=96, output_attention=False,
        moving_avg=25, enc_in=7, dec_in=7, c_out=7, d_model=512, n_heads=8,
        e_layers=2, d_layers=1, d_ff=2048, factor=1, dropout=0.05,
        activation="gelu", embed="timeF", freq="h",
    )
    defaults.update(overrides)
    configs = AutoformerConfig(**defaults)
    return Autoformer(configs)


def run_inference(x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                   model=None, seed=42, device="cpu", **model_overrides):
    """
    Build (if needed) an Autoformer and run one deterministic forward pass.
    Missing inputs are generated as random tensors shaped per the model's
    default hyperparameters. Returns the [B, L_pred, c_out] prediction tensor.
    """
    torch.manual_seed(seed)

    if model is None:
        model = build_autoformer(**model_overrides)
    model = model.to(device)
    model.eval()

    B = 2
    L_enc = model.seq_len
    L_label = model.label_len
    L_pred = model.pred_len
    enc_in = model.enc_embedding.value_embedding.tokenConv.in_channels
    dec_in = model.dec_embedding.value_embedding.tokenConv.in_channels
    d_mark = model.dec_embedding.temporal_embedding.embed.in_features

    if x_enc is None:
        x_enc = torch.randn(B, L_enc, enc_in, device=device)
    if x_mark_enc is None:
        x_mark_enc = torch.randn(B, L_enc, d_mark, device=device)
    if x_dec is None:
        x_dec = torch.randn(B, L_label + L_pred, dec_in, device=device)
    if x_mark_dec is None:
        x_mark_dec = torch.randn(B, L_label + L_pred, d_mark, device=device)

    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    if isinstance(output, tuple):
        output = output[0]
    return output


if __name__ == "__main__":
    out = run_inference()
    print(out.shape)
