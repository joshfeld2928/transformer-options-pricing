"""
Autoformer (arXiv:2106.13008) in JAX/Equinox.

Port of the PyTorch reference. One deliberate difference: the reference
aggregates time delays chosen from the batch-mean correlation during training
and per-sample delays at inference. A one-sample model can't see the batch, so
the per-sample (inference) path is used in both modes. Training and inference
therefore run the same function.

The decomposition encoder/decoder layers here are shared with FEDformer.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .layers import (
    FREQ_MAP,
    CircularConv1d,
    DataEmbedding,
    Dropout,
    Linear,
    MyLayerNorm,
    activation_fn,
    make_decomp,
    split,
)


class AutoCorrelation(eqx.Module):
    """FFT auto-correlation + top-k time-delay aggregation. q/k/v: [L or S, H, E]."""

    factor: int = eqx.field(static=True)

    def __call__(self, q, k, v, *, key=None):
        length, heads, _ = q.shape
        s = v.shape[0]
        if length > s:
            pad = ((0, length - s), (0, 0), (0, 0))
            k, v = jnp.pad(k, pad), jnp.pad(v, pad)
        else:
            k, v = k[:length], v[:length]

        # Correlation on the time axis: [L, H, E] -> [H, E, L].
        q_fft = jnp.fft.rfft(q.transpose(1, 2, 0), axis=-1)
        k_fft = jnp.fft.rfft(k.transpose(1, 2, 0), axis=-1)
        corr = jnp.fft.irfft(q_fft * jnp.conj(k_fft), n=length, axis=-1)  # [H, E, L]

        top_k = int(self.factor * math.log(length))
        weights, delay = jax.lax.top_k(corr.mean(axis=(0, 1)), top_k)  # [k], [k]
        tmp_corr = jax.nn.softmax(weights)
        idx = (jnp.arange(length)[None, :] + delay[:, None]) % length  # [k, L]
        values = v.transpose(1, 2, 0)  # [H, D, L]
        agg = jnp.einsum("hdkl,k->hdl", values[:, :, idx], tmp_corr)
        return agg.transpose(2, 0, 1)  # [L, H, D]


class CorrelationLayer(eqx.Module):
    """Q/K/V projections around an inner correlation block, then out projection.

    The inner block returns ``[L, H, D]`` (Autoformer) or ``[H, E, L]`` (FEDformer);
    either way it's reshaped row-major to ``[L, H*D]``, matching torch's ``.view``.
    """

    inner: eqx.Module
    query_projection: Linear
    key_projection: Linear
    value_projection: Linear
    out_projection: Linear
    n_heads: int = eqx.field(static=True)

    def __init__(self, inner, d_model, n_heads, *, key):
        kq, kk, kv, ko = jr.split(key, 4)
        d_head = d_model // n_heads
        self.inner = inner
        self.query_projection = Linear(d_model, d_head * n_heads, key=kq)
        self.key_projection = Linear(d_model, d_head * n_heads, key=kk)
        self.value_projection = Linear(d_model, d_head * n_heads, key=kv)
        self.out_projection = Linear(d_head * n_heads, d_model, key=ko)
        self.n_heads = n_heads

    def __call__(self, queries, keys, values, *, key=None):
        length, s, h = queries.shape[0], keys.shape[0], self.n_heads
        q = self.query_projection(queries).reshape(length, h, -1)
        k = self.key_projection(keys).reshape(s, h, -1)
        v = self.value_projection(values).reshape(s, h, -1)
        out = self.inner(q, k, v, key=key)
        return self.out_projection(out.reshape(length, -1))


class DecompEncoderLayer(eqx.Module):
    attention: CorrelationLayer
    conv1: Linear
    conv2: Linear
    decomp1: eqx.Module
    decomp2: eqx.Module
    dropout: Dropout
    activation: str = eqx.field(static=True)

    def __init__(self, attention, d_model, d_ff, moving_avg, dropout, activation, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.attention = attention
        self.conv1 = Linear(d_model, d_ff, key=k1, bias=False)
        self.conv2 = Linear(d_ff, d_model, key=k2, bias=False)
        self.decomp1 = make_decomp(moving_avg, key=k3)
        self.decomp2 = make_decomp(moving_avg, key=k4)
        self.dropout = Dropout(dropout)
        self.activation = activation

    def __call__(self, x, *, key=None):
        ka, k1, k2, k3 = split(key, 4)
        x = x + self.dropout(self.attention(x, x, x, key=ka), key=k1)
        x, _ = self.decomp1(x)
        y = self.dropout(activation_fn(self.activation)(self.conv1(x)), key=k2)
        y = self.dropout(self.conv2(y), key=k3)
        res, _ = self.decomp2(x + y)
        return res


class DecompDecoderLayer(eqx.Module):
    self_attention: CorrelationLayer
    cross_attention: CorrelationLayer
    conv1: Linear
    conv2: Linear
    decomp1: eqx.Module
    decomp2: eqx.Module
    decomp3: eqx.Module
    projection: CircularConv1d
    dropout: Dropout
    activation: str = eqx.field(static=True)

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff, moving_avg, dropout,
                 activation, *, key):
        k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = Linear(d_model, d_ff, key=k1, bias=False)
        self.conv2 = Linear(d_ff, d_model, key=k2, bias=False)
        self.decomp1 = make_decomp(moving_avg, key=k3)
        self.decomp2 = make_decomp(moving_avg, key=k4)
        self.decomp3 = make_decomp(moving_avg, key=k5)
        self.projection = CircularConv1d(d_model, c_out, 3, key=k6, bias=False)
        self.dropout = Dropout(dropout)
        self.activation = activation

    def __call__(self, x, cross, *, key=None):
        ks, kc, k1, k2, k3, k4 = split(key, 6)
        x = x + self.dropout(self.self_attention(x, x, x, key=ks), key=k1)
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, key=kc), key=k2)
        x, trend2 = self.decomp2(x)
        y = self.dropout(activation_fn(self.activation)(self.conv1(x)), key=k3)
        y = self.dropout(self.conv2(y), key=k4)
        x, trend3 = self.decomp3(x + y)
        return x, self.projection(trend1 + trend2 + trend3)


class DecompForecaster(eqx.Module):
    """Shared Autoformer/FEDformer body: decomposition init, encoder, decoder, trend accumulation."""

    decomp: eqx.Module
    enc_embedding: DataEmbedding
    dec_embedding: DataEmbedding
    encoder_layers: list
    encoder_norm: MyLayerNorm
    decoder_layers: list
    decoder_norm: MyLayerNorm
    projection: Linear
    seq_len: int = eqx.field(static=True)
    label_len: int = eqx.field(static=True)
    pred_len: int = eqx.field(static=True)

    def __call__(self, x_enc, x_mark_enc, x_dec, x_mark_dec, *, key=None):
        if x_dec.shape[0] != self.label_len + self.pred_len:
            raise ValueError(f"x_dec length {x_dec.shape[0]} != label_len + pred_len")
        k_ee, k_de, k_enc, k_dec = split(key, 4)
        n_enc, n_dec = len(self.encoder_layers), len(self.decoder_layers)

        mean = jnp.broadcast_to(x_enc.mean(0, keepdims=True), (self.pred_len, x_enc.shape[1]))
        seasonal_init, trend_init = self.decomp(x_enc)
        trend = jnp.concatenate([trend_init[-self.label_len:], mean], axis=0)
        seasonal_init = jnp.concatenate(
            [seasonal_init[-self.label_len:], jnp.zeros((self.pred_len, x_dec.shape[1]), x_enc.dtype)], axis=0)

        enc_out = self.enc_embedding(x_enc, x_mark_enc, key=k_ee)
        for layer, k in zip(self.encoder_layers, split(k_enc, n_enc)):
            enc_out = layer(enc_out, key=k)
        enc_out = self.encoder_norm(enc_out)

        x = self.dec_embedding(seasonal_init, x_mark_dec, key=k_de)
        for layer, k in zip(self.decoder_layers, split(k_dec, n_dec)):
            x, residual_trend = layer(x, enc_out, key=k)
            trend = trend + residual_trend
        seasonal = self.projection(self.decoder_norm(x))
        return (trend + seasonal)[-self.pred_len:]


def _assemble(enc_attn, dec_self_attn, dec_cross_attn, cfg, key):
    """Build a DecompForecaster from per-layer attention factories ``f(key) -> CorrelationLayer``."""
    keys = jr.split(key, 9)
    d_mark = FREQ_MAP[cfg["freq"]]
    if cfg["c_out"] != cfg["enc_in"]:
        raise ValueError("decomposition models add the trend channel-wise: c_out must equal enc_in")

    return DecompForecaster(
        decomp=make_decomp(cfg["moving_avg"], key=keys[0]),
        enc_embedding=DataEmbedding(cfg["enc_in"], cfg["d_model"], d_mark, key=keys[1], dropout=cfg["dropout"]),
        dec_embedding=DataEmbedding(cfg["dec_in"], cfg["d_model"], d_mark, key=keys[2], dropout=cfg["dropout"]),
        encoder_layers=[
            DecompEncoderLayer(enc_attn(ka), cfg["d_model"], cfg["d_ff"], cfg["moving_avg"], cfg["dropout"],
                               cfg["activation"], key=kl)
            for ka, kl in zip(jr.split(keys[3], cfg["e_layers"]), jr.split(keys[4], cfg["e_layers"]))
        ],
        encoder_norm=MyLayerNorm(cfg["d_model"]),
        decoder_layers=[
            DecompDecoderLayer(dec_self_attn(ks), dec_cross_attn(kc), cfg["d_model"], cfg["c_out"], cfg["d_ff"],
                               cfg["moving_avg"], cfg["dropout"], cfg["activation"], key=kl)
            for ks, kc, kl in zip(jr.split(keys[5], cfg["d_layers"]), jr.split(keys[6], cfg["d_layers"]),
                                  jr.split(keys[7], cfg["d_layers"]))
        ],
        decoder_norm=MyLayerNorm(cfg["d_model"]),
        projection=Linear(cfg["d_model"], cfg["c_out"], key=keys[8]),
        seq_len=cfg["seq_len"],
        label_len=cfg["label_len"],
        pred_len=cfg["pred_len"],
    )


AUTOFORMER_DEFAULTS = dict(
    seq_len=96, label_len=48, pred_len=96, moving_avg=25, enc_in=7, dec_in=7, c_out=7,
    d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, factor=1, dropout=0.05,
    activation="gelu", embed="timeF", freq="h", output_attention=False,
)


def build_autoformer(*, key=None, **overrides):
    """Autoformer with the paper defaults, overridable by keyword."""
    cfg = {**AUTOFORMER_DEFAULTS, **overrides}
    if cfg["embed"] != "timeF" or cfg["output_attention"]:
        raise NotImplementedError("only embed='timeF' without attention output is supported")
    key = jr.PRNGKey(0) if key is None else key
    corr = AutoCorrelation(factor=cfg["factor"])

    def attn(k):
        return CorrelationLayer(corr, cfg["d_model"], cfg["n_heads"], key=k)

    return _assemble(attn, attn, attn, cfg, key)


if __name__ == "__main__":
    from . import smoke_test

    smoke_test("autoformer")
