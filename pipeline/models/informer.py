"""
Informer (arXiv:2012.07436) in JAX/Equinox.

Port of the PyTorch reference, with these changes:

- ProbSparse key sampling takes an explicit PRNG key. With ``key=None``
  (inference) a fixed key is used, so forecasts are deterministic.
- The distilling ConvLayer (only built when ``e_layers > 1``) uses LayerNorm
  rather than BatchNorm, because batch statistics don't fit the one-sample
  vmap design.
- Only ``embed='timeF'`` is supported.
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
    LayerNorm,
    Linear,
    activation_fn,
    split,
)
from .masks import causal_mask

_MASK_VALUE = -1e9
_INFERENCE_SAMPLE_KEY = jr.PRNGKey(0)


def _sample_keys(key, l_q, n_sample, l_k):
    """Random key indices per query, [L_Q, U]. Module-level so tests can pin it."""
    return jr.randint(key, (l_q, n_sample), 0, l_k)


class FullAttention(eqx.Module):
    """Canonical scaled dot-product attention. q: [L, H, E], k/v: [S, H, E] -> [L, H, D]."""

    dropout: Dropout
    mask_flag: bool = eqx.field(static=True)

    def __call__(self, q, k, v, *, key=None):
        length, _, e = q.shape
        scores = jnp.einsum("lhe,she->hls", q, k)
        if self.mask_flag:
            scores = jnp.where(causal_mask(length)[None], _MASK_VALUE, scores)
        attn = self.dropout(jax.nn.softmax(scores / math.sqrt(e), axis=-1), key=key)
        return jnp.einsum("hls,shd->lhd", attn, v)


class ProbAttention(eqx.Module):
    """ProbSparse self-attention: only the top-u 'active' queries attend; the rest get a lazy context."""

    factor: int = eqx.field(static=True)
    mask_flag: bool = eqx.field(static=True)

    def __call__(self, q, k, v, *, key=None):
        l_q, heads, e = q.shape
        l_k = k.shape[0]
        Q, K, V = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)  # [H, L, E]

        n_sample = min(self.factor * math.ceil(math.log(l_k)), l_k)
        n_top = min(self.factor * math.ceil(math.log(l_q)), l_q)

        # Query sparsity measurement M on a random key sample.
        idx = _sample_keys(_INFERENCE_SAMPLE_KEY if key is None else key, l_q, n_sample, l_k)
        qk_sample = jnp.einsum("hle,hlue->hlu", Q, K[:, idx, :])
        sparsity = qk_sample.max(-1) - qk_sample.sum(-1) / l_k  # [H, L_Q]
        _, top = jax.lax.top_k(sparsity, n_top)  # [H, u]

        q_reduce = jnp.take_along_axis(Q, top[..., None], axis=1)  # [H, u, E]
        scores = jnp.einsum("hue,hse->hus", q_reduce, K) / math.sqrt(e)

        if self.mask_flag:
            if l_q != l_k:
                raise ValueError("causal ProbAttention needs L_Q == L_K")
            context = jnp.cumsum(V, axis=1)
            scores = jnp.where(jnp.asarray(causal_mask(l_q))[top], _MASK_VALUE, scores)
        else:
            context = jnp.broadcast_to(V.mean(1, keepdims=True), (heads, l_q, V.shape[-1]))

        attn = jax.nn.softmax(scores, axis=-1)
        context = context.at[jnp.arange(heads)[:, None], top].set(attn @ V)
        return context.transpose(1, 0, 2)  # [L, H, D]


class AttentionLayer(eqx.Module):
    inner: eqx.Module
    query_projection: Linear
    key_projection: Linear
    value_projection: Linear
    out_projection: Linear
    n_heads: int = eqx.field(static=True)
    mix: bool = eqx.field(static=True)

    def __init__(self, inner, d_model, n_heads, *, key, mix=False):
        kq, kk, kv, ko = jr.split(key, 4)
        d_head = d_model // n_heads
        self.inner = inner
        self.query_projection = Linear(d_model, d_head * n_heads, key=kq)
        self.key_projection = Linear(d_model, d_head * n_heads, key=kk)
        self.value_projection = Linear(d_model, d_head * n_heads, key=kv)
        self.out_projection = Linear(d_head * n_heads, d_model, key=ko)
        self.n_heads = n_heads
        self.mix = mix

    def __call__(self, queries, keys, values, *, key=None):
        length, s, h = queries.shape[0], keys.shape[0], self.n_heads
        q = self.query_projection(queries).reshape(length, h, -1)
        k = self.key_projection(keys).reshape(s, h, -1)
        v = self.value_projection(values).reshape(s, h, -1)
        out = self.inner(q, k, v, key=key)  # [L, H, D]
        if self.mix:
            # Reference quirk: [L, H, D] -> [H, L, D] then viewed as [L, H*D].
            out = out.transpose(1, 0, 2)
        return self.out_projection(out.reshape(length, -1))


class ConvLayer(eqx.Module):
    """Self-attention distilling: circular conv, LayerNorm (BatchNorm in the paper), ELU, max-pool /2."""

    down_conv: CircularConv1d
    norm: LayerNorm

    def __init__(self, c_in, *, key):
        self.down_conv = CircularConv1d(c_in, c_in, 3, key=key)
        self.norm = LayerNorm(c_in)

    def __call__(self, x):
        x = jax.nn.elu(self.norm(self.down_conv(x)))
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (3, 1), (2, 1), ((1, 1), (0, 0)))


class EncoderLayer(eqx.Module):
    attention: AttentionLayer
    conv1: Linear
    conv2: Linear
    norm1: LayerNorm
    norm2: LayerNorm
    dropout: Dropout
    activation: str = eqx.field(static=True)

    def __init__(self, attention, d_model, d_ff, dropout, activation, *, key):
        k1, k2 = jr.split(key)
        self.attention = attention
        self.conv1 = Linear(d_model, d_ff, key=k1)
        self.conv2 = Linear(d_ff, d_model, key=k2)
        self.norm1, self.norm2 = LayerNorm(d_model), LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = activation

    def __call__(self, x, *, key=None):
        ka, k1, k2, k3 = split(key, 4)
        x = self.norm1(x + self.dropout(self.attention(x, x, x, key=ka), key=k1))
        y = self.dropout(activation_fn(self.activation)(self.conv1(x)), key=k2)
        y = self.dropout(self.conv2(y), key=k3)
        return self.norm2(x + y)


class DecoderLayer(eqx.Module):
    self_attention: AttentionLayer
    cross_attention: AttentionLayer
    conv1: Linear
    conv2: Linear
    norm1: LayerNorm
    norm2: LayerNorm
    norm3: LayerNorm
    dropout: Dropout
    activation: str = eqx.field(static=True)

    def __init__(self, self_attention, cross_attention, d_model, d_ff, dropout, activation, *, key):
        k1, k2 = jr.split(key)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = Linear(d_model, d_ff, key=k1)
        self.conv2 = Linear(d_ff, d_model, key=k2)
        self.norm1, self.norm2, self.norm3 = LayerNorm(d_model), LayerNorm(d_model), LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = activation

    def __call__(self, x, cross, *, key=None):
        ks, kc, k1, k2, k3, k4 = split(key, 6)
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, key=ks), key=k1))
        x = self.norm2(x + self.dropout(self.cross_attention(x, cross, cross, key=kc), key=k2))
        y = self.dropout(activation_fn(self.activation)(self.conv1(x)), key=k3)
        y = self.dropout(self.conv2(y), key=k4)
        return self.norm3(x + y)


class Informer(eqx.Module):
    enc_embedding: DataEmbedding
    dec_embedding: DataEmbedding
    encoder_layers: list
    conv_layers: list
    encoder_norm: LayerNorm
    decoder_layers: list
    decoder_norm: LayerNorm
    projection: Linear
    pred_len: int = eqx.field(static=True)

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, *, key, factor=5, d_model=512,
                 n_heads=8, e_layers=3, d_layers=2, d_ff=512, dropout=0.0, attn="prob", embed="timeF",
                 freq="h", activation="gelu", output_attention=False, distil=True, mix=True, device=None):
        if embed != "timeF" or output_attention:
            raise NotImplementedError("only embed='timeF' without attention output is supported")
        if attn not in ("prob", "full"):
            raise ValueError(f"unknown attn {attn!r}")
        keys = iter(jr.split(key, 3 + 3 * e_layers + 3 * d_layers))
        d_mark = FREQ_MAP[freq]

        def self_attn(mask_flag):
            if attn == "prob":
                return ProbAttention(factor=factor, mask_flag=mask_flag)
            return FullAttention(Dropout(dropout), mask_flag=mask_flag)

        self.pred_len = out_len
        emb = dict(dropout=dropout, token_bias=True, temporal_bias=True, positional=True)
        self.enc_embedding = DataEmbedding(enc_in, d_model, d_mark, key=next(keys), **emb)
        self.dec_embedding = DataEmbedding(dec_in, d_model, d_mark, key=next(keys), **emb)
        self.encoder_layers = [
            EncoderLayer(AttentionLayer(self_attn(False), d_model, n_heads, key=next(keys)),
                         d_model, d_ff, dropout, activation, key=next(keys))
            for _ in range(e_layers)
        ]
        self.conv_layers = [ConvLayer(d_model, key=next(keys)) for _ in range(e_layers - 1)] if distil else []
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_layers = [
            DecoderLayer(AttentionLayer(self_attn(True), d_model, n_heads, key=next(keys), mix=mix),
                         AttentionLayer(FullAttention(Dropout(dropout), mask_flag=False), d_model, n_heads,
                                        key=next(keys)),
                         d_model, d_ff, dropout, activation, key=next(keys))
            for _ in range(d_layers)
        ]
        self.decoder_norm = LayerNorm(d_model)
        self.projection = Linear(d_model, c_out, key=next(keys))

    def __call__(self, x_enc, x_mark_enc, x_dec, x_mark_dec, *, key=None):
        k_ee, k_de, k_enc, k_dec = split(key, 4)

        x = self.enc_embedding(x_enc, x_mark_enc, key=k_ee)
        enc_keys = split(k_enc, len(self.encoder_layers))
        # Distilling: a ConvLayer after every attention layer except the last.
        for layer, conv, k in zip(self.encoder_layers, self.conv_layers, enc_keys):
            x = conv(layer(x, key=k))
        for layer, k in list(zip(self.encoder_layers, enc_keys))[len(self.conv_layers):]:
            x = layer(x, key=k)
        enc_out = self.encoder_norm(x)

        x = self.dec_embedding(x_dec, x_mark_dec, key=k_de)
        for layer, k in zip(self.decoder_layers, split(k_dec, len(self.decoder_layers))):
            x = layer(x, enc_out, key=k)
        return self.projection(self.decoder_norm(x))[-self.pred_len:]


INFORMER_DEFAULTS = dict(
    enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, out_len=24, factor=5, d_model=512,
    n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.05, attn="prob", embed="timeF",
    freq="h", activation="gelu", output_attention=False, distil=True, mix=True,
)


def build_informer(*, key=None, **overrides):
    """Informer with the outline's defaults, overridable by keyword."""
    cfg = {**INFORMER_DEFAULTS, **overrides}
    return Informer(**cfg, key=jr.PRNGKey(0) if key is None else key)


if __name__ == "__main__":
    from . import smoke_test

    smoke_test("informer")
