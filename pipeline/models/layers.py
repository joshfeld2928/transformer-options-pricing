"""
Shared Equinox building blocks.

Every module works on ONE sample (time on axis 0, no batch axis); callers
``jax.vmap`` over the batch. Initialisations mirror PyTorch's defaults so the
ports start from the same distributions as the reference implementations.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .masks import sinusoid_table

FREQ_MAP = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}

Dropout = eqx.nn.Dropout


def split(key, n):
    """``jr.split`` that passes ``None`` through (inference mode)."""
    return [None] * n if key is None else list(jr.split(key, n))


def gelu(x):
    # torch's F.gelu is the exact erf form; jax defaults to the tanh approximation.
    return jax.nn.gelu(x, approximate=False)


def activation_fn(name):
    return {"gelu": gelu, "relu": jax.nn.relu}[name]


def _uniform(key, shape, bound):
    return jr.uniform(key, shape, jnp.float32, -bound, bound)


def _init_weight(key, shape, fan_in, fan_out, init):
    if init == "torch":  # kaiming_uniform(a=sqrt(5)) == U(+-1/sqrt(fan_in))
        return _uniform(key, shape, 1.0 / math.sqrt(fan_in))
    if init == "kaiming_normal":  # kaiming_normal_(fan_in, leaky_relu, a=0)
        return jr.normal(key, shape, jnp.float32) * math.sqrt(2.0 / fan_in)
    if init == "xavier_uniform":
        return _uniform(key, shape, math.sqrt(6.0 / (fan_in + fan_out)))
    if init == "xavier_normal":
        return jr.normal(key, shape, jnp.float32) * math.sqrt(2.0 / (fan_in + fan_out))
    raise ValueError(f"unknown init {init!r}")


class Linear(eqx.Module):
    """Dense layer over the last axis; any leading shape. Also serves as a 1x1 conv."""

    weight: jax.Array  # [out, in]
    bias: jax.Array | None

    def __init__(self, in_features, out_features, *, key, bias=True, init="torch"):
        wkey, bkey = jr.split(key)
        self.weight = _init_weight(wkey, (out_features, in_features), in_features, out_features, init)
        self.bias = _uniform(bkey, (out_features,), 1.0 / math.sqrt(in_features)) if bias else None

    def __call__(self, x):
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


class LayerNorm(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim, eps=1e-5):
        self.weight = jnp.ones((dim,), jnp.float32)
        self.bias = jnp.zeros((dim,), jnp.float32)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(-1, keepdims=True)
        var = jnp.square(x - mean).mean(-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + self.eps) * self.weight + self.bias


class CircularConv1d(eqx.Module):
    """Conv1d with odd kernel, stride 1 and circular 'same' padding. x: [L, C_in] -> [L, C_out]."""

    weight: jax.Array  # [out, in, k] (torch layout)
    bias: jax.Array | None

    def __init__(self, in_ch, out_ch, kernel_size=3, *, key, bias=True, init="torch"):
        wkey, bkey = jr.split(key)
        fan_in = in_ch * kernel_size
        self.weight = _init_weight(wkey, (out_ch, in_ch, kernel_size), fan_in, out_ch * kernel_size, init)
        self.bias = _uniform(bkey, (out_ch,), 1.0 / math.sqrt(fan_in)) if bias else None

    def __call__(self, x):
        k = self.weight.shape[-1]
        p = k // 2
        length = x.shape[0]
        xp = jnp.pad(x, ((p, p), (0, 0)), mode="wrap")
        windows = jnp.stack([xp[i:i + length] for i in range(k)], axis=1)  # [L, k, C_in]
        y = jnp.einsum("lkc,ock->lo", windows, self.weight)
        return y if self.bias is None else y + self.bias


class StridedConv1d(eqx.Module):
    """Conv1d with kernel == stride == w, no padding. x: [T, C_in] -> [T // w, C_out]."""

    weight: jax.Array  # [out, in, w]
    bias: jax.Array

    def __init__(self, in_ch, out_ch, window, *, key):
        wkey, bkey = jr.split(key)
        bound = 1.0 / math.sqrt(in_ch * window)
        self.weight = _uniform(wkey, (out_ch, in_ch, window), bound)
        self.bias = _uniform(bkey, (out_ch,), bound)

    def __call__(self, x):
        w = self.weight.shape[-1]
        t = x.shape[0] // w
        blocks = x[:t * w].reshape(t, w, x.shape[1])  # [T', w, C_in]
        return jnp.einsum("twc,ocw->to", blocks, self.weight) + self.bias


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class DataEmbedding(eqx.Module):
    """Token conv value embedding + optional sinusoid position + linear time-feature embedding.

    Informer:   token_bias=True,  positional=True,  temporal_bias=True
    Auto/FED:   token_bias=False, positional=False, temporal_bias=False   (DataEmbedding_wo_pos)
    Pyraformer: token_bias=False, positional=True,  temporal_bias=True, d_mark=4
    """

    value_embedding: CircularConv1d
    temporal_embedding: Linear
    dropout: Dropout
    positional: bool = eqx.field(static=True)

    def __init__(self, c_in, d_model, d_mark, *, key, dropout=0.05, token_bias=False,
                 temporal_bias=False, positional=False):
        k1, k2 = jr.split(key)
        self.value_embedding = CircularConv1d(c_in, d_model, 3, key=k1, bias=token_bias, init="kaiming_normal")
        self.temporal_embedding = Linear(d_mark, d_model, key=k2, bias=temporal_bias)
        self.dropout = Dropout(dropout)
        self.positional = positional

    def __call__(self, x, x_mark, *, key=None):
        out = self.value_embedding(x) + self.temporal_embedding(x_mark)
        if self.positional:
            out = out + sinusoid_table(x.shape[0], out.shape[-1])
        return self.dropout(out, key=key)


# ---------------------------------------------------------------------------
# Series decomposition (Autoformer / FEDformer)
# ---------------------------------------------------------------------------

def moving_avg(x, kernel_size):
    """Edge-replicated moving average along time; output length == input length. x: [L, C]."""
    front = kernel_size - 1 - (kernel_size - 1) // 2
    end = (kernel_size - 1) // 2
    xp = jnp.concatenate([jnp.repeat(x[:1], front, 0), x, jnp.repeat(x[-1:], end, 0)], axis=0)
    total = jax.lax.reduce_window(xp, 0.0, jax.lax.add, (kernel_size, 1), (1, 1), "VALID")
    return total / kernel_size


def series_decomp(x, kernel_size):
    """Returns (seasonal, trend)."""
    trend = moving_avg(x, kernel_size)
    return x - trend, trend


class SeriesDecompMulti(eqx.Module):
    """Several moving averages blended by an input-dependent softmax gate (FEDformer)."""

    layer: Linear
    kernel_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, kernel_sizes, *, key):
        self.kernel_sizes = tuple(kernel_sizes)
        self.layer = Linear(1, len(self.kernel_sizes), key=key)

    def __call__(self, x):
        means = jnp.stack([moving_avg(x, k) for k in self.kernel_sizes], axis=-1)  # [L, C, K]
        w = jax.nn.softmax(self.layer(x[..., None]), axis=-1)  # [L, C, K]
        trend = jnp.sum(means * w, axis=-1)
        return x - trend, trend


class Decomp(eqx.Module):
    """Single-kernel decomposition with the same call signature as SeriesDecompMulti."""

    kernel_size: int = eqx.field(static=True)

    def __call__(self, x):
        return series_decomp(x, self.kernel_size)


def make_decomp(moving_avg_cfg, *, key):
    if isinstance(moving_avg_cfg, (list, tuple)):
        return SeriesDecompMulti(moving_avg_cfg, key=key)
    return Decomp(int(moving_avg_cfg))


class MyLayerNorm(eqx.Module):
    """LayerNorm for the seasonal part: subtracts the temporal mean afterwards."""

    layernorm: LayerNorm

    def __init__(self, channels):
        self.layernorm = LayerNorm(channels)

    def __call__(self, x):
        x_hat = self.layernorm(x)
        return x_hat - x_hat.mean(0, keepdims=True)
