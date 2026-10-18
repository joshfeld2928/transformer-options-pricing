"""
Pyraformer (ICLR 2022) in JAX/Equinox. FC prediction head and Bottleneck CSCM only.

Port of the PyTorch reference, with these changes:

- The pyramidal mask and refer points are cached numpy constants, not buffers.
- The CSCM strided convs normalise with LayerNorm (``cscm_norm="layer"``)
  rather than BatchNorm, because batch statistics don't fit the one-sample
  vmap design. ``cscm_norm="none"`` drops the norm entirely.
- The FC head gathers only the predict token's ancestors (``refer[-1]``),
  which is all the reference ever reads.

The caller appends the +1 zero predict token, so ``x_enc`` is
``[input_size + 1, enc_in]``. ``x_dec``/``x_mark_dec`` are accepted only to keep
the shared model interface.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .layers import DataEmbedding, Dropout, LayerNorm, Linear, StridedConv1d, gelu, split
from .masks import pyraformer_mask, pyramid_sizes, refer_points

_MASK_VALUE = -1e9


class CSCMConv(eqx.Module):
    down_conv: StridedConv1d
    norm: LayerNorm | None

    def __init__(self, c_in, window, norm, *, key):
        self.down_conv = StridedConv1d(c_in, c_in, window, key=key)
        self.norm = LayerNorm(c_in) if norm == "layer" else None

    def __call__(self, x):
        x = self.down_conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return jax.nn.elu(x)


class BottleneckConstruct(eqx.Module):
    """Project down, build coarser scales by strided conv, project back up, append to the finest scale."""

    conv_layers: list
    down: Linear
    up: Linear
    norm: LayerNorm

    def __init__(self, d_model, window_size, d_inner, norm, *, key):
        keys = jr.split(key, len(window_size) + 2)
        self.conv_layers = [CSCMConv(d_inner, w, norm, key=k) for w, k in zip(window_size, keys[2:])]
        self.down = Linear(d_model, d_inner, key=keys[0])
        self.up = Linear(d_inner, d_model, key=keys[1])
        self.norm = LayerNorm(d_model)

    def __call__(self, x):
        temp = self.down(x)
        levels = []
        for conv in self.conv_layers:
            temp = conv(temp)
            levels.append(temp)
        coarse = self.up(jnp.concatenate(levels, axis=0))
        return self.norm(jnp.concatenate([x, coarse], axis=0))  # [N_nodes, d_model]


class MultiHeadAttention(eqx.Module):
    """Post-norm masked multi-head attention."""

    w_qs: Linear
    w_ks: Linear
    w_vs: Linear
    fc: Linear
    layer_norm: LayerNorm
    attn_dropout: Dropout
    dropout: Dropout
    n_head: int = eqx.field(static=True)
    d_k: int = eqx.field(static=True)

    def __init__(self, n_head, d_model, d_k, d_v, dropout, *, key):
        kq, kk, kv, kf = jr.split(key, 4)
        self.w_qs = Linear(d_model, n_head * d_k, key=kq, bias=False, init="xavier_uniform")
        self.w_ks = Linear(d_model, n_head * d_k, key=kk, bias=False, init="xavier_uniform")
        self.w_vs = Linear(d_model, n_head * d_v, key=kv, bias=False, init="xavier_uniform")
        self.fc = Linear(n_head * d_v, d_model, key=kf, init="xavier_uniform")
        self.layer_norm = LayerNorm(d_model, eps=1e-6)
        self.attn_dropout = Dropout(dropout)
        self.dropout = Dropout(dropout)
        self.n_head, self.d_k = n_head, d_k

    def __call__(self, x, mask, *, key=None):
        k1, k2 = split(key, 2)
        n, h = x.shape[0], self.n_head
        q = self.w_qs(x).reshape(n, h, -1).transpose(1, 0, 2)  # [H, N, d_k]
        k = self.w_ks(x).reshape(n, h, -1).transpose(1, 0, 2)
        v = self.w_vs(x).reshape(n, h, -1).transpose(1, 0, 2)
        scores = (q / self.d_k ** 0.5) @ k.transpose(0, 2, 1)
        scores = jnp.where(mask[None], _MASK_VALUE, scores)
        attn = self.attn_dropout(jax.nn.softmax(scores, axis=-1), key=k1)
        out = (attn @ v).transpose(1, 0, 2).reshape(n, -1)
        return self.layer_norm(self.dropout(self.fc(out), key=k2) + x)


class PositionwiseFeedForward(eqx.Module):
    w_1: Linear
    w_2: Linear
    layer_norm: LayerNorm
    dropout: Dropout

    def __init__(self, d_in, d_hid, dropout, *, key):
        k1, k2 = jr.split(key)
        self.w_1 = Linear(d_in, d_hid, key=k1)
        self.w_2 = Linear(d_hid, d_in, key=k2)
        self.layer_norm = LayerNorm(d_in, eps=1e-6)
        self.dropout = Dropout(dropout)

    def __call__(self, x, *, key=None):
        k1, k2 = split(key, 2)
        y = self.dropout(gelu(self.w_1(x)), key=k1)
        return self.layer_norm(self.dropout(self.w_2(y), key=k2) + x)


class EncoderLayer(eqx.Module):
    slf_attn: MultiHeadAttention
    pos_ffn: PositionwiseFeedForward

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout, *, key):
        k1, k2 = jr.split(key)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, key=k1)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout, key=k2)

    def __call__(self, x, mask, *, key=None):
        k1, k2 = split(key, 2)
        return self.pos_ffn(self.slf_attn(x, mask, key=k1), key=k2)


class Pyraformer(eqx.Module):
    enc_embedding: DataEmbedding
    conv_layers: BottleneckConstruct
    layers: list
    predictor: Linear
    input_size: int = eqx.field(static=True)
    predict_step: int = eqx.field(static=True)
    window_size: tuple[int, ...] = eqx.field(static=True)
    inner_size: int = eqx.field(static=True)

    def __init__(self, *, key, d_model, d_inner_hid, d_bottleneck, d_k, d_v, n_head, n_layer, window_size,
                 inner_size, dropout, enc_in, c_out, input_size, predict_step, cscm_norm="layer"):
        if cscm_norm not in ("layer", "none"):
            raise NotImplementedError(f"cscm_norm={cscm_norm!r}; use 'layer' or 'none'")
        keys = jr.split(key, n_layer + 3)
        self.input_size, self.predict_step = input_size, predict_step
        self.window_size, self.inner_size = tuple(window_size), inner_size
        self.enc_embedding = DataEmbedding(enc_in, d_model, 4, key=keys[0], dropout=dropout,
                                           token_bias=False, temporal_bias=True, positional=True)
        self.conv_layers = BottleneckConstruct(d_model, self.window_size, d_bottleneck, cscm_norm, key=keys[1])
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, key=k) for k in keys[3:]]
        n_scales = len(self.window_size) + 1
        self.predictor = Linear(n_scales * d_model, predict_step * c_out, key=keys[2], bias=False,
                                init="xavier_normal")

    def __call__(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, *, key=None):
        size = self.input_size + 1  # +1 predict token for the FC head
        if x_enc.shape[0] != size:
            raise ValueError(f"x_enc length {x_enc.shape[0]} != input_size + 1 = {size}")
        k_emb, k_layers = split(key, 2)
        mask = pyraformer_mask(size, self.window_size, self.inner_size)

        x = self.conv_layers(self.enc_embedding(x_enc, x_mark_enc, key=k_emb))
        for layer, k in zip(self.layers, split(k_layers, len(self.layers))):
            x = layer(x, mask, key=k)

        # The predict token's node at every scale, concatenated.
        token = x[refer_points(size, self.window_size)[-1]].reshape(-1)  # [S * d_model]
        return self.predictor(token).reshape(self.predict_step, -1)


PYRAFORMER_DEFAULTS = dict(
    d_model=512, d_inner_hid=512, d_bottleneck=128, d_k=128, d_v=128, n_head=4, n_layer=4,
    window_size=[4, 4, 4], inner_size=3, cscm="Bottleneck_Construct", decoder="FC", dropout=0.05,
    truncate=False, enc_in=7, c_out=7, input_size=168, predict_step=96, cscm_norm="layer",
)


def build_pyraformer(*, key=None, **overrides):
    """Pyraformer (FC head, bottleneck CSCM) with the outline defaults, overridable by keyword."""
    cfg = {**PYRAFORMER_DEFAULTS, **overrides}
    cfg.pop("device", None)
    if cfg.pop("cscm") != "Bottleneck_Construct" or cfg.pop("decoder") != "FC":
        raise NotImplementedError("only cscm='Bottleneck_Construct' with decoder='FC' is supported")
    cfg.pop("truncate")  # only affects the attention decoder
    pyramid_sizes(cfg["input_size"] + 1, tuple(cfg["window_size"]))  # validate early
    return Pyraformer(**cfg, key=jr.PRNGKey(0) if key is None else key)


if __name__ == "__main__":
    from . import smoke_test

    smoke_test("pyraformer")
