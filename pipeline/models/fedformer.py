"""
FEDformer (Fourier version, arXiv:2201.12740) in JAX/Equinox.

Port of the PyTorch reference, with these changes:

- Frequency modes are drawn from a seeded ``np.random.default_rng(mode_seed)``
  rather than global numpy state, so a model can be rebuilt exactly from its
  build kwargs. Pass ``fixed_modes`` to pin them explicitly.
- Complex weights are stored as separate real/imaginary float32 arrays.
- The decoder blocks are sized for the real decoder length
  ``label_len + pred_len``. The reference used ``seq_len // 2 + pred_len``,
  which only agrees when ``label_len == seq_len // 2``.
- Each layer gets its own Fourier block (the reference shares one instance
  across layers); every layer uses the same mode indices.

Quirks kept from the released code: FourierBlock writes selected bins into
compact output bins 0..M-1, cross-attention uses the keys in place of values,
and the ``[H, E, L]`` block output is reshaped row-major into ``[L, H*E]``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .autoformer import CorrelationLayer, _assemble
from .masks import get_frequency_modes


def _complex_weight(key, shape, scale):
    kr, ki = jr.split(key)
    return scale * jr.uniform(kr, shape, jnp.float32), scale * jr.uniform(ki, shape, jnp.float32)


class FourierBlock(eqx.Module):
    """Frequency Enhanced Block (FEB-f): self-attention replacement. q: [L, H, E] -> [H, E, L]."""

    w_re: jax.Array  # [H, E, O, M]
    w_im: jax.Array
    index: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, d_model, n_heads, index, *, key):
        shape = (n_heads, d_model // n_heads, d_model // n_heads, len(index))
        self.w_re, self.w_im = _complex_weight(key, shape, 1.0 / (d_model * d_model))
        self.index = tuple(index)

    def __call__(self, q, k, v, *, key=None):
        length = q.shape[0]
        x_ft = jnp.fft.rfft(q.transpose(1, 2, 0), axis=-1)  # [H, E, L//2+1]
        sel = x_ft[..., np.asarray(self.index)]  # [H, E, M]
        out = jnp.einsum("him,hiom->hom", sel, jax.lax.complex(self.w_re, self.w_im))
        # Compact bins 0..M-1, as in the released code.
        out_ft = jnp.pad(out, ((0, 0), (0, 0), (0, length // 2 + 1 - len(self.index))))
        return jnp.fft.irfft(out_ft, n=length, axis=-1)  # [H, E, L]


class FourierCrossAttention(eqx.Module):
    """Frequency Enhanced Attention (FEA-f): decoder cross-attention. -> [H, E, L_q]."""

    w_re: jax.Array
    w_im: jax.Array
    index_q: tuple[int, ...] = eqx.field(static=True)
    index_kv: tuple[int, ...] = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(self, d_model, n_heads, index_q, index_kv, activation, *, key):
        if activation not in ("tanh", "softmax"):
            raise ValueError(f"unknown activation {activation!r}")
        shape = (n_heads, d_model // n_heads, d_model // n_heads, len(index_q))
        self.w_re, self.w_im = _complex_weight(key, shape, 1.0 / (d_model * d_model))
        self.index_q, self.index_kv = tuple(index_q), tuple(index_kv)
        self.activation = activation
        self.channels = d_model

    def __call__(self, q, k, v, *, key=None):
        length = q.shape[0]
        idx_q, idx_kv = np.asarray(self.index_q), np.asarray(self.index_kv)
        xq_ft = jnp.fft.rfft(q.transpose(1, 2, 0), axis=-1)[..., idx_q]  # [H, E, Mq]
        xk_ft = jnp.fft.rfft(k.transpose(1, 2, 0), axis=-1)[..., idx_kv]  # [H, E, Mkv]

        xqk = jnp.einsum("hex,hey->hxy", xq_ft, xk_ft)
        if self.activation == "tanh":
            xqk = jnp.tanh(xqk)
        else:
            xqk = jax.lax.complex(jax.nn.softmax(jnp.abs(xqk), axis=-1), jnp.zeros(xqk.shape, jnp.float32))

        xqkv = jnp.einsum("hxy,hey->hex", xqk, xk_ft)  # keys stand in for values (reference behaviour)
        xqkvw = jnp.einsum("hex,heox->hox", xqkv, jax.lax.complex(self.w_re, self.w_im))
        out_ft = jnp.zeros(xqkvw.shape[:2] + (length // 2 + 1,), xqkvw.dtype).at[..., idx_q].set(xqkvw)
        return jnp.fft.irfft(out_ft / self.channels / self.channels, n=length, axis=-1)


FEDFORMER_DEFAULTS = dict(
    enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, pred_len=96, d_model=512, n_heads=8,
    e_layers=2, d_layers=1, d_ff=2048, moving_avg=[24], modes=64, mode_select="random",
    version="Fourier", dropout=0.05, embed="timeF", freq="h", activation="gelu",
    cross_activation="tanh", output_attention=False, mode_seed=0, fixed_modes=None,
)


def frequency_modes(cfg):
    """Mode indices for each Fourier block, from ``fixed_modes`` or ``mode_seed``."""
    if cfg.get("fixed_modes"):
        return {name: tuple(v) for name, v in cfg["fixed_modes"].items()}
    rng = np.random.default_rng(cfg["mode_seed"])
    dec_len = cfg["label_len"] + cfg["pred_len"]
    pick = lambda n: get_frequency_modes(n, cfg["modes"], cfg["mode_select"], rng)  # noqa: E731
    return {"encoder_self": pick(cfg["seq_len"]), "decoder_self": pick(dec_len),
            "cross_q": pick(dec_len), "cross_kv": pick(cfg["seq_len"])}


def build_fedformer(*, key=None, **overrides):
    """FEDformer with the paper defaults, overridable by keyword."""
    cfg = {**FEDFORMER_DEFAULTS, **overrides}
    if cfg["version"] != "Fourier":
        raise NotImplementedError("only version='Fourier' is implemented")
    if cfg["embed"] != "timeF" or cfg["output_attention"]:
        raise NotImplementedError("only embed='timeF' without attention output is supported")
    key = jr.PRNGKey(0) if key is None else key
    modes = frequency_modes(cfg)
    d, h = cfg["d_model"], cfg["n_heads"]

    def enc_attn(k):
        k1, k2 = jr.split(k)
        return CorrelationLayer(FourierBlock(d, h, modes["encoder_self"], key=k1), d, h, key=k2)

    def dec_self_attn(k):
        k1, k2 = jr.split(k)
        return CorrelationLayer(FourierBlock(d, h, modes["decoder_self"], key=k1), d, h, key=k2)

    def dec_cross_attn(k):
        k1, k2 = jr.split(k)
        inner = FourierCrossAttention(d, h, modes["cross_q"], modes["cross_kv"], cfg["cross_activation"], key=k1)
        return CorrelationLayer(inner, d, h, key=k2)

    return _assemble(enc_attn, dec_self_attn, dec_cross_attn, cfg, key)


if __name__ == "__main__":
    from . import smoke_test

    smoke_test("fedformer")
