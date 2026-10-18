"""
JAX/Equinox implementations of the four forecasting transformers.

Every model is a one-sample ``eqx.Module`` with the call signature
``model(x_enc, x_mark_enc, x_dec, x_mark_dec, *, key=None) -> [pred_len, c_out]``.
Batch it with ``jax.vmap``. Pass a PRNG key while training (for dropout and
ProbSparse sampling) and run ``eqx.nn.inference_mode(model)`` with ``key=None``
to predict.
"""

import importlib

from ._jax_setup import configure_jax

configure_jax()

MODEL_NAMES = ("autoformer", "fedformer", "informer", "pyraformer")

# Small dims so CPU training stays fast.
SMALL_STANDARD_KWARGS = dict(d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128, dropout=0.05, freq="d")
SMALL_PYRAFORMER_KWARGS = dict(d_model=64, d_inner_hid=64, d_bottleneck=32, d_k=32, d_v=32,
                               n_head=4, n_layer=2, window_size=[3, 3, 3])


def window_kind(name):
    """'pyraformer' models are encoder-only with a +1 predict token; the rest are encoder-decoder."""
    return "pyraformer" if name == "pyraformer" else "standard"


def mark_cols(name):
    """Number of time-feature columns the model's temporal embedding expects."""
    return 4 if name == "pyraformer" else 3


def model_kwargs(name, n_features, seq_len, label_len, pred_len, **overrides):
    """Full builder kwargs for ``name`` on a panel with ``n_features`` columns."""
    if name == "pyraformer":
        kw = dict(enc_in=n_features, c_out=n_features, input_size=seq_len, predict_step=pred_len,
                  **SMALL_PYRAFORMER_KWARGS)
    else:
        kw = dict(enc_in=n_features, dec_in=n_features, c_out=n_features, seq_len=seq_len,
                  label_len=label_len, **SMALL_STANDARD_KWARGS)
        kw["out_len" if name == "informer" else "pred_len"] = pred_len
        if name == "fedformer":
            kw["modes"] = 16
    kw.update(overrides)
    return kw


def build_model(name, *, key, **kwargs):
    """Build ``name`` from its full kwargs (see ``model_kwargs``)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    # Imported lazily so `python -m models.<name>` doesn't import its own module twice.
    builder = getattr(importlib.import_module(f".{name}", __name__), f"build_{name}")
    return builder(key=key, **kwargs)


def random_inputs(name, batch, n_features, seq_len, label_len, pred_len, *, key):
    """Random batched inputs shaped for ``name``, as (x_enc, x_mark_enc, x_dec, x_mark_dec)."""
    import jax.numpy as jnp
    import jax.random as jr

    k1, k2, k3 = jr.split(key, 3)
    marks = mark_cols(name)
    enc_len = seq_len + 1 if name == "pyraformer" else seq_len
    dec_len = pred_len if name == "pyraformer" else label_len + pred_len
    x_enc = jr.normal(k1, (batch, enc_len, n_features))
    x_mark_enc = jr.uniform(k2, (batch, enc_len, marks)) - 0.5
    x_mark_dec = jr.uniform(k3, (batch, dec_len, marks)) - 0.5
    if name == "pyraformer":
        x_enc = x_enc.at[:, -1].set(0.0)  # the +1 predict token
        x_dec = jnp.zeros((batch, dec_len, n_features))
    else:  # last label_len observed rows, then zero placeholders
        x_dec = jnp.concatenate([x_enc[:, seq_len - label_len:], jnp.zeros((batch, pred_len, n_features))], 1)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def smoke_test(name, batch=4, n_features=7, seq_len=60, label_len=30, pred_len=10):
    """Forward pass on random inputs; prints the output shape and checks determinism."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    kwargs = model_kwargs(name, n_features, seq_len, label_len, pred_len)
    model = build_model(name, key=jr.PRNGKey(0), **kwargs)
    inputs = random_inputs(name, batch, n_features, seq_len, label_len, pred_len, key=jr.PRNGKey(1))

    infer_model = eqx.nn.inference_mode(model)
    infer = eqx.filter_jit(lambda m, *xs: jax.vmap(m)(*xs))
    out = infer(infer_model, *inputs)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    print(f"{name}: output shape {tuple(out.shape)}, {n_params:,} parameters")
    print("Deterministic in inference mode:", bool(jnp.array_equal(out, infer(infer_model, *inputs))))

    train = eqx.filter_jit(lambda m, keys, *xs: jax.vmap(lambda *a: m(*a[1:], key=a[0]))(keys, *xs))
    a = train(model, jr.split(jr.PRNGKey(2), batch), *inputs)
    b = train(model, jr.split(jr.PRNGKey(3), batch), *inputs)
    print("Training-mode output varies with key:", not bool(jnp.allclose(a, b)))
