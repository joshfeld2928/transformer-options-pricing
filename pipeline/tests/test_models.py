"""Shape, gradient, determinism, recompilation and overfitting checks for every model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from conftest import TINY, tiny_kwargs

import models

ALL = list(models.MODEL_NAMES)


def _inputs(name, batch=4, **dims):
    d = {**TINY, **dims}
    return models.random_inputs(name, batch, d["n_features"], d["seq_len"], d["label_len"], d["pred_len"],
                                key=jr.PRNGKey(1))


def _train_forward(model, xs, key):
    keys = jr.split(key, xs[0].shape[0])
    return jax.vmap(lambda k, *a: model(*a, key=k))(keys, *xs)


@pytest.mark.parametrize("name", ALL)
def test_output_shape(name):
    model = models.build_model(name, key=jr.PRNGKey(0), **tiny_kwargs(name))
    out = jax.vmap(eqx.nn.inference_mode(model))(*_inputs(name))
    assert out.shape == (4, TINY["pred_len"], TINY["n_features"])
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize("name", ["autoformer", "fedformer", "informer"])
def test_label_len_not_half_seq_len(name):
    # FEDformer's decoder blocks must follow label_len + pred_len, not seq_len // 2 + pred_len.
    kw = tiny_kwargs(name, label_len=10)
    model = models.build_model(name, key=jr.PRNGKey(0), **kw)
    out = jax.vmap(eqx.nn.inference_mode(model))(*_inputs(name, label_len=10))
    assert out.shape == (4, TINY["pred_len"], TINY["n_features"])


def test_informer_distil_stack():
    kw = tiny_kwargs("informer", e_layers=3)
    model = models.build_model("informer", key=jr.PRNGKey(0), **kw)
    assert len(model.conv_layers) == 2
    out = jax.vmap(eqx.nn.inference_mode(model))(*_inputs("informer"))
    assert out.shape == (4, TINY["pred_len"], TINY["n_features"])


@pytest.mark.parametrize("name", ALL)
def test_gradients_reach_every_parameter(name):
    model = models.build_model(name, key=jr.PRNGKey(0), **tiny_kwargs(name, dropout=0.1))
    xs = _inputs(name)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss(p):
        return jnp.mean(_train_forward(eqx.combine(p, static), xs, jr.PRNGKey(2)) ** 2)

    grads = eqx.filter_jit(jax.grad(loss))(params)
    leaves = jax.tree_util.tree_leaves_with_path(grads)
    assert leaves
    for path, g in leaves:
        where = jax.tree_util.keystr(path)
        assert jnp.isfinite(g).all(), where
        if "decomp" in where and ".layer." in where:
            continue  # FEDformer's single-kernel decomposition gate is a softmax over one value: constant
        if name == "fedformer" and ("value_projection" in where or (
                "key_projection" in where and "cross" not in where)):
            continue  # reference behaviour: FourierBlock ignores k/v; cross-attention uses k in place of v
        assert jnp.any(g != 0), f"no gradient reaches {where}"


@pytest.mark.parametrize("name", ALL)
def test_inference_deterministic_training_stochastic(name):
    model = models.build_model(name, key=jr.PRNGKey(0), **tiny_kwargs(name, dropout=0.2))
    xs = _inputs(name)
    infer = jax.vmap(eqx.nn.inference_mode(model))
    assert jnp.array_equal(infer(*xs), infer(*xs))
    a, b = _train_forward(model, xs, jr.PRNGKey(3)), _train_forward(model, xs, jr.PRNGKey(4))
    assert not jnp.allclose(a, b)


def test_no_recompile_on_new_values():
    model = models.build_model("autoformer", key=jr.PRNGKey(0), **tiny_kwargs("autoformer"))
    f = eqx.filter_jit(lambda m, *xs: jax.vmap(m)(*xs))
    m = eqx.nn.inference_mode(model)
    xs = _inputs("autoformer")
    f(m, *xs)
    f(m, *(x + 1.0 for x in xs))
    assert f._cached._cache_size() == 1


@pytest.mark.parametrize("name,extra", [(n, {}) for n in ALL] + [("informer", {"attn": "full"})])
def test_can_overfit_small_batch(name, extra):
    """A correct model drives training loss far down on 8 fixed windows."""
    import optax

    model = models.build_model(name, key=jr.PRNGKey(0), **tiny_kwargs(name, **extra))
    xs = _inputs(name, batch=8)
    t = np.linspace(0, 6 * np.pi, TINY["pred_len"] * 8).reshape(8, TINY["pred_len"], 1)
    y = jnp.asarray(np.concatenate([np.sin(t), np.cos(t), t / t.max()], axis=-1), jnp.float32)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt = optax.adam(3e-3)
    state = opt.init(params)

    def loss(p, k):
        return jnp.mean((_train_forward(eqx.combine(p, static), xs, k) - y) ** 2)

    @eqx.filter_jit
    def step(p, s, k):
        v, g = jax.value_and_grad(loss)(p, k)
        u, s = opt.update(g, s, p)
        return eqx.apply_updates(p, u), s, v

    first = None
    for i in range(400):
        params, state, v = step(params, state, jr.PRNGKey(i))
        first = float(v) if first is None else first
    assert float(v) < 0.02 * first, f"{name}: loss {first:.4f} -> {float(v):.4f}"
