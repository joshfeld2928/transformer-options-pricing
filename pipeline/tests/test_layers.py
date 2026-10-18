"""Shared building blocks and static index structures vs naive references."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from models.autoformer import AutoCorrelation
from models.fedformer import FourierBlock
from models.layers import CircularConv1d, StridedConv1d, moving_avg
from models.masks import causal_mask, get_frequency_modes, pyraformer_mask, pyramid_sizes, refer_points


def _naive_moving_avg(x, k):
    front, end = k - 1 - (k - 1) // 2, (k - 1) // 2
    xp = np.concatenate([np.repeat(x[:1], front, 0), x, np.repeat(x[-1:], end, 0)])
    return np.stack([xp[i:i + k].mean(0) for i in range(len(x))])


def test_moving_avg_odd_and_even_kernels():
    x = np.random.default_rng(0).standard_normal((40, 3)).astype(np.float32)
    for k in (24, 25, 5):
        np.testing.assert_allclose(moving_avg(jnp.asarray(x), k), _naive_moving_avg(x, k), atol=1e-5)


def test_circular_conv_matches_numpy():
    conv = CircularConv1d(2, 3, 3, key=jr.PRNGKey(0))
    x = np.random.default_rng(1).standard_normal((10, 2)).astype(np.float32)
    w, b = np.asarray(conv.weight), np.asarray(conv.bias)
    xp = np.concatenate([x[-1:], x, x[:1]])
    ref = np.stack([sum(xp[t + j] @ w[:, :, j].T for j in range(3)) + b for t in range(10)])
    np.testing.assert_allclose(conv(jnp.asarray(x)), ref, atol=1e-5)


def test_strided_conv_floors_length():
    conv = StridedConv1d(4, 4, 3, key=jr.PRNGKey(0))
    assert conv(jnp.ones((61, 4))).shape == (20, 4)


def test_autocorrelation_gather_matches_roll_loop():
    rng = np.random.default_rng(2)
    q, k, v = (rng.standard_normal((16, 2, 4)).astype(np.float32) for _ in range(3))
    out = np.asarray(AutoCorrelation(factor=1)(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)))

    corr = np.fft.irfft(np.fft.rfft(q.transpose(1, 2, 0)) * np.conj(np.fft.rfft(k.transpose(1, 2, 0))), n=16)
    mean = corr.mean((0, 1))
    top = int(np.log(16))
    delays = np.argsort(-mean)[:top]
    w = np.exp(mean[delays] - mean[delays].max())
    w /= w.sum()
    vals = v.transpose(1, 2, 0)
    ref = sum(wi * np.roll(vals, -d, axis=-1) for wi, d in zip(w, delays)).transpose(2, 0, 1)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_fourier_block_compact_bins():
    block = FourierBlock(4, 2, (1, 3), key=jr.PRNGKey(0))
    q = np.random.default_rng(3).standard_normal((12, 2, 2)).astype(np.float32)
    out = np.asarray(block(jnp.asarray(q), None, None))
    x_ft = np.fft.rfft(q.transpose(1, 2, 0))
    W = np.asarray(block.w_re) + 1j * np.asarray(block.w_im)
    out_ft = np.zeros((2, 2, 7), complex)
    for wi, i in enumerate((1, 3)):
        out_ft[..., wi] = np.einsum("hi,hio->ho", x_ft[..., i], W[..., wi])
    np.testing.assert_allclose(out, np.fft.irfft(out_ft, n=12), atol=1e-5)


def test_frequency_modes_seeded():
    a = get_frequency_modes(60, 16, "random", np.random.default_rng(5))
    b = get_frequency_modes(60, 16, "random", np.random.default_rng(5))
    assert a == b and len(a) == 16 and list(a) == sorted(a) and max(a) < 30
    assert get_frequency_modes(60, 64, "low", None) == tuple(range(30))


def test_causal_mask():
    m = causal_mask(4)
    assert m[0, 1] and not m[1, 0] and not m.diagonal().any()


def test_pyramid_sizes_and_mask():
    assert pyramid_sizes(61, (3, 3, 3)) == (61, 20, 6, 2)
    mask = pyraformer_mask(61, (3, 3, 3), 3)
    assert mask.shape == (89, 89)
    assert (mask == mask.T).all()  # parent/child links are symmetric
    assert not mask[0, 0] and not mask[0, 1] and mask[0, 2]  # intra-scale window of 3
    assert not mask[0, 61] and not mask[61, 0]  # first leaf <-> first parent
    assert not mask[60, 80] and not mask[80, 60]  # last parent absorbs the tail leaf


def test_refer_points_chain():
    idx = refer_points(61, (3, 3, 3))
    assert idx.shape == (61, 4)
    np.testing.assert_array_equal(idx[0], [0, 61, 81, 87])
    np.testing.assert_array_equal(idx[-1], [60, 80, 86, 88])
