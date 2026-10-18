"""
Static index structures, built once in numpy and cached.

These become XLA constants at trace time. They are never stored as module
leaves, so the optimiser can't touch them.
"""

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def causal_mask(length):
    """[L, L] bool, True == masked out (strictly upper triangle)."""
    return np.triu(np.ones((length, length), dtype=bool), k=1)


@lru_cache(maxsize=None)
def sinusoid_table(length, d_model):
    """Fixed sinusoidal positional encoding, [length, d_model] float32."""
    position = np.arange(length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
    pe = np.zeros((length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def get_frequency_modes(seq_len, modes, method, rng):
    """Sorted tuple of ``modes`` rfft bin indices in [0, seq_len // 2)."""
    modes = min(modes, seq_len // 2)
    if method == "random":
        index = rng.permutation(seq_len // 2)[:modes]
    elif method == "low":
        index = np.arange(modes)
    else:
        raise ValueError(f"unknown mode_select_method {method!r}")
    return tuple(int(i) for i in sorted(index))


@lru_cache(maxsize=None)
def pyramid_sizes(input_size, window_size):
    sizes = [input_size]
    for w in window_size:
        sizes.append(sizes[-1] // w)
    return tuple(sizes)


@lru_cache(maxsize=None)
def pyraformer_mask(input_size, window_size, inner_size):
    """Pyramidal attention mask, [N_nodes, N_nodes] bool, True == masked out."""
    all_size = pyramid_sizes(input_size, window_size)
    n = sum(all_size)
    allowed = np.zeros((n, n), dtype=bool)

    # Intra-scale: A adjacent nodes (incl. self), clipped at level boundaries.
    half = inner_size // 2
    for level, size in enumerate(all_size):
        start = sum(all_size[:level])
        for i in range(start, start + size):
            allowed[i, max(i - half, start):min(i + half + 1, start + size)] = True

    # Inter-scale: children, symmetrised so each child also sees its parent.
    for level in range(1, len(all_size)):
        start = sum(all_size[:level])
        child_base = start - all_size[level - 1]
        w = window_size[level - 1]
        for i in range(start, start + all_size[level]):
            left = child_base + (i - start) * w
            # The last node of a level absorbs the tail.
            right = start if i == start + all_size[level] - 1 else child_base + (i - start + 1) * w
            allowed[i, left:right] = True
            allowed[left:right, i] = True

    return ~allowed


@lru_cache(maxsize=None)
def refer_points(input_size, window_size):
    """Flat node index of every finest-scale position at every scale, [input_size, S] int32."""
    all_size = pyramid_sizes(input_size, window_size)
    idx = np.zeros((input_size, len(all_size)), dtype=np.int32)
    for i in range(input_size):
        idx[i, 0] = former = i
        for j in range(1, len(all_size)):
            start = sum(all_size[:j])
            inner = former - (start - all_size[j - 1])
            former = start + min(inner // window_size[j - 1], all_size[j] - 1)
            idx[i, j] = former
    return idx
