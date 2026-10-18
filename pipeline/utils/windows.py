"""
Sliding-window construction and leakage-safe chronological splits.

Window ``i`` reads rows ``[i, i + seq_len)`` as encoder input and predicts rows
``[i + seq_len, i + seq_len + pred_len)``.
"""

from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Windows(NamedTuple):
    enc_x: np.ndarray     # [N, L_enc, C]
    enc_mark: np.ndarray  # [N, L_enc, M]
    dec_x: np.ndarray     # [N, L_dec, C]
    dec_mark: np.ndarray  # [N, L_dec, M]
    target: np.ndarray | None  # [N, pred_len, C]

    @property
    def size(self):
        # Not __len__: NamedTuple machinery (e.g. _replace) relies on len() == number of fields.
        return len(self.enc_x)

    def take(self, idx):
        return Windows(*(None if a is None else a[idx] for a in self))

    def inputs(self):
        return self[:4]


def _slide(a, length):
    """[T, C] -> [T - length + 1, length, C]."""
    return np.ascontiguousarray(sliding_window_view(a, length, axis=0).transpose(0, 2, 1))


def n_windows(n_rows, cfg):
    return n_rows - cfg.seq_len - cfg.pred_len + 1


def make_windows(values, marks, cfg, kind):
    """All training windows from scaled ``values`` [T, C] and time ``marks`` [T, M].

    ``kind="standard"``: encoder-decoder inputs where the decoder sees the last
    ``label_len`` rows plus ``pred_len`` zero placeholders.
    ``kind="pyraformer"``: encoder-only input with a +1 zero predict token.
    """
    values, marks = values.astype(np.float32), marks.astype(np.float32)
    seq, label, pred = cfg.seq_len, cfg.label_len, cfg.pred_len
    n = n_windows(len(values), cfg)
    if n <= 0:
        raise ValueError(f"{len(values)} rows is too few for seq_len={seq}, pred_len={pred}")
    c = values.shape[1]
    span_v = _slide(values, seq + pred)[:n]  # [N, seq+pred, C]
    span_m = _slide(marks, seq + pred)[:n]
    target = span_v[:, seq:]

    if kind == "pyraformer":
        enc_x = np.concatenate([span_v[:, :seq], np.zeros((n, 1, c), np.float32)], axis=1)
        enc_mark = span_m[:, :seq + 1]
        dec_x = np.zeros((n, pred, c), np.float32)
        dec_mark = np.zeros((n, pred, marks.shape[1]), np.float32)
    else:
        enc_x, enc_mark = span_v[:, :seq], span_m[:, :seq]
        dec_x = np.concatenate([span_v[:, seq - label:seq], np.zeros((n, pred, c), np.float32)], axis=1)
        dec_mark = span_m[:, seq - label:]
    return Windows(enc_x, enc_mark, dec_x, dec_mark, target)


def forecast_inputs(values, marks, future_marks, cfg, kind):
    """A single window (N=1) whose prediction covers the ``pred_len`` rows after ``values``."""
    seq, label, pred = cfg.seq_len, cfg.label_len, cfg.pred_len
    values, marks, future_marks = (a.astype(np.float32) for a in (values, marks, future_marks))
    c = values.shape[1]
    if kind == "pyraformer":
        enc_x = np.concatenate([values[-seq:], np.zeros((1, c), np.float32)])
        enc_mark = np.concatenate([marks[-seq:], future_marks[:1]])
        dec_x = np.zeros((pred, c), np.float32)
        dec_mark = np.zeros((pred, marks.shape[1]), np.float32)
    else:
        enc_x, enc_mark = values[-seq:], marks[-seq:]
        dec_x = np.concatenate([values[-label:], np.zeros((pred, c), np.float32)])
        dec_mark = np.concatenate([marks[-label:], future_marks])
    return Windows(*(a[None] for a in (enc_x, enc_mark, dec_x, dec_mark)), None)


def split_indices(n, pred_len, val_frac=0.15, test_frac=0.0):
    """Chronological train/val/test window indices.

    ``pred_len`` windows are dropped at each boundary (a purge gap), so no
    training target row is also a validation or test target row.
    """
    gap = pred_len
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_val - n_test - gap * ((n_val > 0) + (n_test > 0))
    if n_train < 1:
        raise ValueError(f"{n} windows is too few for val_frac={val_frac}, test_frac={test_frac}")
    train = np.arange(n_train)
    val_start = n_train + (gap if n_val else 0)
    val = np.arange(val_start, val_start + n_val)
    test_start = val_start + n_val + (gap if n_test else 0)
    test = np.arange(test_start, test_start + n_test)
    return {"train": train, "val": val, "test": test}


def rows_touched(idx, cfg):
    """Exclusive end row of everything windows ``idx`` read or predict."""
    return int(idx[-1]) + cfg.seq_len + cfg.pred_len
