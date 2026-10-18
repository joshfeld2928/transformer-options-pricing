"""
Simple forecasters the transformers have to beat. All numpy, all in log-return units.

Each baseline has ``fit(values, target_idx, train_idx, val_idx, cfg)`` and
``predict(values, idx, cfg) -> [len(idx), pred_len]``, where window ``i`` forecasts
the ``pred_len`` rows after ``values[i : i + seq_len]``.
"""

import numpy as np

from utils.windows import rows_touched


class RandomWalk:
    """Zero expected return: the efficient-markets null."""

    name = "random_walk"

    def fit(self, values, target_idx, train_idx, val_idx, cfg):
        return self

    def predict(self, values, idx, cfg):
        return np.zeros((len(idx), cfg.pred_len))


class HistoricalMean:
    """Mean training-period return at every step."""

    name = "hist_mean"

    def fit(self, values, target_idx, train_idx, val_idx, cfg):
        self.mu = float(values[:rows_touched(train_idx, cfg), target_idx].mean())
        return self

    def predict(self, values, idx, cfg):
        return np.full((len(idx), cfg.pred_len), self.mu)


class AR1:
    """r_t = a + b r_{t-1}, OLS on training rows, iterated over the horizon."""

    name = "ar1"

    def fit(self, values, target_idx, train_idx, val_idx, cfg):
        r = values[:rows_touched(train_idx, cfg), target_idx]
        X = np.column_stack([np.ones(len(r) - 1), r[:-1]])
        self.a, self.b = np.linalg.lstsq(X, r[1:], rcond=None)[0]
        self.target_idx = target_idx
        return self

    def predict(self, values, idx, cfg):
        last = values[np.asarray(idx) + cfg.seq_len - 1, self.target_idx]
        out = np.empty((len(idx), cfg.pred_len))
        for h in range(cfg.pred_len):
            last = self.a + self.b * last
            out[:, h] = last
        return out


class RidgeLags:
    """Direct multi-horizon ridge on the last ``n_lags`` returns plus the latest value of every feature.

    The penalty is picked on the validation windows from a log-spaced grid.
    """

    name = "ridge_lags"
    GRID = np.logspace(-2, 4, 13)

    def __init__(self, n_lags=20):
        self.n_lags = n_lags

    def _design(self, values, idx, cfg):
        end = np.asarray(idx) + cfg.seq_len  # exclusive end of each window's input
        lags = np.stack([values[end - 1 - k, self.target_idx] for k in range(min(self.n_lags, cfg.seq_len))], 1)
        X = np.column_stack([lags, values[end - 1]])
        return (X - self.mu) / self.sd if hasattr(self, "mu") else X

    def _targets(self, values, idx, cfg):
        start = np.asarray(idx) + cfg.seq_len
        return np.stack([values[start + h, self.target_idx] for h in range(cfg.pred_len)], 1)

    @staticmethod
    def _solve(X, Y, lam):
        Xb = np.column_stack([np.ones(len(X)), X])
        reg = lam * np.eye(Xb.shape[1])
        reg[0, 0] = 0.0  # don't shrink the intercept
        return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)

    def fit(self, values, target_idx, train_idx, val_idx, cfg):
        self.target_idx = target_idx
        X = self._design(values, train_idx, cfg)
        self.mu, self.sd = X.mean(0), np.where(X.std(0) == 0, 1.0, X.std(0))
        X, Y = (X - self.mu) / self.sd, self._targets(values, train_idx, cfg)
        if len(val_idx):
            Xv, Yv = self._design(values, val_idx, cfg), self._targets(values, val_idx, cfg)
            scores = [np.mean((np.column_stack([np.ones(len(Xv)), Xv]) @ self._solve(X, Y, lam) - Yv) ** 2)
                      for lam in self.GRID]
            self.lam = float(self.GRID[int(np.argmin(scores))])
        else:
            self.lam = 10.0
        self.W = self._solve(X, Y, self.lam)
        return self

    def predict(self, values, idx, cfg):
        X = self._design(values, idx, cfg)
        return np.column_stack([np.ones(len(X)), X]) @ self.W


BASELINES = (RandomWalk, HistoricalMean, AR1, RidgeLags)
