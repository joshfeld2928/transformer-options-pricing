"""Metrics, baselines and fold construction for the evaluation package."""

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.baselines import AR1, HistoricalMean, RandomWalk, RidgeLags
from evaluation.evaluate import walkforward_folds
from evaluation.metrics import diebold_mariano, forecast_metrics

CFG = SimpleNamespace(seq_len=20, label_len=10, pred_len=5, val_frac=0.15)


def test_forecast_metrics_basics():
    y = np.array([[0.01, -0.02], [0.03, 0.01]])
    m, ph = forecast_metrics(y, y.copy(), y_ref=np.zeros_like(y))
    assert m["rmse"] == 0 and m["oos_r2_vs_zero"] == 1 and m["hit_rate_step1"] == 1
    m, _ = forecast_metrics(y, np.zeros_like(y))
    assert m["oos_r2_vs_zero"] == pytest.approx(0) and np.isnan(m["hit_rate_step1"])
    assert ph["rmse"].shape == (2,)


def test_price_metrics_use_compounded_returns():
    y = np.full((1, 3), np.log(1.1))
    close = np.array([100.0])
    actual = 100 * 1.1 ** np.arange(1, 4)[None]
    m, _ = forecast_metrics(y, y, origin_close=close, actual_close=actual)
    assert m["price_mape"] == pytest.approx(0, abs=1e-12)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500)
    good = y + 0.3 * rng.standard_normal(500)
    stat, p = diebold_mariano(good - y, -y)
    assert stat < 0 and p < 1e-6
    stat, p = diebold_mariano(-y, -y + 1e-12 * rng.standard_normal(500), h=5)
    assert np.isnan(stat) or p > 0.05


def test_baselines_shapes_and_ar1_recovers_coefficient():
    rng = np.random.default_rng(1)
    n = 2000
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.5 * r[t - 1] + 0.01 * rng.standard_normal()
    values = np.column_stack([r, rng.standard_normal(n)]).astype(np.float64)
    idx = np.arange(n - CFG.seq_len - CFG.pred_len + 1)
    train, val, test = idx[:1500], idx[1510:1700], idx[1710:]
    for B in (RandomWalk, HistoricalMean, AR1, RidgeLags):
        pred = B().fit(values, 0, train, val, CFG).predict(values, test, CFG)
        assert pred.shape == (len(test), CFG.pred_len)
    ar = AR1().fit(values, 0, train, val, CFG)
    assert ar.b == pytest.approx(0.5, abs=0.05)


def test_walkforward_folds_are_ordered_and_purged():
    folds = walkforward_folds(300, CFG, test_frac=0.2, k=4)
    assert len(folds) == 4
    tests = np.concatenate([f[2] for f in folds])
    assert tests[0] == 240 and tests[-1] == 299 and (np.diff(tests) == 1).all()
    for train, val, test in folds:
        assert val[-1] + CFG.pred_len <= test[0] - 1
        assert train[-1] + CFG.pred_len <= val[0] - 1
