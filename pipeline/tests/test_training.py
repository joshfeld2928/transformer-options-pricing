"""Windows, splits, scaling, training loop, checkpoints and the end-to-end forecast step."""

from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from conftest import tiny_kwargs
from synthetic import synthetic_panel

import models
from utils.training import Scaler, TrainConfig, load_checkpoint, predict, save_checkpoint, train_model
from utils.windows import forecast_inputs, make_windows, n_windows, rows_touched, split_indices

CFG = SimpleNamespace(seq_len=8, label_len=4, pred_len=3)


def test_windows_align_with_rows():
    values = np.arange(30, dtype=np.float32)[:, None].repeat(2, 1)
    marks = np.zeros((30, 3), np.float32)
    w = make_windows(values, marks, CFG, "standard")
    assert w.size == n_windows(30, CFG) == 20
    i = 5
    np.testing.assert_array_equal(w.enc_x[i, :, 0], np.arange(5, 13))
    np.testing.assert_array_equal(w.target[i, :, 0], np.arange(13, 16))
    np.testing.assert_array_equal(w.dec_x[i, :4, 0], np.arange(9, 13))
    assert not w.dec_x[i, 4:].any()

    p = make_windows(values, np.zeros((30, 4), np.float32), CFG, "pyraformer")
    assert p.enc_x.shape == (20, 9, 2) and not p.enc_x[:, -1].any()


def test_forecast_inputs_use_last_rows():
    values = np.arange(30, dtype=np.float32)[:, None]
    w = forecast_inputs(values, np.zeros((30, 3)), np.ones((3, 3)), CFG, "standard")
    np.testing.assert_array_equal(w.enc_x[0, :, 0], np.arange(22, 30))
    assert w.dec_mark[0, -3:].all()


def test_split_purge_gap_prevents_target_overlap():
    s = split_indices(200, CFG.pred_len, val_frac=0.15, test_frac=0.15)
    def target_rows(idx):
        return {r for i in idx for r in range(i + CFG.seq_len, i + CFG.seq_len + CFG.pred_len)}
    assert not target_rows(s["train"]) & target_rows(s["val"])
    assert not target_rows(s["val"]) & target_rows(s["test"])
    assert s["test"][-1] == 199 and len(s["val"]) == 30


def test_scaler_fit_on_train_rows_only():
    values = np.r_[np.zeros(50), np.full(50, 100.0)][:, None].astype(np.float32)
    train = split_indices(n_windows(100, CFG), CFG.pred_len, val_frac=0.4)["train"]
    end = rows_touched(train, CFG)
    assert end < 100
    assert Scaler.fit(values[:end]).mean[0] < 100 * (end - 50) / end + 1e-3


def test_scaler_round_trip():
    v = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float32) * 5 + 2
    s = Scaler.fit(v)
    np.testing.assert_allclose(s.inverse(s.transform(v)), v, atol=1e-5)
    assert Scaler.from_dict(s.to_dict()).std.tolist() == s.std.tolist()


def _tiny_data(name, n=60):
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.standard_normal((n + 40, 3)), 0).astype(np.float32) / 10
    cfg = SimpleNamespace(seq_len=32, label_len=16, pred_len=4)
    marks = np.zeros((len(values), models.mark_cols(name)), np.float32)
    return make_windows(values, marks, cfg, models.window_kind(name))


def test_train_model_early_stopping_restores_best():
    name = "autoformer"
    w = _tiny_data(name)
    model = models.build_model(name, key=jr.PRNGKey(0), **tiny_kwargs(name))
    # Validation targets are noise unrelated to inputs: val loss bottoms early, then rises.
    val = w.take(np.arange(40, 60))._replace(target=np.random.default_rng(9).standard_normal((20, 4, 3)).astype(
        np.float32) * 3)
    res = train_model(model, w.take(np.arange(40)), val, TrainConfig(epochs=60, patience=5, lr=3e-3),
                      target_idx=0, key=jr.PRNGKey(1), log=lambda *_: None)
    assert res.stopped_early and len(res.history["train_loss"]) < 60
    best = res.history["val_target_mse"][res.best_epoch - 1]
    assert best == pytest.approx(min(res.history["val_target_mse"]))
    pred = predict(res.model, val)
    assert np.mean((pred[..., 0] - val.target[..., 0]) ** 2) == pytest.approx(best, rel=1e-4)


def test_checkpoint_round_trip_is_exact(tmp_path):
    for name in models.MODEL_NAMES:
        kw = tiny_kwargs(name, **({"mode_seed": 3} if name == "fedformer" else {}))
        model = models.build_model(name, key=jr.PRNGKey(7), **kw)
        w = _tiny_data(name).take(np.arange(5))
        scaler = Scaler(np.zeros(3, np.float32), np.ones(3, np.float32))
        save_checkpoint(tmp_path / name, name, model, kw, scaler, extra={"features": ["a", "b", "c"]})
        loaded, s2, meta = load_checkpoint(tmp_path / name)
        assert jnp.array_equal(predict(model, w), predict(loaded, w)), name
        assert meta["features"] == ["a", "b", "c"]


def test_run_forecasts_end_to_end():
    from utils.forecasting import run_forecasts
    from utils.schemas import ForecastConfig, MarketDataRequest

    panel = synthetic_panel(220)
    req = MarketDataRequest(ticker="TEST", features=list(panel.columns))
    cfg = ForecastConfig(models=["fedformer", "pyraformer"], epochs=4, pred_len=5)
    forecasts, diag, last_close = run_forecasts(panel, req, cfg, log=lambda *_: None)
    assert set(forecasts) == {"fedformer", "pyraformer"}
    for name, f in forecasts.items():
        assert len(f.predicted_log_return) == 5
        assert diag[name]["n_val_windows"] >= 10 and diag[name]["best_epoch"] >= 1
        assert f.implied_close[0] == pytest.approx(last_close * np.exp(f.predicted_log_return[0]))
