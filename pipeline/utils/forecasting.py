"""
Feature-panel construction and transformer train/forecast, parameterised
from ``smoke_test_logreturn.py``.

``run_forecasts`` refuses any panel that fails ``require_valid_panel``, so the
models can only ever see data that passed the schema gate.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from .company_data import _get_underlying_history
from .macro_indicators import get_macro_data
from .schemas import (
    MACRO_FEATURES,
    TARGET,
    ForecastConfig,
    MarketDataRequest,
    ModelForecast,
    require_valid_panel,
)

# Small dims so CPU training finishes in seconds.
SMALL_STANDARD_KWARGS = dict(d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128, dropout=0.05, freq="d")
SMALL_PYRAFORMER_KWARGS = dict(d_model=64, d_inner_hid=64, d_bottleneck=32, d_k=32, d_v=32,
                               n_head=4, n_layer=2, window_size=[3, 3, 3])


def build_feature_panel(request: MarketDataRequest, end_date=None) -> pd.DataFrame:
    """Pull, clean and align underlying + macro data onto the trading calendar."""
    end_date = end_date or datetime.today()
    window_start = end_date - timedelta(days=30 * request.history_months)
    # Lead-in so the 21-day realized-vol window isn't NaN at window_start.
    pull_start = window_start - timedelta(days=45)
    start_str, end_str = pull_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    panel = _get_underlying_history(request.ticker, start_str, end_str)
    if set(request.features) & set(MACRO_FEATURES):
        macro = get_macro_data(start_date=start_str, end_date=end_str)
        panel = panel.join(macro, how="left")

    panel = panel.ffill().dropna(subset=request.features)
    panel = panel.loc[panel.index >= window_start]
    return panel[request.features]


def time_features(dates, n_cols=3):
    """Normalized (month, day, weekday[, hour=0]) features; Pyraformer needs 4 columns."""
    dates = pd.DatetimeIndex(dates)
    cols = [dates.month.values / 12.0 - 0.5, dates.day.values / 31.0 - 0.5, dates.weekday.values / 6.0 - 0.5]
    if n_cols == 4:
        cols.append(np.zeros(len(dates)))
    return np.stack(cols, axis=-1).astype(np.float32)


def _make_windows(values, marks, cfg):
    n_features = values.shape[1]
    enc_x, enc_mark, dec_x, dec_mark, target = [], [], [], [], []
    for i in range(len(values) - cfg.seq_len - cfg.pred_len + 1):
        enc_x.append(values[i:i + cfg.seq_len])
        enc_mark.append(marks[i:i + cfg.seq_len])
        dec_hist = values[i + cfg.seq_len - cfg.label_len:i + cfg.seq_len]
        dec_x.append(np.concatenate([dec_hist, np.zeros((cfg.pred_len, n_features), np.float32)]))
        dec_mark.append(marks[i + cfg.seq_len - cfg.label_len:i + cfg.seq_len + cfg.pred_len])
        target.append(values[i + cfg.seq_len:i + cfg.seq_len + cfg.pred_len])
    return [torch.from_numpy(np.stack(a)) for a in (enc_x, enc_mark, dec_x, dec_mark, target)]


def _make_windows_pyraformer(values, marks4, cfg):
    """Encoder-only windows with the FC head's +1 zero predict token."""
    n_features = values.shape[1]
    enc_x, enc_mark, target = [], [], []
    for i in range(len(values) - cfg.seq_len - cfg.pred_len + 1):
        enc_x.append(np.concatenate([values[i:i + cfg.seq_len], np.zeros((1, n_features), np.float32)]))
        enc_mark.append(marks4[i:i + cfg.seq_len + 1])
        target.append(values[i + cfg.seq_len:i + cfg.seq_len + cfg.pred_len])
    enc_x, enc_mark, target = (torch.from_numpy(np.stack(a)) for a in (enc_x, enc_mark, target))
    dec_x = torch.zeros(len(enc_x), cfg.pred_len, n_features)
    dec_mark = torch.zeros(len(enc_x), cfg.pred_len, marks4.shape[1])
    return enc_x, enc_mark, dec_x, dec_mark, target


def _forecast_inputs(values, marks, future_marks, cfg):
    n_features = values.shape[1]
    enc_x = values[-cfg.seq_len:]
    enc_mark = marks[-cfg.seq_len:]
    dec_x = np.concatenate([values[-cfg.label_len:], np.zeros((cfg.pred_len, n_features), np.float32)])
    dec_mark = np.concatenate([marks[-cfg.label_len:], future_marks])
    return [torch.from_numpy(a[None, ...]) for a in (enc_x, enc_mark, dec_x, dec_mark)]


def _forecast_inputs_pyraformer(values, marks4, future_marks4, cfg):
    n_features = values.shape[1]
    enc_x = np.concatenate([values[-cfg.seq_len:], np.zeros((1, n_features), np.float32)])
    enc_mark = np.concatenate([marks4[-cfg.seq_len:], future_marks4[:1]])
    return [torch.from_numpy(enc_x[None, ...]), torch.from_numpy(enc_mark[None, ...]),
            torch.zeros(1, cfg.pred_len, n_features), torch.zeros(1, cfg.pred_len, marks4.shape[1])]


def _build_model(name, n_features, cfg):
    dims = dict(enc_in=n_features, dec_in=n_features, c_out=n_features,
                seq_len=cfg.seq_len, label_len=cfg.label_len)
    if name == "autoformer":
        from inference.autoformer_inference import build_autoformer
        return build_autoformer(**dims, pred_len=cfg.pred_len, **SMALL_STANDARD_KWARGS)
    if name == "fedformer":
        from inference.fedformer_inference import build_fedformer
        return build_fedformer(**dims, pred_len=cfg.pred_len, modes=16, **SMALL_STANDARD_KWARGS)
    if name == "informer":
        from inference.informer_inference import build_informer
        return build_informer(**dims, out_len=cfg.pred_len, **SMALL_STANDARD_KWARGS)
    if name == "pyraformer":
        from inference.pyraformer_inference import build_pyraformer
        return build_pyraformer(enc_in=n_features, c_out=n_features, input_size=cfg.seq_len,
                                predict_step=cfg.pred_len, device="cpu", **SMALL_PYRAFORMER_KWARGS)
    raise ValueError(f"unknown model {name}")


def _train(model, windows, cfg):
    enc_x, enc_mark, dec_x, dec_mark, target = windows
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    loss = None
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(enc_x, enc_mark, dec_x, dec_mark), target)
        loss.backward()
        optimizer.step()
    return float(loss.item())


def run_forecasts(panel, request: MarketDataRequest, cfg: ForecastConfig, log=print):
    """Train each configured model on ``panel`` and forecast ``pred_len`` days.

    Returns ``(forecasts, train_losses, last_close)`` where ``forecasts`` maps
    model name to a validated ``ModelForecast``.
    """
    require_valid_panel(panel, request, cfg)
    torch.manual_seed(cfg.seed)

    values = panel.values.astype(np.float32)
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std == 0] = 1.0
    scaled = ((values - mean) / std).astype(np.float32)
    n_features = scaled.shape[1]
    target_idx = list(panel.columns).index(TARGET)

    last_date = panel.index[-1]
    last_close = _last_close(panel, request)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=cfg.pred_len)

    marks3, marks4 = time_features(panel.index), time_features(panel.index, n_cols=4)
    fut3, fut4 = time_features(future_dates), time_features(future_dates, n_cols=4)

    forecasts, losses = {}, {}
    for name in cfg.models:
        if name == "pyraformer":
            windows = _make_windows_pyraformer(scaled, marks4, cfg)
            inputs = _forecast_inputs_pyraformer(scaled, marks4, fut4, cfg)
        else:
            windows = _make_windows(scaled, marks3, cfg)
            inputs = _forecast_inputs(scaled, marks3, fut3, cfg)

        log(f"Training {name} on {len(windows[0])} windows...")
        model = _build_model(name, n_features, cfg)
        losses[name] = _train(model, windows, cfg)

        model.eval()
        with torch.no_grad():
            pred = model(*inputs).squeeze(0).numpy()
        log_ret = pred[:, target_idx] * std[target_idx] + mean[target_idx]
        forecasts[name] = ModelForecast(
            model=name,
            dates=[str(d.date()) for d in future_dates],
            predicted_log_return=log_ret.astype(float).tolist(),
            implied_close=(last_close * np.exp(np.cumsum(log_ret))).astype(float).tolist(),
        )
    return forecasts, losses, last_close


def _last_close(panel, request):
    if "underlying_close" in panel.columns:
        return float(panel["underlying_close"].iloc[-1])
    # Target-only panels still need a price anchor.
    hist = _get_underlying_history(request.ticker, str((panel.index[-1] - pd.Timedelta(days=10)).date()),
                                   str((panel.index[-1] + pd.Timedelta(days=1)).date()))
    return float(hist["underlying_close"].iloc[-1])


def forecasts_frame(forecasts: dict) -> pd.DataFrame:
    """Long-format frame of every model's forecast."""
    return pd.concat(
        [pd.DataFrame(f.model_dump()) for f in forecasts.values()], ignore_index=True
    )
