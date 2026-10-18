"""
Feature-panel construction and transformer train/forecast (JAX).

``run_forecasts`` refuses any panel that fails ``require_valid_panel``, so the
models can only ever see data that passed the schema gate.

Training is leakage-safe: windows are split chronologically with a purge gap,
the scaler is fit on training rows only, and early stopping watches the
validation split.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import jax.random as jr
import numpy as np
import pandas as pd

from models import MODEL_NAMES, build_model, mark_cols, model_kwargs, window_kind

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
from .training import Scaler, TrainConfig, TrainResult, predict, save_checkpoint, train_model
from .windows import forecast_inputs, make_windows, n_windows, rows_touched, split_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"

# Below this many validation windows, early stopping is too noisy to trust.
MIN_VAL_WINDOWS = 10
FALLBACK_EPOCHS = 30


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


def train_config(cfg: ForecastConfig) -> TrainConfig:
    return TrainConfig(epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr, weight_decay=cfg.weight_decay,
                       patience=cfg.patience, target_weight=cfg.target_weight)


def model_key(cfg: ForecastConfig, name):
    """Stable per-model init/training keys (fold_in by registry index, not hash())."""
    init_key, train_key = jr.split(jr.fold_in(jr.PRNGKey(cfg.seed), MODEL_NAMES.index(name)))
    return init_key, train_key


@dataclass
class FittedModel:
    """A trained model plus everything needed to window, predict and invert its output."""

    name: str
    model: object
    scaler: Scaler
    build_kwargs: dict
    result: TrainResult
    n_train_windows: int
    n_val_windows: int

    def windows(self, values, dates, cfg):
        """Scaled windows over the full ``values`` array, for indexing by window id."""
        return make_windows(self.scaler.transform(values), time_features(dates, mark_cols(self.name)), cfg,
                            window_kind(self.name))


def fit_model(name, values, dates, cfg: ForecastConfig, train_idx, val_idx, *, target_idx, epochs=None,
              log=print) -> FittedModel:
    """Fit the scaler on rows touched by ``train_idx`` windows, then train ``name``.

    ``val_idx`` drives early stopping. If it has fewer than ``MIN_VAL_WINDOWS``
    windows, training runs a fixed ``FALLBACK_EPOCHS`` instead.
    """
    scaler = Scaler.fit(values[:rows_touched(train_idx, cfg)])
    marks = time_features(dates, mark_cols(name))
    windows = make_windows(scaler.transform(values), marks, cfg, window_kind(name))
    train = windows.take(train_idx)
    val = windows.take(val_idx) if len(val_idx) >= MIN_VAL_WINDOWS else None
    if val is None and epochs is None:
        if len(val_idx):
            log(f"  only {len(val_idx)} validation windows (< {MIN_VAL_WINDOWS}); "
                f"training {FALLBACK_EPOCHS} epochs without early stopping")
        epochs = min(cfg.epochs, FALLBACK_EPOCHS)

    kwargs = model_kwargs(name, values.shape[1], cfg.seq_len, cfg.label_len, cfg.pred_len)
    if name == "fedformer":
        kwargs["mode_seed"] = cfg.seed
    init_key, train_key = model_key(cfg, name)
    model = build_model(name, key=init_key, **kwargs)
    result = train_model(model, train, val, train_config(cfg), target_idx=target_idx, key=train_key,
                         epochs=epochs, log=log)
    return FittedModel(name, result.model, scaler, kwargs, result, train.size, 0 if val is None else val.size)


def run_forecasts(panel, request: MarketDataRequest, cfg: ForecastConfig, log=print):
    """Train each configured model on ``panel`` and forecast ``pred_len`` days.

    Returns ``(forecasts, diagnostics, last_close)``. ``forecasts`` maps model
    name to a validated ``ModelForecast``; ``diagnostics`` maps it to training
    stats (losses, best epoch, window counts, time).
    """
    require_valid_panel(panel, request, cfg)

    values = panel.values.astype(np.float32)
    target_idx = list(panel.columns).index(TARGET)
    last_close = _last_close(panel, request)
    future_dates = pd.bdate_range(start=panel.index[-1] + pd.Timedelta(days=1), periods=cfg.pred_len)
    splits = split_indices(n_windows(len(values), cfg), cfg.pred_len, val_frac=cfg.val_frac)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    forecasts, diagnostics = {}, {}
    for name in cfg.models:
        log(f"Training {name} on {len(splits['train'])} windows (val {len(splits['val'])})...")
        fitted = fit_model(name, values, panel.index, cfg, splits["train"], splits["val"],
                           target_idx=target_idx, log=log)
        res = fitted.result
        diag = {
            "train_loss": res.history["train_loss"][-1],
            "best_val_target_mse": res.best_val_target_mse,
            "best_epoch": res.best_epoch,
            "epochs_run": len(res.history["train_loss"]),
            "n_train_windows": fitted.n_train_windows,
            "n_val_windows": fitted.n_val_windows,
            "n_params": res.n_params,
            "train_time_s": round(res.train_time_s, 2),
        }

        if cfg.refit_full and fitted.n_val_windows:
            # Use the newest data too: retrain on train+val for the epoch count early stopping picked.
            all_idx = np.arange(splits["val"][-1] + 1)
            fitted = fit_model(name, values, panel.index, cfg, all_idx, np.array([], int),
                               target_idx=target_idx, epochs=res.best_epoch, log=log)
            diag["refit_windows"] = len(all_idx)
            diag["train_time_s"] = round(diag["train_time_s"] + fitted.result.train_time_s, 2)

        marks, future_marks = time_features(panel.index, mark_cols(name)), time_features(future_dates,
                                                                                         mark_cols(name))
        inputs = forecast_inputs(fitted.scaler.transform(values), marks, future_marks, cfg, window_kind(name))
        pred = predict(fitted.model, inputs)[0]
        log_ret = fitted.scaler.inverse(pred[:, target_idx], target_idx)
        forecasts[name] = ModelForecast(
            model=name,
            dates=[str(d.date()) for d in future_dates],
            predicted_log_return=log_ret.astype(float).tolist(),
            implied_close=(last_close * np.exp(np.cumsum(log_ret))).astype(float).tolist(),
        )
        diagnostics[name] = diag
        log(f"  {name}: best epoch {diag['best_epoch']}, val target MSE {diag['best_val_target_mse']}, "
            f"{diag['train_time_s']}s")

        if cfg.save_checkpoints:
            path = save_checkpoint(
                CHECKPOINT_ROOT / request.ticker / name / stamp, name, fitted.model, fitted.build_kwargs,
                fitted.scaler,
                extra={"features": list(panel.columns), "target_idx": target_idx,
                       "forecast_config": cfg.model_dump(), "train_end": str(panel.index[-1].date()),
                       "diagnostics": diag, "history": fitted.result.history})
            log(f"  checkpoint: {path}")
    return forecasts, diagnostics, last_close


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
