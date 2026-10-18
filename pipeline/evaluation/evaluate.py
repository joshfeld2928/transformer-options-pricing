"""
Ad-hoc evaluation of the transformer forecasters against simple baselines.

Usage (from ``pipeline/``):
    python -m evaluation --ticker AAPL --history-months 24
    python -m evaluation --ticker AAPL --mode walkforward --folds 4
    python -m evaluation --panel "../valuation runs/<run>/panel.csv" --models informer fedformer
    python -m evaluation --panel panel.csv --checkpoint ../checkpoints/AAPL/informer

Writes ``evaluation runs/<timestamp>/``: config.json, metrics.json, metrics.csv,
per_horizon.csv, predictions.csv, plots/*.png and report.md.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MODEL_NAMES, mark_cols, window_kind  # noqa: E402
from utils.forecasting import CHECKPOINT_ROOT, fit_model, time_features  # noqa: E402
from utils.schemas import (  # noqa: E402
    DEFAULT_FEATURES,
    TARGET,
    ForecastConfig,
    MarketDataRequest,
    validate_feature_panel,
)
from utils.training import count_params, load_checkpoint, predict, save_checkpoint  # noqa: E402
from utils.windows import make_windows, n_windows, split_indices  # noqa: E402

from . import plots  # noqa: E402
from .baselines import BASELINES  # noqa: E402
from .metrics import forecast_metrics  # noqa: E402
from .report import write_outputs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "evaluation runs"


@dataclass
class SeriesResult:
    name: str
    kind: str  # "model" | "baseline"
    preds: list = field(default_factory=list)  # per fold, [N_fold, H] log returns
    histories: list = field(default_factory=list)  # per fold, (history, best_epoch)
    train_time_s: float = 0.0
    n_params: int | None = None
    latency_ms: dict = field(default_factory=dict)

    @property
    def yhat(self):
        return np.concatenate(self.preds)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_panel(args, log=print):
    """Panel from ``--panel``, or pulled live and cached per ticker/window/day."""
    if args.panel:
        panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)
        panel.index.name = None
        return panel, Path(args.panel).stem
    features = args.features or list(DEFAULT_FEATURES)
    request = MarketDataRequest(ticker=args.ticker, history_months=args.history_months, features=features)
    cache = EVAL_ROOT / "_cache" / f"{request.ticker}_{request.history_months}m_{datetime.today():%Y%m%d}.csv"
    if cache.exists() and not args.refresh:
        log(f"Using cached panel {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True), request.ticker

    from utils.forecasting import build_feature_panel

    log(f"Pulling {request.ticker} ({request.history_months} months, {len(features)} features)...")
    panel = build_feature_panel(request)
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache)
    return panel, request.ticker


def _check_panel(panel, cfg):
    request = MarketDataRequest(ticker="EVAL", features=list(panel.columns))
    report = validate_feature_panel(panel, request, cfg)
    if not report.ok:
        raise SystemExit("panel failed validation:\n- " + "\n- ".join(report.errors))
    return report


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def holdout_folds(n, cfg, test_frac):
    s = split_indices(n, cfg.pred_len, val_frac=cfg.val_frac, test_frac=test_frac)
    return [(s["train"], s["val"], s["test"])]


def walkforward_folds(n, cfg, test_frac, k):
    """Expanding window: the last ``test_frac`` of windows is cut into ``k`` sequential test blocks.

    Each fold trains on everything before its block (minus purge gaps),
    holding out the newest ``val_frac`` of that for early stopping.
    """
    gap = cfg.pred_len
    n_test = int(round(test_frac * n))
    block = n_test // k
    if block < 1:
        raise SystemExit(f"{n_test} test windows can't be cut into {k} folds")
    folds = []
    for i in range(k):
        start = n - n_test + i * block
        stop = n if i == k - 1 else start + block
        pre = start - gap
        n_val = int(round(cfg.val_frac * pre))
        n_train = pre - n_val - (gap if n_val else 0)
        if n_train < 1:
            raise SystemExit("not enough history for walk-forward; raise --history-months or lower --folds")
        folds.append((np.arange(n_train), np.arange(pre - n_val, pre), np.arange(start, stop)))
    return folds


def _latency(model, windows, repeats=20):
    """Median ms per predict() call for a single window and for the whole batch (after warm-up)."""
    out = {}
    for label, w in (("batch1", windows.take(slice(0, 1))), (f"batch{windows.size}", windows)):
        predict(model, w)
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            predict(model, w)
            times.append((time.perf_counter() - t) * 1000)
        out[label] = float(np.median(times))
    return out


def run_folds(panel, cfg, folds, model_names, *, save_ckpt=False, ticker="EVAL", log=print):
    values = panel.values.astype(np.float32)
    dates = panel.index
    target_idx = list(panel.columns).index(TARGET)
    results = {name: SeriesResult(name, "model") for name in model_names}
    results.update({b.name: SeriesResult(b.name, "baseline") for b in BASELINES})

    for f, (train, val, test) in enumerate(folds, 1):
        log(f"Fold {f}/{len(folds)}: train {len(train)}, val {len(val)}, test {len(test)} windows "
            f"(test targets {dates[test[0] + cfg.seq_len].date()} .. "
            f"{dates[test[-1] + cfg.seq_len + cfg.pred_len - 1].date()})")
        for b in BASELINES:
            results[b.name].preds.append(b().fit(values, target_idx, train, val, cfg).predict(values, test, cfg))
        for name in model_names:
            fitted = fit_model(name, values, dates, cfg, train, val, target_idx=target_idx, log=log)
            test_w = fitted.windows(values, dates, cfg).take(test)
            pred = predict(fitted.model, test_w)[..., target_idx]
            r = results[name]
            r.preds.append(fitted.scaler.inverse(pred, target_idx))
            r.histories.append((fitted.result.history, fitted.result.best_epoch))
            r.train_time_s += fitted.result.train_time_s
            r.n_params = fitted.result.n_params
            log(f"  {name}: best epoch {fitted.result.best_epoch}, {fitted.result.train_time_s:.1f}s")
            if f == len(folds):
                r.latency_ms = _latency(fitted.model, test_w)
                if save_ckpt:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = save_checkpoint(
                        CHECKPOINT_ROOT / ticker / name / stamp, name, fitted.model, fitted.build_kwargs,
                        fitted.scaler,
                        extra={"features": list(panel.columns), "target_idx": target_idx,
                               "forecast_config": cfg.model_dump(),
                               "train_end": str(dates[val[-1] + cfg.seq_len + cfg.pred_len - 1].date()),
                               "history": fitted.result.history})
                    log(f"  checkpoint: {path}")
    return np.concatenate([f[2] for f in folds]), results


def _resolve_checkpoint(path):
    path = Path(path)
    if (path / "meta.json").exists():
        return path
    runs = sorted(p for p in path.iterdir() if (p / "meta.json").exists()) if path.is_dir() else []
    if not runs:
        raise SystemExit(f"no checkpoint (meta.json) under {path}")
    return runs[-1]


def run_checkpoints(panel, paths, log=print):
    """Score saved models on windows whose targets all fall after their training data. No retraining."""
    values = panel.values.astype(np.float32)
    dates = panel.index
    target_idx = list(panel.columns).index(TARGET)
    loaded = []
    for p in paths:
        p = _resolve_checkpoint(p)
        model, scaler, meta = load_checkpoint(p)
        if meta["features"] != list(panel.columns):
            raise SystemExit(f"{p}: trained on {meta['features']}, panel has {list(panel.columns)}")
        loaded.append((model, scaler, meta, p))

    cfgs = {json.dumps({k: m["forecast_config"][k] for k in ("seq_len", "label_len", "pred_len")})
            for _, _, m, _ in loaded}
    if len(cfgs) > 1:
        raise SystemExit("checkpoints use different seq_len/label_len/pred_len; evaluate them separately")
    cfg = ForecastConfig(**loaded[0][2]["forecast_config"])
    train_end = max(pd.Timestamp(m["train_end"]) for _, _, m, _ in loaded)
    n = n_windows(len(values), cfg)
    first_target = dates[np.arange(n) + cfg.seq_len]
    test = np.flatnonzero(first_target > train_end)
    if len(test) < 5:
        raise SystemExit(f"only {len(test)} windows after the checkpoint train end {train_end.date()}; "
                         "supply a panel extending past it")
    train = np.flatnonzero(dates[np.arange(n) + cfg.seq_len + cfg.pred_len - 1] <= train_end)
    log(f"Scoring {len(test)} windows after {train_end.date()}")

    results = {}
    for b in BASELINES:
        r = results[b.name] = SeriesResult(b.name, "baseline")
        r.preds.append(b().fit(values, target_idx, train, np.array([], int), cfg).predict(values, test, cfg))
    for model, scaler, meta, p in loaded:
        name = meta["model"]
        w = make_windows(scaler.transform(values), time_features(dates, mark_cols(name)), cfg,
                         window_kind(name)).take(test)
        r = results[name] = SeriesResult(name, "model")
        r.preds.append(scaler.inverse(predict(model, w)[..., target_idx], target_idx))
        if meta.get("history"):
            r.histories.append((meta["history"], meta.get("diagnostics", {}).get("best_epoch", 0)))
        r.latency_ms = _latency(model, w)
        r.n_params = count_params(model)
        log(f"  {name}: loaded {p}")
    return cfg, test, results


# ---------------------------------------------------------------------------
# Scoring, plots, entry point
# ---------------------------------------------------------------------------

def score(panel, cfg, test, results):
    values = panel.values.astype(np.float32)
    target_idx = list(panel.columns).index(TARGET)
    starts = test + cfg.seq_len
    rows = starts[:, None] + np.arange(cfg.pred_len)[None]
    y = values[rows, target_idx]
    has_close = "underlying_close" in panel.columns
    close = panel["underlying_close"].to_numpy() if has_close else None
    y_rw = results["random_walk"].yhat

    metrics, per_h = {}, {}
    for name, r in results.items():
        m, ph = forecast_metrics(y, r.yhat, y_ref=y_rw,
                                 origin_close=close[starts - 1] if has_close else None,
                                 actual_close=close[rows] if has_close else None)
        m.update(kind=r.kind, train_time_s=round(r.train_time_s, 2), n_params=r.n_params,
                 best_epochs=[h[1] for h in r.histories] or None,
                 **{f"latency_ms_{k}": round(v, 3) for k, v in r.latency_ms.items()})
        metrics[name], per_h[name] = m, ph
    return y, rows, metrics, per_h


def make_plots(out, panel, cfg, test, y, rows, results, per_h, metrics):
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dates = panel.index
    models = [n for n, r in results.items() if r.kind == "model"]
    shown = models + ["ridge_lags"]
    target_dates = dates[test + cfg.seq_len]
    horizon_dates = dates[test + cfg.seq_len + cfg.pred_len - 1]
    paths = []

    for name in models:
        hist = results[name].histories
        if hist:
            suffix = f" (fold {len(hist)} of {len(hist)})" if len(hist) > 1 else ""
            paths.append(plots.loss_curves(name, hist[-1][0], hist[-1][1], plot_dir / f"loss_curves_{name}.png",
                                           suffix))
    paths.append(plots.pred_vs_actual(target_dates, y[:, 0], {n: results[n].yhat[:, 0] for n in shown},
                                      plot_dir / "pred_vs_actual_step1.png",
                                      "Next-day log return: forecast vs actual", "log return"))
    paths.append(plots.pred_vs_actual(horizon_dates, y.sum(1), {n: results[n].yhat.sum(1) for n in shown},
                                      plot_dir / "pred_vs_actual_cum.png",
                                      f"Cumulative {cfg.pred_len}-day log return: forecast vs actual",
                                      "cum. log return"))
    paths.append(plots.error_by_horizon(per_h, plot_dir / "error_by_horizon.png"))
    paths.append(plots.scatter_step1(y[:, 0], {n: results[n].yhat[:, 0] for n in shown},
                                     plot_dir / "scatter_step1.png"))
    paths.append(plots.residual_hist({n: results[n].yhat[:, 0] - y[:, 0] for n in shown},
                                     plot_dir / "residual_hist.png"))
    paths.append(plots.model_comparison(metrics, plot_dir / "model_comparison.png"))
    if "underlying_close" in panel.columns:
        close = panel["underlying_close"].to_numpy()
        origin = close[test + cfg.seq_len - 1]
        implied = {n: origin * np.exp(results[n].yhat.sum(1)) for n in shown}
        paths.append(plots.implied_close(horizon_dates, close[rows[:, -1]], implied,
                                         plot_dir / "implied_close.png", cfg.pred_len))
    return paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the transformer forecasters against baselines.")
    src = p.add_argument_group("data")
    src.add_argument("--ticker", default="AAPL")
    src.add_argument("--history-months", type=int, default=24)
    src.add_argument("--features", nargs="+", help=f"default: {list(DEFAULT_FEATURES)}")
    src.add_argument("--panel", help="evaluate a saved panel CSV instead of pulling data")
    src.add_argument("--refresh", action="store_true", help="ignore today's cached panel")
    proto = p.add_argument_group("protocol")
    proto.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    proto.add_argument("--mode", choices=["holdout", "walkforward"], default="holdout")
    proto.add_argument("--folds", type=int, default=4, help="walk-forward folds")
    proto.add_argument("--test-frac", type=float, default=0.2, help="share of windows used for testing")
    proto.add_argument("--checkpoint", nargs="+", help="score saved checkpoints instead of training")
    train = p.add_argument_group("training (ForecastConfig)")
    for flag, typ in [("seq-len", int), ("label-len", int), ("pred-len", int), ("epochs", int), ("batch-size", int),
                      ("lr", float), ("weight-decay", float), ("patience", int), ("val-frac", float),
                      ("target-weight", float), ("seed", int)]:
        train.add_argument(f"--{flag}", type=typ)
    out = p.add_argument_group("output")
    out.add_argument("--out", help="output directory (default: evaluation runs/<timestamp>)")
    out.add_argument("--no-plots", action="store_true")
    out.add_argument("--save-checkpoints", action="store_true", help="save the final fold's models")
    return p.parse_args(argv)


def main(argv=None, log=print):
    args = parse_args(argv)
    t0 = time.perf_counter()
    panel, source = load_panel(args, log)
    out = Path(args.out) if args.out else EVAL_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.checkpoint:
        cfg, test, results = run_checkpoints(panel, args.checkpoint, log)
        mode = "checkpoint"
    else:
        overrides = {k: v for k, v in vars(args).items() if v is not None and k in ForecastConfig.model_fields}
        cfg = ForecastConfig(**overrides)
        _check_panel(panel, cfg)
        n = n_windows(len(panel), cfg)
        folds = (walkforward_folds(n, cfg, args.test_frac, args.folds) if args.mode == "walkforward"
                 else holdout_folds(n, cfg, args.test_frac))
        test, results = run_folds(panel, cfg, folds, args.models, save_ckpt=args.save_checkpoints,
                                  ticker=args.ticker if not args.panel else source, log=log)
        mode = args.mode

    y, rows, metrics, per_h = score(panel, cfg, test, results)
    plot_paths = [] if args.no_plots else make_plots(out, panel, cfg, test, y, rows, results, per_h, metrics)
    config = {
        "source": source, "mode": mode, "folds": args.folds if mode == "walkforward" else 1,
        "test_frac": args.test_frac, "checkpoints": args.checkpoint, "forecast_config": cfg.model_dump(),
        "panel": {"rows": len(panel), "start": str(panel.index[0].date()), "end": str(panel.index[-1].date()),
                  "features": list(panel.columns)},
        "test_targets": {"start": str(panel.index[rows[0, 0]].date()), "end": str(panel.index[rows[-1, -1]].date()),
                         "origins": int(len(test))},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    write_outputs(out, config, panel.index, test, rows, y, results, metrics, per_h, plot_paths)
    log(f"\nWrote {out}")
    log((out / "report.md").read_text(encoding="utf-8").split("## Plots")[0])
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
