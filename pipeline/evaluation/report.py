"""Writes the evaluation artifacts: JSON/CSV metrics, predictions and a markdown report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_COLS = [
    ("rmse", "RMSE", "{:.5f}"), ("mae", "MAE", "{:.5f}"), ("oos_r2_vs_zero", "R² vs 0", "{:+.3f}"),
    ("hit_rate_step1", "Hit 1d", "{:.1%}"), ("hit_rate_cum", "Hit cum", "{:.1%}"), ("ic_step1", "IC 1d", "{:+.3f}"),
    ("cum_return_rmse", "Cum RMSE", "{:.4f}"), ("price_mape_horizon", "MAPE @H", "{:.2%}"),
    ("dm_p_step1", "DM p 1d", "{:.3f}"), ("dm_p_cum", "DM p cum", "{:.3f}"),
]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f.format(v)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _verdicts(metrics):
    """Plain-English verdict per model against the random walk and the best baseline."""
    baselines = {n: m for n, m in metrics.items() if m["kind"] == "baseline"}
    best_base = min(baselines, key=lambda n: baselines[n]["rmse"])
    lines = []
    for name, m in metrics.items():
        if m["kind"] != "model":
            continue
        p, stat = m.get("dm_p_step1"), m.get("dm_stat_step1")
        if p is None or np.isnan(p):
            rw = "DM test unavailable"
        elif p < 0.05 and stat < 0:
            rw = f"**significantly beats** the random walk next-day (DM p={p:.3f})"
        elif p < 0.05:
            rw = f"is **significantly worse** than the random walk next-day (DM p={p:.3f})"
        else:
            rw = f"is not distinguishable from the random walk next-day (DM p={p:.3f})"
        vs = "beats" if m["rmse"] < baselines[best_base]["rmse"] else "does not beat"
        lines.append(f"- **{name}** {rw}; RMSE {m['rmse']:.5f} {vs} the best baseline "
                     f"({best_base}, {baselines[best_base]['rmse']:.5f}).")
    return lines


def write_outputs(out, config, dates, test, rows, y, results, metrics, per_h, plot_paths):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(_clean(config), indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")

    flat = pd.DataFrame({n: {k: v for k, v in m.items() if not isinstance(v, list)} for n, m in metrics.items()}).T
    flat.index.name = "series"
    flat.sort_values("rmse").to_csv(out / "metrics.csv")

    per_rows = [{"series": n, "horizon": h + 1, **{k: float(v[h]) for k, v in ph.items()}}
                for n, ph in per_h.items() for h in range(len(next(iter(ph.values()))))]
    pd.DataFrame(per_rows).to_csv(out / "per_horizon.csv", index=False)

    pred_rows = []
    for i, origin in enumerate(test):
        for h in range(y.shape[1]):
            row = {"origin_row": int(origin), "target_date": str(dates[rows[i, h]].date()), "horizon": h + 1,
                   "actual": float(y[i, h])}
            row.update({n: float(r.yhat[i, h]) for n, r in results.items()})
            pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)

    ranked = sorted(metrics, key=lambda n: metrics[n]["rmse"])
    header = "| series | kind | " + " | ".join(c[1] for c in TABLE_COLS) + " |"
    table = [header, "|" + "---|" * (len(TABLE_COLS) + 2)]
    for n in ranked:
        m = metrics[n]
        table.append(f"| {n} | {m['kind']} | " + " | ".join(_fmt(m.get(k), f) for k, _, f in TABLE_COLS) + " |")

    eff = ["| model | params | train time (s) | best epoch(s) | predict 1 window (ms) |", "|---|---|---|---|---|"]
    for n in ranked:
        m = metrics[n]
        if m["kind"] == "model":
            eff.append(f"| {n} | {m['n_params']:,} | {m['train_time_s']} | {m['best_epochs']} | "
                       f"{_fmt(m.get('latency_ms_batch1'), '{:.2f}')} |")

    fc, panel = config["forecast_config"], config["panel"]
    md = [
        f"# Evaluation: {config['source']} ({config['mode']})",
        "",
        f"- Panel: {panel['rows']} rows, {panel['start']} → {panel['end']}; features {panel['features']}",
        f"- Windows: seq_len={fc['seq_len']}, label_len={fc['label_len']}, pred_len={fc['pred_len']}",
        f"- Test targets: {config['test_targets']['start']} → {config['test_targets']['end']} "
        f"({config['test_targets']['origins']} forecast origins"
        + (f", {config['folds']} walk-forward folds" if config["mode"] == "walkforward" else "") + ")",
        f"- Runtime: {config['runtime_s']}s",
        "",
        "## Verdict",
        "",
        *_verdicts(metrics),
        "",
        "## Metrics (test set, daily log returns; ranked by RMSE)",
        "",
        *table,
        "",
        "*R² vs 0*: 1 − SSE/Σy², so >0 means better than forecasting zero. *Hit*: share of correct signs "
        "(zero forecasts excluded; next-day up-day rate is "
        f"{_fmt(metrics['random_walk']['up_rate_step1'], '{:.1%}')}). *IC*: correlation of forecast and outcome. "
        "*DM p*: Diebold-Mariano p-value vs the random walk (squared error; cumulative uses HAC lag H−1). "
        "*MAPE @H*: price error at the horizon.",
        "",
        "## Efficiency",
        "",
        *eff,
        "",
        "## Caveats",
        "",
        "- Daily returns are close to unpredictable; a random walk is a strong benchmark, and small test sets "
        "make differences noisy. Prefer walk-forward mode when judging a model.",
        "- `underlying_close` is non-stationary, so test-period levels can sit outside the training scaler's range.",
        "- Deviations from the papers: Autoformer uses per-sample delay aggregation in training; Pyraformer's CSCM "
        "and Informer's distilling layers use LayerNorm instead of BatchNorm.",
        "",
        "## Plots",
        "",
        *[f"![{Path(p).stem}](plots/{Path(p).name})" for p in plot_paths],
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
