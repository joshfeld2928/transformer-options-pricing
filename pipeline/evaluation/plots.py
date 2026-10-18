"""

Evaluation figures (matplotlib, Agg backend, PNG).



Colour rules:

- Transformers take the first four categorical slots in a fixed order, so

  a model keeps its colour across every figure.

- Baselines are neutral greys, told apart by dash pattern.

- Actuals are drawn in primary ink.

Comparisons of many series use small multiples rather than one crowded axis.

"""



import matplotlib



matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402



SURFACE = "#fcfcfb"

INK = "#0b0b0b"

INK_2 = "#52514e"

GRID = "#e4e3df"

MODEL_COLORS = {"autoformer": "#2a78d6", "fedformer": "#eb6834", "informer": "#1baf7a", "pyraformer": "#eda100"}

BASELINE_STYLES = {

    "random_walk": ("#52514e", (0, (5, 2))),

    "hist_mean": ("#7d7c77", (0, (1, 1.5))),

    "ar1": ("#9a9993", (0, (6, 2, 1, 2))),

    "ridge_lags": ("#3a3936", (0, (2, 2))),

}



plt.rcParams.update({

    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,

    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2,

    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,

    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",

    "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,

    "figure.dpi": 110,

})





def _style(name):

    """(color, linestyle) for a series."""

    if name in MODEL_COLORS:

        return MODEL_COLORS[name], "-"

    return BASELINE_STYLES.get(name, (INK_2, "--"))





def _save(fig, path):

    fig.tight_layout()

    fig.savefig(path, bbox_inches="tight")

    plt.close(fig)

    return path





def _grid(n, ncols=2, size=(5.2, 3.0)):

    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)

    for ax in axes.flat[n:]:

        ax.set_visible(False)

    return fig, axes.flat





def loss_curves(name, history, best_epoch, path, title_suffix=""):

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], color="#2a78d6", label="train loss")

    if history.get("val_loss"):

        ax.plot(epochs[:len(history["val_loss"])], history["val_loss"], color="#eb6834", label="validation loss")

        ax.axvline(best_epoch, color=INK_2, linestyle=(0, (2, 2)), linewidth=1)

        ax.annotate(f"best epoch {best_epoch}", (best_epoch, ax.get_ylim()[1]), xytext=(4, -12),

                    textcoords="offset points", color=INK_2, fontsize=8)

    ax.set_xlabel("epoch")

    ax.set_ylabel("weighted MSE (scaled units)")

    ax.set_title(f"{name}: training curves{title_suffix}")

    ax.legend(loc="upper right")

    return _save(fig, path)





def pred_vs_actual(dates, y_true, preds, path, title, ylabel):

    """Small multiples: one panel per model, actual in ink, forecast in the model colour."""

    names = list(preds)

    fig, axes = _grid(len(names))

    for ax, name in zip(axes, names):

        ax.plot(dates, y_true, color=INK, linewidth=1, alpha=0.55, label="actual")

        color, ls = _style(name)

        ax.plot(dates, preds[name], color=color, linestyle=ls, label=name)

        ax.axhline(0, color=INK_2, linewidth=0.8)

        ax.set_title(name)

        ax.set_ylabel(ylabel)

        ax.legend(loc="upper left", ncols=2)

        ax.tick_params(axis="x", labelrotation=30)

    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")

    return _save(fig, path)





def error_by_horizon(per_horizon, path, metric="rmse"):

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for name, ph in per_horizon.items():

        values = ph[metric]

        h = np.arange(1, len(values) + 1)

        color, ls = _style(name)

        model = name in MODEL_COLORS

        ax.plot(h, values, color=color, linestyle=ls, linewidth=2 if model else 1.6,

                marker="o" if model else None, markersize=4, label=name)

    ax.set_xlabel("horizon (trading days ahead)")

    ax.set_ylabel(f"{metric.upper()} of daily log return")

    ax.set_title(f"{metric.upper()} by forecast horizon (lower is better)")

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    return _save(fig, path)





def scatter_step1(y_true, preds, path):

    names = list(preds)

    fig, axes = _grid(len(names), size=(4.2, 3.6))

    lim = float(np.nanmax(np.abs(y_true))) * 1.05

    for ax, name in zip(axes, names):

        color, _ = _style(name)

        ax.scatter(preds[name], y_true, s=14, color=color, alpha=0.6, edgecolors=SURFACE, linewidths=0.5)

        ax.plot([-lim, lim], [-lim, lim], color=INK_2, linewidth=1, linestyle=(0, (2, 2)))

        ax.set_xlabel("predicted (next day)")

        ax.set_ylabel("actual")

        corr = np.corrcoef(preds[name], y_true)[0, 1] if np.std(preds[name]) > 0 else float("nan")

        ax.set_title(f"{name}  (corr {corr:+.2f})")

    fig.suptitle("Next-day log return: predicted vs actual", x=0.01, ha="left", fontweight="bold")

    return _save(fig, path)





def residual_hist(residuals, path):

    names = list(residuals)

    fig, axes = _grid(len(names), size=(4.2, 3.0))

    bins = np.histogram_bin_edges(np.concatenate(list(residuals.values())), bins=30)

    for ax, name in zip(axes, names):

        color, _ = _style(name)

        r = residuals[name]

        ax.hist(r, bins=bins, color=color, edgecolor=SURFACE, linewidth=1)

        ax.axvline(0, color=INK_2, linewidth=1)

        ax.set_title(f"{name}  (mean {r.mean():+.4f}, sd {r.std():.4f})")

        ax.set_xlabel("residual (pred - actual), next day")

    fig.suptitle("Next-day residual distribution", x=0.01, ha="left", fontweight="bold")

    return _save(fig, path)





def model_comparison(metrics, path):

    """One horizontal-bar panel per metric (they don't share a scale).



    Errors are shown as % change vs the random walk: all the raw RMSEs sit

    within a few percent of each other, which bars from zero would hide.

    """

    rw = metrics["random_walk"]

    panels = [

        ("rmse", "RMSE vs random walk (%, lower better)", lambda m: 100 * (m["rmse"] / rw["rmse"] - 1), "{:+.1f}%"),

        ("mae", "MAE vs random walk (%, lower better)", lambda m: 100 * (m["mae"] / rw["mae"] - 1), "{:+.1f}%"),

        ("hit_rate_cum", "Direction hit rate, cumulative horizon", lambda m: m.get("hit_rate_cum"), "{:.0%}"),

        ("oos_r2_vs_zero", "Out-of-sample R² vs zero forecast", lambda m: m.get("oos_r2_vs_zero"), "{:+.3f}"),

    ]

    names = list(metrics)

    fig, axes = _grid(len(panels), size=(5.0, 0.32 * len(names) + 1.2))

    y = np.arange(len(names))

    for ax, (key, title, value, fmt) in zip(axes, panels):

        vals = np.array([np.nan if value(metrics[n]) is None else value(metrics[n]) for n in names], float)

        colors = [_style(n)[0] for n in names]

        ax.barh(y, np.nan_to_num(vals), height=0.6, color=colors, edgecolor=SURFACE, linewidth=2)

        span = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0

        for yi, v in zip(y, vals):

            # Labels sit just past the bar's outer end, on whichever side of zero it grows.

            x, ha, dx = (0, "left", 3) if np.isnan(v) else (v, "left" if v >= 0 else "right", 3 if v >= 0 else -3)

            ax.annotate("n/a" if np.isnan(v) else fmt.format(v), (x, yi), xytext=(dx, 0),

                        textcoords="offset points", va="center", ha=ha, fontsize=8, color=INK_2)

        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.set_xlim(lo - 0.25 * span if lo < 0 else 0, hi + 0.25 * span if hi > 0 else 0.05 * span)

        ax.set_yticks(y, names)

        ax.invert_yaxis()

        ax.axvline(0, color=INK_2, linewidth=0.8)

        if key == "hit_rate_cum":

            ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=(0, (2, 2)))

        ax.grid(axis="y", visible=False)

        ax.set_title(title)

    fig.suptitle("Model comparison on the test set", x=0.01, ha="left", fontweight="bold")

    return _save(fig, path)





def implied_close(dates, actual, preds, path, horizon):

    names = list(preds)

    fig, axes = _grid(len(names))

    for ax, name in zip(axes, names):

        color, ls = _style(name)

        ax.plot(dates, actual, color=INK, linewidth=1.2, alpha=0.7, label="actual close")

        ax.plot(dates, preds[name], color=color, linestyle=ls, label=f"{name} forecast")

        ax.set_title(name)

        ax.set_ylabel("price")

        ax.legend(loc="upper left")

        ax.tick_params(axis="x", labelrotation=30)

    fig.suptitle(f"Close {horizon} trading days ahead: forecast vs actual (by target date)", x=0.01, ha="left",

                 fontweight="bold")

    return _save(fig, path)

