"""
Forecast-accuracy metrics on unscaled log returns.

``y`` and ``yhat`` are ``[N, H]``: N forecast origins, H horizon steps.
"""

import numpy as np
from scipy import stats


def _hit_rate(pred, true):
    """Sign agreement, ignoring zero predictions (no call made) and zero actuals."""
    keep = (pred != 0) & (true != 0)
    return float(np.mean(np.sign(pred[keep]) == np.sign(true[keep]))) if keep.any() else float("nan")


def diebold_mariano(e_model, e_ref, h=1):
    """DM test on squared-error loss, with Newey-West HAC variance (Bartlett weights, lag h-1, always
    non-negative) and the Harvey small-sample correction.

    Negative statistic == the model has lower loss than the reference.
    Returns ``(stat, two_sided_p)``; NaNs if the loss differential is degenerate.
    """
    d = np.square(e_model) - np.square(e_ref)
    t = len(d)
    if t < 3 or np.allclose(d, 0):
        return float("nan"), float("nan")
    d_bar = d.mean()
    dc = d - d_bar
    var = dc @ dc / t
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / t
    if var <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var / t)
    dm *= np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=t - 1))


def forecast_metrics(y, yhat, y_ref=None, origin_close=None, actual_close=None):
    """Headline metrics plus per-horizon arrays.

    ``y_ref`` (the random-walk forecast, i.e. zeros) enables the DM tests.
    ``origin_close`` [N] and ``actual_close`` [N, H] enable price metrics.
    """
    err = yhat - y
    cum_y, cum_hat = y.sum(1), yhat.sum(1)
    m = {
        "n_origins": int(len(y)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "oos_r2_vs_zero": float(1 - np.sum(err ** 2) / np.sum(y ** 2)),
        "rmse_step1": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "hit_rate_step1": _hit_rate(yhat[:, 0], y[:, 0]),
        "hit_rate_cum": _hit_rate(cum_hat, cum_y),
        "up_rate_step1": float(np.mean(y[:, 0] > 0)),
        "ic_step1": float(np.corrcoef(yhat[:, 0], y[:, 0])[0, 1]) if np.std(yhat[:, 0]) > 0 else float("nan"),
        "cum_return_rmse": float(np.sqrt(np.mean((cum_hat - cum_y) ** 2))),
        "cum_return_mae": float(np.mean(np.abs(cum_hat - cum_y))),
    }
    if y_ref is not None:
        e_ref = y_ref - y
        m["dm_stat_step1"], m["dm_p_step1"] = diebold_mariano(err[:, 0], e_ref[:, 0], h=1)
        m["dm_stat_cum"], m["dm_p_cum"] = diebold_mariano(cum_hat - cum_y, e_ref.sum(1), h=y.shape[1])
    if origin_close is not None and actual_close is not None:
        price_hat = origin_close[:, None] * np.exp(np.cumsum(yhat, axis=1))
        perr = price_hat - actual_close
        m["price_rmse"] = float(np.sqrt(np.mean(perr ** 2)))
        m["price_mape"] = float(np.mean(np.abs(perr) / actual_close))
        m["price_mape_horizon"] = float(np.mean(np.abs(perr[:, -1]) / actual_close[:, -1]))

    per_h = {
        "rmse": np.sqrt(np.mean(err ** 2, axis=0)),
        "mae": np.mean(np.abs(err), axis=0),
        "hit_rate": np.array([_hit_rate(yhat[:, h], y[:, h]) for h in range(y.shape[1])]),
    }
    return m, per_h
