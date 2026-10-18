"""Synthetic feature panels for tests: noisy returns with a weak predictable cycle."""

import numpy as np
import pandas as pd


def synthetic_panel(n=500, seed=0, signal=0.004, noise=0.01, end=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n)
    r = noise * rng.standard_normal(n) + signal * np.sin(np.arange(n) / 5)
    close = 100 * np.exp(np.cumsum(r))
    log_return = np.r_[r[0], np.diff(np.log(close))]
    vol = pd.Series(log_return).rolling(21, min_periods=2).std().bfill().to_numpy() * np.sqrt(252)
    vix = 15 + 5 * np.sin(np.arange(n) / 40) + rng.standard_normal(n)
    return pd.DataFrame({"log_return": log_return, "underlying_close": close, "realized_vol": vol, "VIX": vix},
                        index=idx)
