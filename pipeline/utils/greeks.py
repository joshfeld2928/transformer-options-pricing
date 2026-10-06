"""
Vectorised Black–Scholes pricing and Greeks, plus the forecast-driven option
valuation used by the Greeks/validation agent and the final valuation step.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import yfinance as yf

from .company_data import _retry

_MIN_TAU = 1 / 365


def bs_price_and_greeks(spot, strike, tau, rate, sigma, is_call):
    """Price, delta, gamma, vega (per 1 vol pt), theta (per day), rho (per 1%)."""
    spot, strike, tau, sigma = (np.asarray(a, dtype=float) for a in (spot, strike, tau, sigma))
    is_call = np.asarray(is_call).astype(bool)

    root_tau = sigma * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / root_tau
    d2 = d1 - root_tau
    disc = strike * np.exp(-rate * tau)
    pdf = norm.pdf(d1)

    call = spot * norm.cdf(d1) - disc * norm.cdf(d2)
    put = disc * norm.cdf(-d2) - spot * norm.cdf(-d1)
    theta_common = -spot * pdf * sigma / (2 * np.sqrt(tau))

    return pd.DataFrame({
        "price": np.where(is_call, call, put),
        "delta": np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1),
        "gamma": pdf / (spot * root_tau),
        "vega": spot * pdf * np.sqrt(tau) / 100,
        "theta": np.where(is_call,
                          theta_common - rate * disc * norm.cdf(d2),
                          theta_common + rate * disc * norm.cdf(-d2)) / 365,
        "rho": np.where(is_call, disc * tau * norm.cdf(d2), -disc * tau * norm.cdf(-d2)) / 100,
    })


def load_contracts(ticker, spot, min_expiry, max_expiries=2, moneyness_band=0.1):
    """Near-the-money contracts from the first ``max_expiries`` expiries after
    ``min_expiry``, keeping only those with a usable quote and implied vol."""
    handle = yf.Ticker(ticker)
    expiries = [e for e in _retry(lambda: handle.options, f"expiry list of {ticker}")
                if pd.Timestamp(e) > pd.Timestamp(min_expiry)][:max_expiries]
    if not expiries:
        raise RuntimeError(f"{ticker} has no listed expiries after {min_expiry}")

    legs = []
    for expiry in expiries:
        chain = _retry(lambda e=expiry: handle.option_chain(e), f"{ticker} chain expiring {expiry}")
        for side, is_call in (("calls", 1), ("puts", 0)):
            leg = getattr(chain, side).assign(expiry=pd.Timestamp(expiry), option_type=is_call)
            legs.append(leg)
    chain = pd.concat(legs, ignore_index=True).rename(columns={"contractSymbol": "contract_symbol"})

    bid, ask = chain["bid"].replace(0.0, np.nan), chain["ask"].replace(0.0, np.nan)
    chain["mid_price"] = ((bid + ask) / 2).fillna(chain["lastPrice"].replace(0.0, np.nan))
    chain["moneyness"] = spot / chain["strike"]
    keep = (
        chain["moneyness"].between(1 - moneyness_band, 1 + moneyness_band)
        & chain["mid_price"].gt(0)
        & chain["impliedVolatility"].between(0.01, 5.0)
    )
    return chain.loc[keep, ["contract_symbol", "expiry", "strike", "option_type",
                            "mid_price", "impliedVolatility", "moneyness"]].reset_index(drop=True)


def value_contracts(contracts, forecasts, accepted_models, as_of, spot, rate):
    """Value each contract today and at the forecast horizon.

    The horizon spot is the mean of the accepted models' implied closes on the
    final forecast day; ``value_low``/``value_high`` span the individual models.
    Contracts expiring before the horizon are dropped.
    """
    as_of = pd.Timestamp(as_of)
    horizon = pd.Timestamp(forecasts[accepted_models[0]].dates[-1])
    horizon_spots = {m: forecasts[m].implied_close[-1] for m in accepted_models}
    spot_h = float(np.mean(list(horizon_spots.values())))

    c = contracts.copy()
    c["tau_now"] = (c["expiry"] - as_of).dt.days / 365
    c["tau_horizon"] = (c["expiry"] - horizon).dt.days / 365
    c = c[c["tau_horizon"] > _MIN_TAU].reset_index(drop=True)
    if c.empty:
        return c

    sigma, k, call = c["impliedVolatility"], c["strike"], c["option_type"]
    now = bs_price_and_greeks(spot, k, c["tau_now"], rate, sigma, call).add_suffix("_now")
    at_h = bs_price_and_greeks(spot_h, k, c["tau_horizon"], rate, sigma, call).add_suffix("_horizon")

    per_model = np.column_stack([
        bs_price_and_greeks(s, k, c["tau_horizon"], rate, sigma, call)["price"]
        for s in horizon_spots.values()
    ])

    out = pd.concat([c, now, at_h], axis=1)
    out["value_low"], out["value_high"] = per_model.min(axis=1), per_model.max(axis=1)
    out["expected_pnl"] = out["price_horizon"] - out["mid_price"]
    out["model_vs_market"] = out["price_now"] - out["mid_price"]
    out["horizon_date"], out["horizon_spot"] = horizon.date(), spot_h
    return out


def sanity_check_forecasts(forecasts, last_close, max_daily_abs_return=0.15, max_total_move=0.5):
    """Deterministic checks on each model's forecast; returns issues per model."""
    issues = {}
    for name, f in forecasts.items():
        found = []
        r = np.asarray(f.predicted_log_return)
        if np.abs(r).max() > max_daily_abs_return:
            found.append(f"daily |log return| up to {np.abs(r).max():.3f}")
        move = f.implied_close[-1] / last_close - 1
        if abs(move) > max_total_move:
            found.append(f"horizon move {move:+.1%}")
        if np.std(r) < 1e-6:
            found.append("flat forecast")
        issues[name] = found
    terminal = [f.implied_close[-1] for f in forecasts.values()]
    dispersion = (max(terminal) - min(terminal)) / last_close
    return {"per_model": issues, "terminal_close_dispersion": round(float(dispersion), 4)}


def greek_bound_violations(table):
    """Count rows breaking no-arbitrage Greek/price bounds, per check."""
    calls, puts = table["option_type"] == 1, table["option_type"] == 0
    spot = table["moneyness"] * table["strike"]
    intrinsic = np.where(calls, np.maximum(spot - table["strike"], 0), np.maximum(table["strike"] - spot, 0))
    checks = {
        "call_delta_outside_[0,1]": calls & ~table["delta_now"].between(0, 1),
        "put_delta_outside_[-1,0]": puts & ~table["delta_now"].between(-1, 0),
        "negative_gamma": table["gamma_now"] < 0,
        "negative_vega": table["vega_now"] < 0,
        # European puts can legitimately price below intrinsic, so only calls.
        "call_below_intrinsic": calls & (table["price_now"] < intrinsic - 1e-8),
    }
    return {name: int(mask.sum()) for name, mask in checks.items()}
