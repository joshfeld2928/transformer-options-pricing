import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

"""
For each contract–day pair we construct the following features:
– Underlying close price St and log return rt = log(St/St− 1)
– Past option mid-price Ct = (Cbid t + Cask t )/2 – only for forecasting
– Black–Scholes implied volatility σimp,t, solved numerically
– Time to maturity τ and moneyness mt = St /Kt
– Option type: 1 for call, 0 for put

"""

# Yahoo exposes only the *current* option chain, never historical quotes. The
# contract terms (strike, expiry, type) and the quoted bid/ask are therefore a
# single snapshot; only the underlying-derived columns (St, rt, tau, moneyness)
# vary across the day axis. Columns sourced from the snapshot are suffixed
# ``_snapshot`` so nothing downstream mistakes them for a time series.
_SNAPSHOT_COLUMNS = ("bid", "ask", "lastPrice", "volume", "openInterest")

# Trading days per calendar year, used to annualise realised volatility.
_TRADING_DAYS = 252

# Bounds for the implied-volatility root search, in annualised vol units.
_IV_BOUNDS = (1e-4, 5.0)


def _retry(fn, what, max_retries=3, pause=2.0):
    """Call ``fn`` until it returns a non-empty result, then return it.

    Yahoo intermittently answers with an empty frame or a transient HTTP error
    rather than failing outright, which otherwise surfaces downstream as a
    confusing KeyError. Raises RuntimeError once the attempts are exhausted.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = fn()
            if result is not None and len(result):
                return result
            last_error = "empty response"
        except Exception as exc:  # noqa: BLE001 - surfaced in the raise below
            last_error = exc

        if attempt < max_retries - 1:
            print(f"Retrying {what} ({last_error})")
            time.sleep(pause)

    raise RuntimeError(
        f"yfinance returned no data for {what} "
        f"(after {max_retries} attempts): {last_error}. "
        "Check the ticker symbol, the date range, and your connection."
    )


def _get_underlying_history(ticker, start_date, end_date):
    """Daily close and log return for one underlying, indexed by date."""
    history = _retry(
        lambda: yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            group_by="column",
        ),
        f"underlying history of {ticker}",
    )

    close = history["Close"]
    # A single ticker can come back either as a plain column or under a
    # ticker-keyed second level, depending on the yfinance version.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    frame = pd.DataFrame({"underlying_close": close.astype(float)})
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    frame = frame.sort_index()

    frame["log_return"] = np.log(
        frame["underlying_close"] / frame["underlying_close"].shift(1)
    )
    # Realised vol over the trailing month, the seed for the IV root search.
    frame["realized_vol"] = frame["log_return"].rolling(21).std() * math.sqrt(
        _TRADING_DAYS
    )

    return frame


def _get_option_chain(ticker, max_expiries=None):
    """Every listed contract for one underlying as a single tidy frame."""
    handle = yf.Ticker(ticker)
    expiries = _retry(lambda: handle.options, f"expiry list of {ticker}")

    if max_expiries is not None:
        expiries = list(expiries)[:max_expiries]

    contracts = []
    for expiry in expiries:
        chain = _retry(
            lambda expiry=expiry: handle.option_chain(expiry),
            f"{ticker} chain expiring {expiry}",
        )

        for side, is_call in (("calls", 1), ("puts", 0)):
            leg = getattr(chain, side).copy()
            if leg.empty:
                continue
            leg["expiry"] = pd.Timestamp(expiry).normalize()
            leg["option_type"] = is_call
            contracts.append(leg)

    if not contracts:
        raise RuntimeError(f"{ticker} has no listed option contracts.")

    chain_df = pd.concat(contracts, ignore_index=True)
    chain_df = chain_df.rename(columns={"contractSymbol": "contract_symbol"})

    return chain_df


def _implied_vol(price, spot, strike, tau, rate, is_call, guess):
    """Black–Scholes implied volatility, solved numerically on the mid-price.

    Returns NaN when the quote is outside the no-arbitrage bounds, which is
    common for stale or wide-spread contracts and has no real root.
    """
    if not np.isfinite([price, spot, strike, tau]).all() or tau <= 0 or price <= 0:
        return np.nan

    discount = strike * math.exp(-rate * tau)
    intrinsic = max(spot - discount, 0.0) if is_call else max(discount - spot, 0.0)
    upper = spot if is_call else discount
    if price <= intrinsic or price >= upper:
        return np.nan

    def difference(sigma):
        return _bs_price(spot, strike, tau, rate, sigma, is_call) - price

    low, high = _IV_BOUNDS
    try:
        if difference(low) * difference(high) > 0:
            return np.nan
        return brentq(difference, low, high, xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        # ``guess`` keeps a usable feature where the solve fails outright.
        return guess if np.isfinite(guess) else np.nan


def _bs_price(spot, strike, tau, rate, sigma, is_call):
    """Black–Scholes price of a European option on a non-dividend payer."""
    root_tau = sigma * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / root_tau
    d2 = d1 - root_tau
    discount = strike * math.exp(-rate * tau)

    if is_call:
        return spot * norm.cdf(d1) - discount * norm.cdf(d2)
    return discount * norm.cdf(-d2) - spot * norm.cdf(-d1)


def get_company_data(
    tickers,
    start_date="2025-01-01",
    end_date=None,
    risk_free_rate=0.04,
    max_expiries=None,
    min_time_to_maturity=1 / 365,
    solve_iv=False,
):
    """Build the contract–day feature panel for one or more underlyings.

    ``tickers`` is a single symbol or an iterable of them. Returns a DataFrame
    with one row per (contract, day) carrying the features named at the top of
    this module, indexed by ``(ticker, contract_symbol, date)`` and sorted.

    Rows are emitted only for days on which the contract was still alive, i.e.
    where time to maturity exceeds ``min_time_to_maturity``.

    Set ``max_expiries`` to cap how many expiry dates are pulled per ticker —
    a full chain is several hundred contracts per expiry. ``solve_iv=False``
    skips the root search and falls back to Yahoo's quoted implied vol, which
    is much faster but noisier.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("Pass at least one ticker symbol.")

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    panels = []

    for ticker in tickers:
        print(f"Pulling option data for {ticker}...")


        underlying = _get_underlying_history(ticker, start_date, end_date)
        chain = _get_option_chain(ticker, max_expiries=max_expiries)

        # Cross-join the contracts with the daily calendar: every contract is
        # repeated once per trading day of the underlying.
        panel = chain.merge(underlying.reset_index(), how="cross")

        panel["time_to_maturity"] = (
            panel["expiry"] - panel["date"]
        ).dt.days / 365.0
        panel = panel[panel["time_to_maturity"] > min_time_to_maturity].copy()
        if panel.empty:
            print(f"  {ticker}: no contracts alive in the requested window.")
            continue

        panel["moneyness"] = panel["underlying_close"] / panel["strike"]
        # Mid-price, falling back to the last trade when a side is missing —
        # Yahoo reports 0.0 rather than NaN for an absent quote.
        bid = panel["bid"].replace(0.0, np.nan)
        ask = panel["ask"].replace(0.0, np.nan)
        panel["option_mid_price"] = (bid + ask) / 2
        panel["option_mid_price"] = panel["option_mid_price"].fillna(
            panel["lastPrice"].replace(0.0, np.nan)
        )

        if solve_iv:
            print("***Solving for implied volatility, expect wait***")
            panel["implied_vol"] = [
                _implied_vol(
                    price=row.option_mid_price,
                    spot=row.underlying_close,
                    strike=row.strike,
                    tau=row.time_to_maturity,
                    rate=risk_free_rate,
                    is_call=bool(row.option_type),
                    guess=row.realized_vol,
                )
                for row in panel.itertuples()
            ]
        else:
            panel["implied_vol"] = panel["impliedVolatility"]

        panel["ticker"] = ticker
        panel = panel.rename(
            columns={c: f"{c}_snapshot" for c in _SNAPSHOT_COLUMNS if c in panel}
        )
        panels.append(panel)
        print(f"***Finished Pulling data for {ticker}***")

    if not panels:
        raise RuntimeError(
            "No contract–day rows were produced for any requested ticker. "
            "Widen the date range or relax min_time_to_maturity."
        )

    columns = [
        "ticker",
        "contract_symbol",
        "date",
        "expiry",
        "strike",
        "option_type",
        "underlying_close",
        "log_return",
        "option_mid_price",
        "implied_vol",
        "time_to_maturity",
        "moneyness",
        "realized_vol",
        *[f"{c}_snapshot" for c in _SNAPSHOT_COLUMNS],
    ]

    master_df = pd.concat(panels, ignore_index=True)
    master_df = master_df[[c for c in columns if c in master_df.columns]]

    return master_df.set_index(["ticker", "contract_symbol", "date"]).sort_index()


if __name__ == "__main__":
    df = get_company_data(["AAPL"], max_expiries=1)
    print("\n--- Contract–Day Panel Sample ---")
    print(df.head(5))
    print(f"\n{len(df):,} contract-day rows")
