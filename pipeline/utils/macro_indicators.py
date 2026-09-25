import os
import time
from datetime import datetime

import dotenv
import pandas as pd
import yfinance as yf
from fredapi import Fred

# Default tickers. ^VIX = CBOE Volatility Index.
# ^TNX = 10-Year Treasury Yield Index (quoted x10, e.g. 42.5 => 4.25%).
DEFAULT_YF_TICKERS = {"^VIX": "VIX", "^TNX": "10Y_Treasury_Yield"}

# Tickers whose raw value must be divided by 10 to become a percent.
_SCALE_BY_TENTH = {"^TNX"}


def _download_yf(tickers, start_date, end_date, max_retries=3, pause=2.0):
    """Download close prices for every ticker, retrying tickers Yahoo drops.

    ``yf.download`` reports partial success ("1 of 2 completed") and returns a
    frame that is silently missing the failed ticker's column, which makes
    downstream renames and arithmetic fail non-deterministically. This retries
    only the missing tickers and raises if any are still absent.
    """
    tickers = list(tickers)
    collected = pd.DataFrame()
    missing = tickers

    for attempt in range(max_retries):
        raw = yf.download(
            missing,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            group_by="column",
        )

        if not raw.empty:
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            # A single ticker comes back as a Series or a single-column frame
            # with no ticker name; label it explicitly.
            if isinstance(close, pd.Series):
                close = close.to_frame(name=missing[0])
            elif len(missing) == 1:
                close.columns = [missing[0]]

            # Keep only columns that actually carry data.
            close = close.dropna(axis=1, how="all")
            collected = (
                close if collected.empty else collected.join(close, how="outer")
            )

        missing = [t for t in tickers if t not in collected.columns]
        if not missing:
            break

        if attempt < max_retries - 1:
            print(f"Retrying {len(missing)} ticker(s): {', '.join(missing)}")
            time.sleep(pause)

    if missing:
        raise RuntimeError(
            "yfinance returned no data for: "
            f"{', '.join(missing)} (after {max_retries} attempts). "
            "Check the ticker symbols, the date range, and your connection."
        )

    return collected[tickers]


def get_macro_data(
    start_date="2025-01-01",
    end_date=None,
    yf_tickers=None,
    cpi_lookback_start="2024-01-01",
):
    """Build a daily macro indicator table.

    Returns a DataFrame indexed by calendar day from ``start_date`` to
    ``end_date`` with VIX, the 10-year Treasury yield, daily breakeven
    inflation, and official CPI year-over-year inflation.

    ``yf_tickers`` maps a Yahoo symbol to its output column name.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if yf_tickers is None:
        yf_tickers = dict(DEFAULT_YF_TICKERS)

    # Load variables from the .env file into the environment.
    dotenv.load_dotenv()
    if not os.environ.get("FRED_API_KEY"):
        raise RuntimeError(
            "FRED_API_KEY is not set. Add it to your .env file or environment."
        )

    # fredapi automatically reads os.environ['FRED_API_KEY'].
    fred = Fred()

    print("Pulling financial data...")

    # 1. Fetch market indicators from yfinance.
    market_data = _download_yf(list(yf_tickers), start_date, end_date)

    # Rescale quoted-x10 yield indices before renaming, while we still know
    # the original symbols.
    for symbol in market_data.columns.intersection(list(_SCALE_BY_TENTH)):
        market_data[symbol] = market_data[symbol] / 10

    market_data = market_data.rename(columns=yf_tickers)

    # 2. Daily 10-Year Breakeven Inflation Rate, in percent.
    daily_inflation = fred.get_series("T10YIE", start_date, end_date)
    inflation_df = pd.DataFrame(
        {"Daily_Breakeven_Inflation": daily_inflation}
    )

    # 3. Official monthly inflation (CPI). Pulled from an earlier start so the
    # 12-month change is defined at start_date.
    cpi_series = fred.get_series("CPIAUCSL", cpi_lookback_start, end_date)
    cpi_df = pd.DataFrame({"CPI": cpi_series})
    # Resample to month start so a missing release can't shift the 12-period
    # window into comparing the wrong months.
    cpi_df = cpi_df.resample("MS").last()
    cpi_df["Official_Inflation_YoY"] = cpi_df["CPI"].pct_change(periods=12) * 100

    # 4. Merge onto a unified daily calendar.
    daily_calendar = pd.date_range(start=start_date, end=end_date, freq="D")
    master_df = pd.DataFrame(index=daily_calendar)

    for frame in (market_data, inflation_df, cpi_df[["Official_Inflation_YoY"]]):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        master_df = master_df.join(frame, how="left")

    # Forward-fill weekends, market holidays, and the monthly CPI cadence.
    master_df = master_df.ffill()

    return master_df


if __name__ == "__main__":
    df = get_macro_data()
    print("\n--- Final Consolidated Dataset Sample ---")
    print(df.head(5))
