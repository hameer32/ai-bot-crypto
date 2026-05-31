"""
yfinance OHLCV Fetcher
======================
Fetches OHLCV data from Yahoo Finance (via yfinance) and normalises it
to the same format as our Binance-sourced cache:
  - UTC DatetimeIndex named "timestamp"
  - columns: open, high, low, close, volume
  - Parquet-compatible (snappy compression)

Supported assets:
  - Indian indices:  ^NSEI (NIFTY 50), ^BSESN (SENSEX), ^CNX200 (NIFTY 200), ^NSEBANK (NIFTY Bank)
  - Indian equities: RELIANCE.NS, HDFCBANK.NS, … (NSE suffix)
  - Commodities:     GC=F (Gold futures), SI=F (Silver futures)
  - US indices:      ^SPX, ^NDX, ^DJI — anything yfinance supports

Timeframe mapping (yfinance → our naming convention):
  "1d"  → "1d"
  "1w"  → "1wk" internally, cached as "1w"
  "1h"  → "1h"
  "4h"  → resampled from 1h, cached as "4h"
  "5m"  → "5m"
  "15m" → "15m"

Data availability limits (approximate):
  5m / 15m : ~60 days
  1h       : ~730 days (2 years)
  1d / 1w  : full history

Note: Indian market hours are 9:15–15:30 IST (UTC+5:30). Intraday candles
will have gaps outside those hours — this is correct, not a data error.
"""
from __future__ import annotations
import logging
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Map our internal TF names → yfinance interval strings
_YF_INTERVAL: dict[str, str | None] = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  None,   # must resample from 1h
    "1d":  "1d",
    "1w":  "1wk",
}

# Maximum lookback per yfinance interval (conservative limits)
_MAX_DAYS: dict[str, int] = {
    "1m":  7,
    "5m":  59,
    "15m": 59,
    "30m": 59,
    "1h":  720,   # 720 to stay safely within yfinance's rolling 730-day window
    "4h":  720,   # limited by 1h source
    "1d":  9999,
    "1w":  9999,
}


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV DataFrame to 4h bars."""
    if df_1h.empty:
        return df_1h
    agg = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume": "sum",
    }
    df4 = df_1h.resample("4h", closed="left", label="left").agg(agg)
    df4 = df4.dropna(subset=["close"])
    return df4


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert yfinance output to standard UTC DatetimeIndex OHLCV DataFrame."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = raw.copy()

    # yfinance may return MultiIndex columns when auto_adjust=True
    if isinstance(df.columns, pd.MultiIndex):
        # columns are (Price, Ticker) — drop the ticker level
        df.columns = df.columns.get_level_values(0)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()

    # Ensure we have all OHLCV columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # Normalise index to UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Drop rows where all OHLCV values are NaN
    df = df.dropna(subset=["close"])

    # Cast to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Normalised %d rows for %s", len(df), ticker)
    return df


def fetch_yf_ohlcv(
    ticker: str,
    timeframe: str,
    lookback_days: int = 730,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance for a single ticker + timeframe.

    Parameters
    ----------
    ticker        : yfinance ticker string (e.g. "^NSEI", "RELIANCE.NS", "GC=F")
    timeframe     : our TF string ("1d", "1w", "1h", "4h", "5m", "15m")
    lookback_days : how many calendar days back to fetch

    Returns
    -------
    pd.DataFrame with UTC DatetimeIndex and columns: open, high, low, close, volume
    """
    max_days = _MAX_DAYS.get(timeframe, 730)
    actual_days = min(lookback_days, max_days)

    if actual_days < lookback_days:
        logger.warning(
            "%s %s: yfinance limits to %d days (requested %d)",
            ticker, timeframe, actual_days, lookback_days
        )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=actual_days)

    # For 4h: fetch 1h then resample
    fetch_tf = timeframe if timeframe != "4h" else "1h"
    yf_interval = _YF_INTERVAL.get(fetch_tf)
    if yf_interval is None:
        logger.error("Unsupported timeframe: %s", timeframe)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)
            raw = t.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )
    except Exception as exc:
        logger.error("yfinance fetch error for %s %s: %s", ticker, timeframe, exc)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = _normalise(raw, ticker)

    if timeframe == "4h" and not df.empty:
        df = _resample_4h(df)

    return df


def fetch_yf_all(
    tickers: list[str],
    timeframes: list[str],
    lookback_days: int = 730,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Fetch all tickers × timeframes from yfinance.

    Returns dict keyed by (ticker, timeframe) → DataFrame.
    Tickers that fail or return empty data are logged but not raised.
    """
    results: dict[tuple[str, str], pd.DataFrame] = {}
    total = len(tickers) * len(timeframes)
    done = 0

    for ticker in tickers:
        for tf in timeframes:
            done += 1
            logger.info("[%d/%d] Fetching %s %s ...", done, total, ticker, tf)
            df = fetch_yf_ohlcv(ticker, tf, lookback_days)
            if df.empty:
                logger.warning("No data: %s %s", ticker, tf)
            results[(ticker, tf)] = df

    return results
