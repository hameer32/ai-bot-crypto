"""
TradingView Data Fetcher
========================
Fetches OHLCV data via tvDatafeed (TradingView's internal API).
Provides significantly more historical intraday data than yfinance:
  5m  : ~3 months (5000 bars)
  15m : ~2 months
  1h  : ~7 months
  1d  : ~2 years

Supports:
  - NSE/BSE stocks (RELIANCE, HDFCBANK, TCS, etc.)
  - Indian indices (NIFTY, SENSEX, NSEBANK, CNX200)
  - MCX commodities (GOLD, SILVER)
  - Forex (EURUSD, GBPUSD, etc. via FX_IDC exchange)
  - Crypto (BTCUSDT via BINANCE exchange)

Usage:
  from data.tv_fetcher import fetch_tv_ohlcv, TV_SYMBOL_MAP
  df = fetch_tv_ohlcv("RELIANCE", "NSE", "5m", n_bars=5000)
"""
from __future__ import annotations
import logging
import time
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# TF name → tvDatafeed Interval
def _get_interval(tf: str):
    from tvDatafeed import Interval
    MAP = {
        "1m":  Interval.in_1_minute,
        "5m":  Interval.in_5_minute,
        "15m": Interval.in_15_minute,
        "30m": Interval.in_30_minute,
        "1h":  Interval.in_1_hour,
        "2h":  Interval.in_2_hour,
        "4h":  Interval.in_4_hour,
        "1d":  Interval.in_daily,
        "1w":  Interval.in_weekly,
    }
    return MAP.get(tf)

# Max bars per timeframe (TradingView limits)
TV_MAX_BARS = {
    "1m":  5000,
    "5m":  5000,
    "15m": 5000,
    "30m": 5000,
    "1h":  5000,
    "4h":  5000,
    "1d":  5000,
    "1w":  5000,
}

# NSE stock symbol mapping (our cache name → TradingView symbol + exchange)
NSE_STOCKS_TV = {
    "ADANIENT":   ("ADANIENT",   "NSE"),
    "ADANIPORTS": ("ADANIPORTS", "NSE"),
    "APOLLOHOSP": ("APOLLOHOSP", "NSE"),
    "ASIANPAINT": ("ASIANPAINT", "NSE"),
    "AXISBANK":   ("AXISBANK",   "NSE"),
    "BAJAJ-AUTO": ("BAJAJ_AUTO", "NSE"),   # dash → underscore on TV
    "BAJAJFINSV": ("BAJAJFINSV", "NSE"),
    "BAJFINANCE": ("BAJFINANCE", "NSE"),
    "BHARTIARTL": ("BHARTIARTL", "NSE"),
    "BPCL":       ("BPCL",       "NSE"),
    "BRITANNIA":  ("BRITANNIA",  "NSE"),
    "CIPLA":      ("CIPLA",      "NSE"),
    "COALINDIA":  ("COALINDIA",  "NSE"),
    "DIVISLAB":   ("DIVISLAB",   "NSE"),
    "DRREDDY":    ("DRREDDY",    "NSE"),
    "EICHERMOT":  ("EICHERMOT",  "NSE"),
    "GRASIM":     ("GRASIM",     "NSE"),
    "HCLTECH":    ("HCLTECH",    "NSE"),
    "HDFCBANK":   ("HDFCBANK",   "NSE"),
    "HDFCLIFE":   ("HDFCLIFE",   "NSE"),
    "HEROMOTOCO": ("HEROMOTOCO", "NSE"),
    "HINDALCO":   ("HINDALCO",   "NSE"),
    "HINDUNILVR": ("HINDUNILVR", "NSE"),
    "ICICIBANK":  ("ICICIBANK",  "NSE"),
    "INDUSINDBK": ("INDUSINDBK", "NSE"),
    "INFY":       ("INFY",       "NSE"),
    "ITC":        ("ITC",        "NSE"),
    "JSWSTEEL":   ("JSWSTEEL",   "NSE"),
    "KOTAKBANK":  ("KOTAKBANK",  "NSE"),
    "LT":         ("LT",         "NSE"),
    "MM":         ("M_M",        "NSE"),   # M&M → M_M on TradingView
    "MARUTI":     ("MARUTI",     "NSE"),
    "NESTLEIND":  ("NESTLEIND",  "NSE"),
    "NTPC":       ("NTPC",       "NSE"),
    "ONGC":       ("ONGC",       "NSE"),
    "POWERGRID":  ("POWERGRID",  "NSE"),
    "RELIANCE":   ("RELIANCE",   "NSE"),
    "SBICARD":    ("SBICARD",    "NSE"),
    "SBIN":       ("SBIN",       "NSE"),
    "SHRIRAMFIN": ("SHRIRAMFIN", "NSE"),
    "SUNPHARMA":  ("SUNPHARMA",  "NSE"),
    "TATACONSUM": ("TATACONSUM", "NSE"),
    "TATAMOTORS": ("TATAMOTORS", "NSE"),
    "TATASTEEL":  ("TATASTEEL",  "NSE"),
    "TCS":        ("TCS",        "NSE"),
    "TECHM":      ("TECHM",      "NSE"),
    "TITAN":      ("TITAN",      "NSE"),
    "TRENT":      ("TRENT",      "NSE"),
    "ULTRACEMCO": ("ULTRACEMCO", "NSE"),
    "WIPRO":      ("WIPRO",      "NSE"),
}

# Indian indices
INDIA_INDEX_TV = {
    "NSEI":     ("NIFTY",   "NSE"),
    "NSEBANK":  ("BANKNIFTY", "NSE"),
    "BSESN":    ("SENSEX",  "BSE"),
    "CNX200":   ("NIFTY200", "NSE"),
}


def fetch_tv_ohlcv(
    tv_symbol: str,
    exchange: str,
    timeframe: str,
    n_bars: int = 5000,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch OHLCV from TradingView for a single symbol + timeframe.
    Returns UTC DatetimeIndex DataFrame with open/high/low/close/volume.
    """
    try:
        from tvDatafeed import TvDatafeed
    except ImportError:
        logger.error("tvDatafeed not installed. Run: pip install git+https://github.com/rongardF/tvdatafeed.git")
        return pd.DataFrame()

    interval = _get_interval(timeframe)
    if interval is None:
        logger.error("Unsupported timeframe: %s", timeframe)
        return pd.DataFrame()

    tv = TvDatafeed()   # no-login mode

    for attempt in range(retries):
        try:
            df = tv.get_hist(
                symbol=tv_symbol,
                exchange=exchange,
                interval=interval,
                n_bars=n_bars,
            )
            if df is None or df.empty:
                return pd.DataFrame()

            # Normalise to our standard format
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            # Drop 'symbol' column if present
            df = df.drop(columns=["symbol"], errors="ignore")

            # Index is already DatetimeIndex — convert to UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index.name = "timestamp"

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = df.dropna(subset=["close"])

            for col in ["open","high","low","close","volume"]:
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info("tvDatafeed %s/%s %s: %d rows (%s → %s)",
                        tv_symbol, exchange, timeframe, len(df),
                        df.index[0].date(), df.index[-1].date())
            return df[["open","high","low","close","volume"]]

        except Exception as e:
            if attempt < retries - 1:
                logger.warning("tvDatafeed retry %d/%d for %s %s: %s",
                               attempt+1, retries, tv_symbol, timeframe, e)
                time.sleep(2 * (attempt + 1))
            else:
                logger.error("tvDatafeed failed %s %s after %d retries: %s",
                             tv_symbol, timeframe, retries, e)
    return pd.DataFrame()


def fetch_india_all(
    cache_dir: str = "cache",
    timeframes: list[str] | None = None,
    force: bool = False,
) -> dict:
    """
    Fetch all 49 NSE stocks + 4 Indian indices across all timeframes via TradingView.
    Returns {(sym, tf): DataFrame} and saves to cache.
    """
    import os
    from data.cache import cache_path, save, is_stale

    if timeframes is None:
        timeframes = ["1w","1d","1h","15m","5m"]

    all_syms = {}
    all_syms.update(NSE_STOCKS_TV)
    all_syms.update(INDIA_INDEX_TV)

    results = {}
    total = len(all_syms) * len(timeframes)
    done = 0

    for cache_sym, (tv_sym, exchange) in all_syms.items():
        for tf in timeframes:
            done += 1
            path = cache_path(cache_dir, cache_sym, tf)
            if not force and not is_stale(path, max_age_hours=6):
                logger.info("[%d/%d] Cache fresh: %s %s", done, total, cache_sym, tf)
                continue

            n_bars = TV_MAX_BARS.get(tf, 5000)
            logger.info("[%d/%d] Fetching %s (%s/%s) %s ...",
                        done, total, cache_sym, tv_sym, exchange, tf)
            df = fetch_tv_ohlcv(tv_sym, exchange, tf, n_bars=n_bars)

            if not df.empty:
                save(df, path)
                results[(cache_sym, tf)] = df
                logger.info("  Saved %d rows → %s", len(df), path)
            else:
                logger.warning("  No data: %s %s", cache_sym, tf)

            time.sleep(0.3)   # polite rate limiting

    return results
