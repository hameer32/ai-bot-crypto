"""
Live Data Feed — Binance REST API
===================================
Fetches latest OHLCV bars for each symbol/timeframe from Binance public API.
No authentication required.

Usage:
    from live.data_feed import fetch_live_bars, update_all_data
"""
from __future__ import annotations
import logging
import time
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com/api/v3"

# Our TF names → Binance interval strings
TF_TO_BINANCE = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
    "1w":  "1w",
}


def fetch_live_bars(
    symbol: str,
    timeframe: str,
    limit: int = 200,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch latest `limit` OHLCV bars from Binance for symbol/timeframe.
    Returns UTC DatetimeIndex DataFrame matching our cache format.
    """
    interval = TF_TO_BINANCE.get(timeframe)
    if not interval:
        logger.error("Unknown timeframe: %s", timeframe)
        return pd.DataFrame()

    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                return pd.DataFrame()

            df = pd.DataFrame(raw, columns=[
                "timestamp","open","high","low","close","volume",
                "close_time","quote_volume","n_trades",
                "taker_buy_base","taker_buy_quote","ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df[["open","high","low","close","volume"]].copy()
            df.index.name = "timestamp"

            # Drop the last (still-forming) candle
            if len(df) > 1:
                df = df.iloc[:-1]

            return df

        except Exception as e:
            if attempt < retries - 1:
                logger.warning("Binance retry %d/%d for %s %s: %s",
                               attempt+1, retries, symbol, timeframe, e)
                time.sleep(1)
            else:
                logger.error("Binance failed %s %s: %s", symbol, timeframe, e)

    return pd.DataFrame()


def update_all_data(
    symbols: list[str],
    timeframes: list[str],
    bars_per_tf: dict[str, int],
    existing_data: dict | None = None,
) -> dict:
    """
    Fetch/update all (symbol, timeframe) pairs from Binance.
    Merges with existing_data if provided (append new bars, keep history).

    Returns: {(symbol, timeframe): pd.DataFrame}
    """
    all_data = dict(existing_data) if existing_data else {}
    total = len(symbols) * len(timeframes)
    done  = 0

    for sym in symbols:
        for tf in timeframes:
            done += 1
            limit = bars_per_tf.get(tf, 200)

            new_df = fetch_live_bars(sym, tf, limit=limit)
            if new_df.empty:
                # Keep existing if available
                continue

            existing = all_data.get((sym, tf))
            if existing is not None and not existing.empty:
                # Merge: existing older bars + new recent bars
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                all_data[(sym, tf)] = combined
            else:
                all_data[(sym, tf)] = new_df

            time.sleep(0.05)   # gentle rate limiting

    logger.info("Updated %d symbol/TF pairs from Binance", total)
    return all_data


def get_current_price(symbol: str) -> float:
    """Get the latest mid-price for a symbol."""
    try:
        resp = requests.get(f"{BINANCE_BASE}/ticker/price",
                            params={"symbol": symbol}, timeout=5)
        return float(resp.json()["price"])
    except Exception:
        return 0.0
