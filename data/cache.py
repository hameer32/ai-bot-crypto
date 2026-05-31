from __future__ import annotations
import os
import time
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def cache_path(cache_dir: str, symbol: str, timeframe: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    return os.path.join(cache_dir, f"{safe_symbol}_{timeframe}.parquet")


def save(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, compression="snappy")
    logger.info("Saved %d rows to %s", len(df), path)


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def is_stale(path: str, max_age_hours: int = 24) -> bool:
    if not os.path.exists(path):
        return True
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds > max_age_hours * 3600


def load_or_fetch(
    path: str,
    fetch_fn,
    max_age_hours: int = 24,
) -> pd.DataFrame:
    if not is_stale(path, max_age_hours):
        logger.info("Loading from cache: %s", path)
        return load(path)
    logger.info("Cache stale or missing, fetching: %s", path)
    df = fetch_fn()
    save(df, path)
    return df
