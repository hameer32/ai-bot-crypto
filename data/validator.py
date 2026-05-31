from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Binance timeframe string → pandas offset alias
TF_TO_OFFSET = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
    "3d": "3D", "1w": "1W",
}


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df.set_index("timestamp")
    return df


def fill_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    offset = TF_TO_OFFSET.get(timeframe)
    if offset is None:
        logger.warning("Unknown timeframe '%s', skipping gap fill", timeframe)
        return df

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=offset, tz="UTC")
    df = df.reindex(full_index)

    gap_mask = df["close"].isna()
    n_gaps = int(gap_mask.sum())
    if n_gaps:
        logger.info("Filled %d gap candles in %s data", n_gaps, timeframe)

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].ffill()
    df["volume"] = df["volume"].fillna(0.0)
    df["gap_filled"] = gap_mask.astype(bool)
    df.index.name = "timestamp"
    return df


def remove_outliers(df: pd.DataFrame, z_thresh: float = 5.0) -> pd.DataFrame:
    df = df.copy()
    mean = df["close"].mean()
    std = df["close"].std()
    if std == 0:
        return df

    z = (df["close"] - mean).abs() / std
    outlier_mask = z > z_thresh
    n_outliers = int(outlier_mask.sum())
    if n_outliers:
        logger.warning("Removing %d outlier candles (z > %.1f)", n_outliers, z_thresh)
        df.loc[outlier_mask, ["open", "high", "low", "close"]] = np.nan
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].ffill()

    return df


def validate(df: pd.DataFrame, timeframe: str, z_thresh: float = 5.0) -> pd.DataFrame:
    df = normalize_timestamps(df)
    df = fill_gaps(df, timeframe)
    df = remove_outliers(df, z_thresh)
    return df
