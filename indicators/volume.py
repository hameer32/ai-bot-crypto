from __future__ import annotations
import numpy as np
import pandas as pd


def volume_percentile(df: pd.DataFrame, window: int = 20) -> pd.Series:
    def rank_last(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        return float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1) * 100

    result = df["volume"].rolling(window).apply(rank_last, raw=True)
    result.name = "vol_pct"
    return result


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    RVOL: Current volume / Average volume of last N bars.
    > 2.0 indicates high institutional interest.
    """
    avg_vol = df["volume"].rolling(window).mean()
    rvol = df["volume"] / avg_vol
    rvol.name = "rvol"
    return rvol


def spread_percentile(df: pd.DataFrame, window: int = 20) -> pd.Series:
    spread = df["high"] - df["low"]
    def rank_last(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        return float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1) * 100

    result = spread.rolling(window).apply(rank_last, raw=True)
    result.name = "spread_pct"
    return result


def volume_spike(df: pd.DataFrame, threshold_pct: float = 80.0, window: int = 20) -> pd.Series:
    pct = volume_percentile(df, window)
    result = pct > threshold_pct
    result.name = "vol_spike"
    return result
