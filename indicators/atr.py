from __future__ import annotations
import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.name = "tr"
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    result = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    result.name = "atr"
    return result


def atr_percentile(atr_series: pd.Series, window: int = 100) -> pd.Series:
    def rank_last(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        return float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1) * 100

    result = atr_series.rolling(window).apply(rank_last, raw=True)
    result.name = "atr_pct"
    return result
