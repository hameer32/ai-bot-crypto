from __future__ import annotations
import pandas as pd


def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    result = df[col].ewm(span=period, adjust=False).mean()
    result.name = f"ema_{period}"
    return result


def ema_slope(df: pd.DataFrame, period: int, lookback: int = 5) -> pd.Series:
    e = ema(df, period)
    result = (e - e.shift(lookback)) / lookback
    result.name = f"ema_slope_{period}"
    return result
