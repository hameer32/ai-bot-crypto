from __future__ import annotations
import pandas as pd


def candle_spread(df: pd.DataFrame) -> pd.Series:
    result = df["high"] - df["low"]
    result.name = "spread"
    return result


def body_size(df: pd.DataFrame) -> pd.Series:
    result = (df["close"] - df["open"]).abs()
    result.name = "body"
    return result


def body_ratio(df: pd.DataFrame) -> pd.Series:
    spread = candle_spread(df).replace(0, float("nan"))
    result = (body_size(df) / spread).clip(0, 1).fillna(0)
    result.name = "body_ratio"
    return result


def is_strong_candle(df: pd.DataFrame, body_ratio_thresh: float = 0.6) -> pd.Series:
    result = body_ratio(df) > body_ratio_thresh
    result.name = "strong_candle"
    return result
