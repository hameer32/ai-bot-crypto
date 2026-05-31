from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_high(df: pd.DataFrame, n: int) -> pd.Series:
    result = df["high"].rolling(n).max()
    result.name = f"roll_high_{n}"
    return result


def rolling_low(df: pd.DataFrame, n: int) -> pd.Series:
    result = df["low"].rolling(n).min()
    result.name = f"roll_low_{n}"
    return result


def swing_highs(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.Series:
    highs = df["high"]
    n = len(highs)
    result = pd.Series(False, index=df.index, name="swing_high")
    for i in range(left, n - right):
        window = highs.iloc[i - left: i + right + 1]
        if highs.iloc[i] == window.max():
            result.iloc[i] = True
    return result


def swing_lows(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.Series:
    lows = df["low"]
    n = len(lows)
    result = pd.Series(False, index=df.index, name="swing_low")
    for i in range(left, n - right):
        window = lows.iloc[i - left: i + right + 1]
        if lows.iloc[i] == window.min():
            result.iloc[i] = True
    return result
