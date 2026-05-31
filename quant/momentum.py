from __future__ import annotations
import numpy as np
import pandas as pd


def tsm_score(df: pd.DataFrame, lookback: int = 120, short_window: int = 20) -> pd.Series:
    """
    Time-Series Momentum (Moskowitz et al. 2012).
    Return over lookback normalized to [0, 1]:
      strong positive → 1.0 (long bias)
      strong negative → 0.0 (short bias)
      neutral         → 0.5
    """
    ret = df["close"].pct_change(lookback)

    # Normalize via rolling z-score then sigmoid
    roll_mean = ret.rolling(lookback).mean()
    roll_std = ret.rolling(lookback).std().replace(0, np.nan)
    z = (ret - roll_mean) / roll_std

    # Sigmoid: maps z-score to (0, 1)
    result = 1 / (1 + np.exp(-z.clip(-5, 5)))
    result = result.fillna(0.5)
    result.name = "tsm_score"
    return result


def short_term_reversal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Short-term reversal contra-momentum signal ∈ [0, 1].
    Extreme negative short-term return → approaching 1.0 (expect bounce).
    Extreme positive short-term return → approaching 0.0 (expect pullback).
    """
    ret = df["close"].pct_change(window)
    roll_mean = ret.rolling(window * 2).mean()
    roll_std = ret.rolling(window * 2).std().replace(0, np.nan)
    z = (ret - roll_mean) / roll_std

    # Invert z: negative z (oversold) → high score
    result = 1 / (1 + np.exp(z.clip(-5, 5)))
    result = result.fillna(0.5)
    result.name = "str_score"
    return result
