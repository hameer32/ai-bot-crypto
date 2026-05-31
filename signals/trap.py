from __future__ import annotations
import pandas as pd


def trap_detection(df: pd.DataFrame, body_ratio_thresh: float = 0.4) -> pd.DataFrame:
    df = df.copy()

    # roll_high/roll_low are rolling max/min of the current window (inclusive).
    # To detect a breakout of the PRIOR N-period range, we shift by N+1 to get
    # the N-period high/low that was valid N bars ago (before current bar contributed).
    # Simpler and correct: use the rolling max shifted by 1 extra period.
    n_bar_high = df["high"].rolling(df.attrs.get("n_period", 30)).max()
    n_bar_low  = df["low"].rolling(df.attrs.get("n_period", 30)).min()

    # Bearish trap:
    #   1. Previous bar's HIGH broke above the N-period high that existed 1 bar before it
    #   2. Current bar closes back below that level (rejection/trap)
    #   3. Current bar is bearish and strong
    prior_n_high = n_bar_high.shift(2)   # N-period high as of 2 bars ago
    prior_broke_high = df["high"].shift(1) > prior_n_high
    current_bearish = df["close"] < df["open"]
    current_strong = df["body_ratio"] > body_ratio_thresh
    closes_back_below = df["close"] < prior_n_high
    df["trap_bear"] = prior_broke_high & current_bearish & current_strong & closes_back_below

    # Bullish trap:
    #   1. Previous bar's LOW broke below the N-period low that existed 1 bar before it
    #   2. Current bar closes back above that level
    #   3. Current bar is bullish and strong
    prior_n_low = n_bar_low.shift(2)
    prior_broke_low = df["low"].shift(1) < prior_n_low
    current_bullish = df["close"] > df["open"]
    closes_back_above = df["close"] > prior_n_low
    df["trap_bull"] = prior_broke_low & current_bullish & current_strong & closes_back_above

    return df
