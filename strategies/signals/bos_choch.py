"""
BOS (Break of Structure) and CHoCH (Change of Character)
---------------------------------------------------------
ICT Definitions:
  BOS  : Price closes beyond a prior confirmed swing high/low — trend continuation.
  CHoCH: First BOS that REVERSES the prevailing structure — marks potential trend flip.

Swing detection is lookahead-safe: a swing high at bar i is only confirmed once
swing_window bars have closed after it (shift forward by window).

Outputs per bar
---------------
  last_sh       : float — most recent confirmed swing high price
  last_sl       : float — most recent confirmed swing low price
  bos_bull      : bool  — this bar broke the last swing high (bullish BOS)
  bos_bear      : bool  — this bar broke the last swing low (bearish BOS)
  choch_bull    : bool  — first bullish BOS after a bearish structure
  choch_bear    : bool  — first bearish BOS after a bullish structure
  structure     : str   — 'bullish' | 'bearish' | 'neutral' (forward-filled)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    Return (swing_high_levels, swing_low_levels) series.
    A swing high at bar i: df.high[i] is the maximum of [i-w .. i+w].
    Lookahead-safe: confirmed values shifted forward by window bars.
    """
    highs = df["high"]
    lows  = df["low"]
    w     = window

    roll_max = highs.rolling(window=2 * w + 1, center=True, min_periods=w + 1).max()
    roll_min = lows.rolling( window=2 * w + 1, center=True, min_periods=w + 1).min()

    is_sh = (highs == roll_max)
    is_sl = (lows  == roll_min)

    # Shift forward so swing is only "known" after window confirmation bars
    sh_confirmed = is_sh.shift(w).fillna(False)
    sl_confirmed = is_sl.shift(w).fillna(False)

    sh_levels = highs.where(sh_confirmed).ffill()
    sl_levels = lows.where(sl_confirmed).ffill()

    return sh_levels, sl_levels


def detect_bos_choch(
    df: pd.DataFrame,
    swing_window: int = 5,
) -> pd.DataFrame:
    """
    Attach BOS / CHoCH / structure columns to df. Returns augmented copy.
    """
    df = df.copy()
    closes = df["close"]

    sh_levels, sl_levels = detect_swings(df, swing_window)
    df["last_sh"] = sh_levels
    df["last_sl"] = sl_levels

    # BOS: close exceeds the last confirmed swing level
    bos_bull = (closes > sh_levels) & sh_levels.notna()
    bos_bear = (closes < sl_levels) & sl_levels.notna()

    # Track structure state and detect CHoCH
    structure  = pd.Series("neutral", index=df.index)
    choch_bull = pd.Series(False, index=df.index)
    choch_bear = pd.Series(False, index=df.index)

    current_structure = "neutral"
    for i in range(len(df)):
        if bos_bull.iloc[i]:
            if current_structure == "bearish":
                choch_bull.iloc[i] = True   # first bullish BOS → CHoCH
            current_structure = "bullish"
            structure.iloc[i] = "bullish"
        elif bos_bear.iloc[i]:
            if current_structure == "bullish":
                choch_bear.iloc[i] = True   # first bearish BOS → CHoCH
            current_structure = "bearish"
            structure.iloc[i] = "bearish"
        else:
            structure.iloc[i] = current_structure

    df["bos_bull"]   = bos_bull
    df["bos_bear"]   = bos_bear
    df["choch_bull"] = choch_bull
    df["choch_bear"] = choch_bear
    df["structure"]  = structure.replace("neutral", None).ffill().fillna("neutral")

    return df
