"""
HTF Market Structure & Trend
-----------------------------
htf_structure : 'bullish' | 'bearish' | 'neutral'
  Based on whether the most recent swing pattern is bullish (HH or HL)
  or bearish (LH or LL). Forward-fills between swing points.

htf_trend : float in [-1, 1]
  EMA slope normalized by ATR — continuous trend strength.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def htf_structure(df: pd.DataFrame, swing_window: int = 5) -> pd.Series:
    """
    Returns a forward-filled Series of 'bullish'/'bearish'/'neutral'.

    Logic: at each swing high/low, compare to the PREVIOUS swing high/low.
    Higher high OR higher low = bullish bar at that swing.
    Lower low OR lower high = bearish bar at that swing.
    Forward-fill between swings so every bar has a label.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)
    w     = min(swing_window, n // 3)
    if w < 2:
        return pd.Series("neutral", index=df.index)

    structure = pd.Series(np.nan, index=df.index, dtype=object)

    prev_sh = None
    prev_sl = None

    for i in range(w, n - w):
        window_h = highs[i - w: i + w + 1]
        window_l = lows[i - w: i + w + 1]
        is_sh = highs[i] == max(window_h)
        is_sl = lows[i]  == min(window_l)

        if is_sh and prev_sh is not None:
            if highs[i] > prev_sh:
                structure.iloc[i] = "bullish"   # HH
            else:
                structure.iloc[i] = "bearish"   # LH
            prev_sh = highs[i]
        elif is_sh:
            prev_sh = highs[i]

        if is_sl and prev_sl is not None:
            if lows[i] > prev_sl:
                structure.iloc[i] = "bullish"   # HL
            elif lows[i] < prev_sl:
                structure.iloc[i] = "bearish"   # LL
            prev_sl = lows[i]
        elif is_sl:
            prev_sl = lows[i]

    return structure.ffill().fillna("neutral")


def htf_trend(df: pd.DataFrame, ema_period: int = 50, slope_window: int = 5) -> pd.Series:
    """Returns float in [-1, 1]: EMA slope normalized by ATR."""
    ep = min(ema_period, max(len(df) // 2, 5))
    ema = df["close"].ewm(span=ep, adjust=False).mean()
    slope = (ema - ema.shift(min(slope_window, ep // 2))) / min(slope_window, ep // 2)

    atr_raw = df.get("atr", (df["high"] - df["low"]).rolling(14).mean())
    atr = atr_raw.replace(0, np.nan).ffill().fillna(df["close"] * 0.01)

    normalized = (slope / atr).clip(-3, 3) / 3
    return normalized.fillna(0)


def align_htf_series(
    ltf_df: pd.DataFrame,
    htf_series: pd.Series,
    name: str,
) -> pd.DataFrame:
    """Backward merge (lookahead-safe) of a single HTF series onto LTF df."""
    htf_reset = htf_series.rename(name).reset_index()
    ltf_reset = ltf_df.reset_index()

    merged = pd.merge_asof(
        ltf_reset.sort_values("timestamp"),
        htf_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged.set_index("timestamp")
