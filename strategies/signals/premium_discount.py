"""
Premium / Discount / OTE Zones
-------------------------------
ICT Equilibrium model:
  Range     = last swing high → last swing low
  50%       = Equilibrium
  Discount  = below 50%  → valid for LONG entries
  Premium   = above 50%  → valid for SHORT entries

OTE (Optimal Trade Entry):
  After a displacement move, price retraces into the 61.8–79% Fibonacci zone.
  This is the highest-probability reversal area inside an OB or FVG.

Outputs per bar
---------------
  range_high   : float — recent swing high defining the range
  range_low    : float — recent swing low defining the range
  equilibrium  : float — 50% of range
  pd_zone      : str   — 'premium' | 'discount' | 'equilibrium'
  ote_low      : float — 61.8% retracement level (bottom of OTE zone)
  ote_high     : float — 79.0% retracement level (top of OTE zone)
  in_ote       : bool  — current price is inside OTE zone
"""
from __future__ import annotations
import numpy as np
import pandas as pd

OTE_MIN = 0.618   # minimum retracement for OTE
OTE_MAX = 0.790   # maximum retracement for OTE


def build_premium_discount(
    df: pd.DataFrame,
    swing_window: int = 5,
    lookback: int = 100,
) -> pd.DataFrame:
    """
    Compute premium/discount zones and OTE levels for each bar.
    Uses rolling lookback window to find the defining range.
    """
    df = df.copy()
    n = len(df)

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    range_high  = np.full(n, np.nan)
    range_low   = np.full(n, np.nan)
    equilibrium = np.full(n, np.nan)
    ote_low     = np.full(n, np.nan)
    ote_high    = np.full(n, np.nan)
    pd_zone     = np.full(n, "neutral", dtype=object)

    for i in range(swing_window, n):
        start = max(0, i - lookback)
        h = highs[start:i].max()
        l = lows[start:i].min()
        eq = (h + l) / 2

        range_high[i]  = h
        range_low[i]   = l
        equilibrium[i] = eq

        c = closes[i]
        if c < eq:
            pd_zone[i] = "discount"
        elif c > eq:
            pd_zone[i] = "premium"
        else:
            pd_zone[i] = "equilibrium"

        # OTE: Fibonacci retracement of the full range (for long = from low up)
        # Long OTE: retracing from high down toward low (discount side)
        ote_low[i]  = h - OTE_MAX * (h - l)   # deeper retrace (61.8% from low perspective = 79% from top)
        ote_high[i] = h - OTE_MIN * (h - l)   # shallower retrace

    df["range_high"]  = range_high
    df["range_low"]   = range_low
    df["equilibrium"] = equilibrium
    df["pd_zone"]     = pd_zone
    df["ote_low"]     = ote_low
    df["ote_high"]    = ote_high
    df["in_ote"]      = (closes >= ote_low) & (closes <= ote_high)

    return df
