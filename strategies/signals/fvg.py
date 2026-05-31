"""
Fair Value Gap (FVG) / Imbalance Detection
-------------------------------------------
An FVG is a 3-candle pattern where candle[i-2].high < candle[i].low  (bull FVG)
or candle[i-2].low > candle[i].high (bear FVG), leaving an uncovered gap.

These gaps act as:
  - Entry zones when price retraces into them (high-probability reversal)
  - TP magnets when price moves away and the gap is unfilled

Each FVG stored as:
  {'type': 'bull'|'bear', 'top': float, 'bottom': float,
   'ts': Timestamp, 'filled': bool, 'fill_pct': float}

Partial fill: fill_pct tracks how much of the gap has been covered.
A gap is considered fully filled when price closes inside it (fill_pct >= 1.0).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from indicators.volume import relative_volume


def detect_fvg(
    df: pd.DataFrame,
    min_size_atr: float = 0.3,
    lookback: int = 50,
) -> list[dict]:
    """
    Detect all Fair Value Gaps in df up to the last bar.

    Parameters
    ----------
    min_size_atr : minimum gap size as multiple of ATR (filters noise)
    lookback     : max gaps to keep (newest first)

    Returns list of FVG dicts, newest first, unfilled first.
    """
    if "atr" not in df.columns or "volume" not in df.columns or len(df) < 3:
        return []

    # Calculate RVOL
    rvols = relative_volume(df).values

    fvgs = []
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["atr"].values
    idx    = df.index

    for i in range(2, len(df)):
        atr = atrs[i]
        if atr <= 0:
            continue

        # Bull FVG: gap between candle[i-2].high and candle[i].low
        gap_bottom = highs[i - 2]
        gap_top    = lows[i]
        if gap_top > gap_bottom and (gap_top - gap_bottom) >= min_size_atr * atr:
            fvgs.append({
                "type":     "bull",
                "top":      gap_top,
                "bottom":   gap_bottom,
                "ts":       idx[i],
                "filled":   False,
                "fill_pct": 0.0,
                "bar_idx":  i,
                "rvol":     rvols[i],  # Volume of the gap-forming candle
            })

        # Bear FVG: gap between candle[i-2].low and candle[i].high
        gap_top2    = lows[i - 2]
        gap_bottom2 = highs[i]
        if gap_top2 > gap_bottom2 and (gap_top2 - gap_bottom2) >= min_size_atr * atr:
            fvgs.append({
                "type":     "bear",
                "top":      gap_top2,
                "bottom":   gap_bottom2,
                "ts":       idx[i],
                "filled":   False,
                "fill_pct": 0.0,
                "bar_idx":  i,
                "rvol":     rvols[i],
            })

    # compute fill status using bars after the FVG formed
    for fvg in fvgs:
        bi = fvg["bar_idx"]
        future = df.iloc[bi + 1:]
        gap_size = fvg["top"] - fvg["bottom"]
        if gap_size <= 0:
            fvg["filled"] = True
            continue
        if fvg["type"] == "bull":
            # filled when price closes below top of gap (retraces into it)
            deepest = future["low"].min() if len(future) else fvg["top"]
            penetration = fvg["top"] - max(deepest, fvg["bottom"])
            fvg["fill_pct"] = min(penetration / gap_size, 1.0)
            fvg["filled"] = fvg["fill_pct"] >= 1.0
        else:
            deepest = future["high"].max() if len(future) else fvg["bottom"]
            penetration = min(deepest, fvg["top"]) - fvg["bottom"]
            fvg["fill_pct"] = min(penetration / gap_size, 1.0)
            fvg["filled"] = fvg["fill_pct"] >= 1.0

    # return all (callers filter by timestamp)
    fvgs.sort(key=lambda f: f["ts"], reverse=True)
    return fvgs[:lookback]


def nearest_fvg(
    fvgs: list[dict],
    price: float,
    direction: str,
    proximity_atr: float,
    atr: float,
) -> dict | None:
    """
    Return the nearest unfilled FVG of matching type within proximity_atr × atr.
    Bull FVG → valid entry zone for long trades (price retracing into demand).
    Bear FVG → valid entry zone for short trades.
    """
    candidates = [f for f in fvgs if f["type"] == direction and not f["filled"]]
    best = None
    best_dist = float("inf")
    for fvg in candidates:
        mid = (fvg["top"] + fvg["bottom"]) / 2
        dist = abs(price - mid)
        if dist < proximity_atr * atr and dist < best_dist:
            best_dist = dist
            best = fvg
    return best


def fvg_tp_target(fvgs: list[dict], price: float, direction: str) -> float | None:
    """
    Find the nearest unfilled FVG in the trade direction as a TP magnet.
    Long trade → look for nearest bear FVG above price (price will fill it going up).
    Short trade → look for nearest bull FVG below price.
    """
    if direction == "long":
        candidates = [f for f in fvgs if f["type"] == "bear" and f["bottom"] > price]
        if candidates:
            return min(candidates, key=lambda f: f["bottom"])["bottom"]
    else:
        candidates = [f for f in fvgs if f["type"] == "bull" and f["top"] < price]
        if candidates:
            return max(candidates, key=lambda f: f["top"])["top"]
    return None
