"""
Liquidity Zone Detection
-------------------------
Liquidity pools sit above swing highs (buy-side liquidity) and below swing lows
(sell-side liquidity). These are areas where stop-losses cluster — smart money
hunts these before reversing.

Zone types:
  'bsl' (buy-side liquidity)  : above a cluster of swing highs  → short trigger zone
  'ssl' (sell-side liquidity) : below a cluster of swing lows   → long trigger zone

Each zone:
  {'type': 'bsl'|'ssl', 'level': float, 'strength': int,
   'ts': Timestamp, 'swept': bool}

Strength = number of swing points within clustering tolerance (higher = more liquidity).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_liquidity_zones(
    df: pd.DataFrame,
    swing_window: int = 5,
    cluster_atr: float = 0.5,
    min_strength: int = 2,
    lookback: int = 30,
) -> list[dict]:
    """
    Identify buy-side and sell-side liquidity zones.

    Parameters
    ----------
    swing_window  : bars each side for swing point detection
    cluster_atr   : tolerance to cluster nearby swings (in ATR units)
    min_strength  : minimum clustered swing count to qualify as a zone
    lookback      : max zones to return (newest first)
    """
    if "atr" not in df.columns or len(df) < swing_window * 2 + 1:
        return []

    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values
    closes = df["close"].values
    idx    = df.index
    n      = len(df)
    w      = swing_window

    # Identify swing highs and lows
    sh_levels, sh_times = [], []
    sl_levels, sl_times = [], []

    for i in range(w, n - w):
        if highs[i] == max(highs[i - w: i + w + 1]):
            sh_levels.append(highs[i])
            sh_times.append(idx[i])
        if lows[i] == min(lows[i - w: i + w + 1]):
            sl_levels.append(lows[i])
            sl_times.append(idx[i])

    atr_now = float(np.nanmean(atrs[-20:])) if len(atrs) >= 20 else float(atrs[-1])

    def cluster(levels, times, zone_type):
        if not levels:
            return []
        zones = []
        used = [False] * len(levels)
        for i in range(len(levels)):
            if used[i]:
                continue
            group_l = [levels[i]]
            group_t = [times[i]]
            used[i] = True
            for j in range(i + 1, len(levels)):
                if not used[j] and abs(levels[j] - levels[i]) <= cluster_atr * atr_now:
                    group_l.append(levels[j])
                    group_t.append(times[j])
                    used[j] = True
            if len(group_l) >= min_strength:
                zones.append({
                    "type":     zone_type,
                    "level":    float(np.mean(group_l)),
                    "strength": len(group_l),
                    "ts":       max(group_t),
                    "swept":    False,
                })
        return zones

    bsl_zones = cluster(sh_levels, sh_times, "bsl")  # buy-side above highs
    ssl_zones = cluster(sl_levels, sl_times, "ssl")  # sell-side below lows

    all_zones = bsl_zones + ssl_zones

    # mark swept: zone swept when price closes through it
    last_close = closes[-1]
    for z in all_zones:
        if z["type"] == "bsl" and last_close > z["level"]:
            z["swept"] = True
        if z["type"] == "ssl" and last_close < z["level"]:
            z["swept"] = True

    active = [z for z in all_zones if not z["swept"]]
    active.sort(key=lambda z: z["ts"], reverse=True)
    return active[:lookback]


def nearest_zone(
    zones: list[dict],
    price: float,
    zone_type: str,
    proximity_atr: float,
    atr: float,
) -> dict | None:
    """Return closest unswept zone of given type within proximity range."""
    candidates = [z for z in zones if z["type"] == zone_type and not z["swept"]]
    best, best_dist = None, float("inf")
    for z in candidates:
        dist = abs(price - z["level"])
        if dist < proximity_atr * atr and dist < best_dist:
            best_dist = dist
            best = z
    return best


def liquidity_tp(zones: list[dict], price: float, direction: str) -> float | None:
    """
    Find the nearest liquidity pool in the trade direction as TP target.
    Long → nearest BSL above price (buy stops = TP target).
    Short → nearest SSL below price.
    """
    if direction == "long":
        candidates = [z for z in zones if z["type"] == "bsl" and z["level"] > price]
        if candidates:
            return min(candidates, key=lambda z: z["level"])["level"]
    else:
        candidates = [z for z in zones if z["type"] == "ssl" and z["level"] < price]
        if candidates:
            return max(candidates, key=lambda z: z["level"])["level"]
    return None
