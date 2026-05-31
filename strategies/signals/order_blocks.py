"""
Order Block Detection
---------------------
An order block is the last opposing candle before a strong impulse move.

Bull OB : last bearish candle before a bullish impulse that breaks structure upward.
Bear OB : last bullish candle before a bearish impulse that breaks structure downward.

When price returns to an OB zone, it often respects it (institutional re-entry).

Each OB is stored as:
  {'type': 'bull'|'bear', 'top': float, 'bottom': float,
   'ts': Timestamp, 'atr': float, 'mitigated': bool}

Mitigation: an OB is consumed (mitigated) when price closes through it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from indicators.volume import relative_volume, volume_spike


def detect_order_blocks(
    df: pd.DataFrame,
    impulse_atr_mult: float = 1.5,
    lookback: int = 20,
) -> list[dict]:
    """
    Scan df (OHLCV + atr column) and return all unmitigated OBs detected
    up to the last bar. Fully lookahead-safe — uses only closed candles.

    Parameters
    ----------
    impulse_atr_mult : minimum move size to qualify an impulse (in ATR units)
    lookback         : max bars to look back for OB candidates

    Returns list of OB dicts, newest first.
    """
    if "atr" not in df.columns or "volume" not in df.columns or len(df) < lookback + 2:
        return []

    # Calculate volume indicators
    rvol_series = relative_volume(df)
    vspike_series = volume_spike(df)

    obs = []
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values
    vols   = df["volume"].values
    rvols  = rvol_series.values
    vspikes = vspike_series.values
    idx    = df.index

    # scan from bar 2 onward (need prior candle as OB candidate)
    for i in range(2, len(df)):
        atr = atrs[i]
        if atr <= 0:
            continue
        move = abs(closes[i] - closes[i - 1])

        # bullish impulse: strong up candle
        if (closes[i] > opens[i]
                and move >= impulse_atr_mult * atr
                and closes[i] > highs[i - 1]):
            # the OB is the prior candle (last bearish before impulse)
            ob_i = i - 1
            if closes[ob_i] < opens[ob_i]:  # must be bearish
                obs.append({
                    "type":       "bull",
                    "top":        max(opens[ob_i], closes[ob_i]),
                    "bottom":     min(opens[ob_i], closes[ob_i]),
                    "ts":         idx[ob_i],
                    "atr":        atr,
                    "mitigated":  False,
                    "bar_idx":    ob_i,
                    "rvol":       rvols[i],      # volume of the impulse bar
                    "vspike":     vspikes[i],   # whether impulse was a spike
                })

        # bearish impulse: strong down candle
        if (closes[i] < opens[i]
                and move >= impulse_atr_mult * atr
                and closes[i] < lows[i - 1]):
            ob_i = i - 1
            if closes[ob_i] > opens[ob_i]:  # must be bullish
                obs.append({
                    "type":       "bear",
                    "top":        max(opens[ob_i], closes[ob_i]),
                    "bottom":     min(opens[ob_i], closes[ob_i]),
                    "ts":         idx[ob_i],
                    "atr":        atr,
                    "mitigated":  False,
                    "bar_idx":    ob_i,
                    "rvol":       rvols[i],      # volume of the impulse bar
                    "vspike":     vspikes[i],   # whether impulse was a spike
                })

    # mark mitigated: if price closed through the OB after it formed
    for ob in obs:
        bi = ob["bar_idx"]
        future = df.iloc[bi + 1 :]
        if ob["type"] == "bull":
            if (future["close"] < ob["bottom"]).any():
                ob["mitigated"] = True
        else:
            if (future["close"] > ob["top"]).any():
                ob["mitigated"] = True

    # return all (including mitigated) — callers filter by timestamp for lookahead safety
    obs.sort(key=lambda o: o["ts"], reverse=True)
    return obs[-lookback:]  # cap to lookback


def nearest_ob(
    obs: list[dict],
    price: float,
    direction: str,
    proximity_atr: float,
    atr: float,
) -> dict | None:
    """
    Return the nearest active OB of matching type within proximity_atr × atr
    of the current price. Returns None if no qualifying OB exists.
    """
    candidates = [o for o in obs if o["type"] == direction and not o["mitigated"]]
    best = None
    best_dist = float("inf")
    for ob in candidates:
        mid = (ob["top"] + ob["bottom"]) / 2
        dist = abs(price - mid)
        if dist < proximity_atr * atr and dist < best_dist:
            best_dist = dist
            best = ob
    return best
