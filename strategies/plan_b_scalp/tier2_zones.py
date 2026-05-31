"""
Plan B — Tier 2: Intraday Liquidity Zones (30m + 15m)
=======================================================
Same vectorized zone lookup as Plan A Tier 2 but using 30m + 15m data.

Outputs: t2_near_ob, t2_near_fvg, t2_near_liq, t2_ob_type,
         t2_fvg_tp, t2_liq_tp, t2_zone_score
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.order_blocks import detect_order_blocks
from strategies.signals.fvg import detect_fvg, fvg_tp_target
from strategies.signals.liquidity_zones import detect_liquidity_zones, liquidity_tp
def build_tier2(
    m30_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    ob_impulse_mult: float = 1.2,
    fvg_min_atr: float = 0.2,
    proximity_atr: float = 1.5,
    swing_window: int = 4,
) -> pd.DataFrame:
    df = ltf_df.copy()

    for htf in [m30_df, m15_df]:
        if "atr" not in htf.columns:
            prev_c = htf["close"].shift(1)
            tr = pd.concat([
                htf["high"] - htf["low"],
                (htf["high"] - prev_c).abs(),
                (htf["low"]  - prev_c).abs(),
            ], axis=1).max(axis=1)
            htf["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # compute zones once on full datasets
    m30_obs  = detect_order_blocks(m30_df, impulse_atr_mult=ob_impulse_mult, lookback=50)
    m30_fvgs = detect_fvg(m30_df, min_size_atr=fvg_min_atr, lookback=100)
    m30_liq  = detect_liquidity_zones(m30_df, swing_window=swing_window, lookback=50)

    m15_obs  = detect_order_blocks(m15_df, impulse_atr_mult=ob_impulse_mult, lookback=50)
    m15_fvgs = detect_fvg(m15_df, min_size_atr=fvg_min_atr, lookback=100)
    m15_liq  = detect_liquidity_zones(m15_df, swing_window=swing_window, lookback=50)

    all_obs  = m30_obs  + m15_obs
    all_fvgs = m30_fvgs + m15_fvgs
    all_liq  = m30_liq  + m15_liq

    obs_df  = _zones_to_df(all_obs,  ("top", "bottom", "ts", "type"))
    fvgs_df = _zones_to_df(all_fvgs, ("top", "bottom", "ts", "type"))
    liq_df  = _zones_to_df(all_liq,  ("level", "ts", "type"))

    if "atr" not in df.columns:
        prev_c = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    atr    = df["atr"].values
    price  = df["close"].values
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        ts_arr = idx.tz_localize(None).values
    else:
        ts_arr = idx.values
    n      = len(df)

    near_ob   = np.zeros(n, dtype=bool)
    near_fvg  = np.zeros(n, dtype=bool)
    near_liq  = np.zeros(n, dtype=bool)
    ob_type_arr = np.full(n, "none", dtype=object)
    fvg_tp_arr  = np.full(n, np.nan)
    liq_tp_arr  = np.full(n, np.nan)

    for i in range(n):
        p   = price[i]
        a   = atr[i] if atr[i] > 0 else p * 0.001
        tol = proximity_atr * a
        ts  = ts_arr[i]

        if len(obs_df):
            past = obs_df[obs_df["ts"] < ts]
            if len(past):
                mid = (past["top"] + past["bottom"]) / 2
                dist = (mid - p).abs()
                cm = dist <= tol
                if cm.any():
                    near_ob[i] = True
                    ob_type_arr[i] = past[cm].iloc[dist[cm].argmin()]["type"]

        if len(fvgs_df):
            past = fvgs_df[fvgs_df["ts"] < ts]
            if len(past):
                mid = (past["top"] + past["bottom"]) / 2
                dist = (mid - p).abs()
                cm = dist <= tol
                if cm.any():
                    near_fvg[i] = True
                    nearest = past[cm].iloc[dist[cm].argmin()]
                    fdir = "long" if nearest["type"] == "bull" else "short"
                    tp = fvg_tp_target(all_fvgs, p, fdir)
                    fvg_tp_arr[i] = tp if tp is not None else np.nan

        if len(liq_df):
            past = liq_df[liq_df["ts"] < ts]
            if len(past):
                dist = (past["level"] - p).abs()
                cm = dist <= tol
                if cm.any():
                    near_liq[i] = True
                    nr = past[cm].iloc[dist[cm].argmin()]
                    ldir = "long" if nr["type"] == "ssl" else "short"
                    tp = liquidity_tp(all_liq, p, ldir)
                    liq_tp_arr[i] = tp if tp is not None else np.nan

    df["t2_near_ob"]    = near_ob
    df["t2_near_fvg"]   = near_fvg
    df["t2_near_liq"]   = near_liq
    df["t2_ob_type"]    = ob_type_arr
    df["t2_fvg_tp"]     = fvg_tp_arr
    df["t2_liq_tp"]     = liq_tp_arr
    df["t2_zone_score"] = (near_ob.astype(float) + near_fvg + near_liq) / 3

    return df


def _zones_to_df(zones: list[dict], keys: tuple) -> pd.DataFrame:
    if not zones:
        return pd.DataFrame(columns=list(keys))
    rows = [{k: z.get(k) for k in keys} for z in zones]
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    return df
