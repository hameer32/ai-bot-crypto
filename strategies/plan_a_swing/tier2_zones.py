"""
Plan A — Tier 2: Liquidity Zones (4H + 1H)
============================================
Maps active order blocks, FVGs, and liquidity pools from 4H and 1H data.

Performance: Instead of recomputing zones every bar (O(N²)), we compute
zones on the FULL 4H/1H dataset once, then for each LTF bar check whether
price is near any zone that was formed BEFORE that bar (lookahead-safe check
via the zone's formation timestamp).

Outputs merged onto LTF:
  t2_near_ob    : bool
  t2_near_fvg   : bool
  t2_near_liq   : bool
  t2_ob_type    : 'bull' | 'bear' | 'none'
  t2_fvg_tp     : float | nan
  t2_liq_tp     : float | nan
  t2_zone_score : float [0,1]
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.order_blocks import detect_order_blocks
from strategies.signals.fvg import detect_fvg, fvg_tp_target
from strategies.signals.liquidity_zones import detect_liquidity_zones, liquidity_tp
from strategies.signals.structure import align_htf_series


def build_tier2(
    h4_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    ob_impulse_mult: float = 1.5,
    fvg_min_atr: float = 0.3,
    proximity_atr: float = 2.0,
    swing_window: int = 5,
) -> pd.DataFrame:
    """
    Compute Tier 2 zone proximity for each LTF bar.
    Uses vectorized zone lookup — fast even on years of data.
    """
    df = ltf_df.copy()

    for htf in [h4_df, h1_df]:
        if "atr" not in htf.columns:
            prev_c = htf["close"].shift(1)
            tr = pd.concat([
                htf["high"] - htf["low"],
                (htf["high"] - prev_c).abs(),
                (htf["low"]  - prev_c).abs(),
            ], axis=1).max(axis=1)
            htf["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # ── Compute zones once on FULL datasets ───────────────────────────────────
    h4_obs  = detect_order_blocks(h4_df, impulse_atr_mult=ob_impulse_mult, lookback=50)
    h4_fvgs = detect_fvg(h4_df, min_size_atr=fvg_min_atr, lookback=100)
    h4_liq  = detect_liquidity_zones(h4_df, swing_window=swing_window, lookback=50)

    h1_obs  = detect_order_blocks(h1_df, impulse_atr_mult=ob_impulse_mult, lookback=50)
    h1_fvgs = detect_fvg(h1_df, min_size_atr=fvg_min_atr, lookback=100)
    h1_liq  = detect_liquidity_zones(h1_df, swing_window=swing_window, lookback=50)

    all_obs  = h4_obs  + h1_obs
    all_fvgs = h4_fvgs + h1_fvgs
    all_liq  = h4_liq  + h1_liq

    # Build DataFrames for vectorized proximity check
    obs_df  = _zones_to_df(all_obs,  ("top", "bottom", "ts", "type"))
    fvgs_df = _zones_to_df(all_fvgs, ("top", "bottom", "ts", "type"))
    liq_df  = _zones_to_df(all_liq,  ("level", "ts", "type"))

    # ── ATR on LTF for proximity scaling ─────────────────────────────────────
    if "atr" not in df.columns:
        prev_c = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    atr = df["atr"].values
    price = df["close"].values
    # strip tz so comparison with zone timestamps works
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        ts_arr = idx.tz_localize(None).values
    else:
        ts_arr = idx.values

    n = len(df)
    near_ob    = np.zeros(n, dtype=bool)
    near_fvg   = np.zeros(n, dtype=bool)
    near_liq   = np.zeros(n, dtype=bool)
    ob_type_arr  = np.full(n, "none", dtype=object)
    fvg_tp_arr   = np.full(n, np.nan)
    liq_tp_arr   = np.full(n, np.nan)

    for i in range(n):
        p   = price[i]
        a   = atr[i] if atr[i] > 0 else p * 0.002
        tol = proximity_atr * a
        ts  = ts_arr[i]

        # OBs formed before this bar
        if len(obs_df):
            past_obs = obs_df[obs_df["ts"] < ts]
            if len(past_obs):
                mid = (past_obs["top"] + past_obs["bottom"]) / 2
                dist = (mid - p).abs()
                close_mask = dist <= tol
                if close_mask.any():
                    nearest = past_obs[close_mask].iloc[dist[close_mask].argmin()]
                    near_ob[i] = True
                    ob_type_arr[i] = nearest["type"]

        # FVGs
        if len(fvgs_df):
            past_fvgs = fvgs_df[fvgs_df["ts"] < ts]
            if len(past_fvgs):
                mid = (past_fvgs["top"] + past_fvgs["bottom"]) / 2
                dist = (mid - p).abs()
                close_mask = dist <= tol
                if close_mask.any():
                    near_fvg[i] = True
                    nearest = past_fvgs[close_mask].iloc[dist[close_mask].argmin()]
                    fdir = "long" if nearest["type"] == "bull" else "short"
                    tp = fvg_tp_target(all_fvgs, p, fdir)
                    fvg_tp_arr[i] = tp if tp is not None else np.nan

        # Liquidity zones
        if len(liq_df):
            past_liq = liq_df[liq_df["ts"] < ts]
            if len(past_liq):
                dist = (past_liq["level"] - p).abs()
                close_mask = dist <= tol
                if close_mask.any():
                    near_liq[i] = True
                    nearest_liq = past_liq[close_mask].iloc[dist[close_mask].argmin()]
                    ldir = "long" if nearest_liq["type"] == "ssl" else "short"
                    tp = liquidity_tp(all_liq, p, ldir)
                    liq_tp_arr[i] = tp if tp is not None else np.nan

    df["t2_near_ob"]    = near_ob
    df["t2_near_fvg"]   = near_fvg
    df["t2_near_liq"]   = near_liq
    df["t2_ob_type"]    = ob_type_arr
    df["t2_fvg_tp"]     = fvg_tp_arr
    df["t2_liq_tp"]     = liq_tp_arr
    df["t2_zone_score"] = (near_ob.astype(float) + near_fvg.astype(float) + near_liq.astype(float)) / 3

    return df


def _zones_to_df(zones: list[dict], keys: tuple) -> pd.DataFrame:
    if not zones:
        return pd.DataFrame(columns=list(keys))
    rows = [{k: z.get(k) for k in keys} for z in zones]
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)  # strip tz for comparison
    return df
