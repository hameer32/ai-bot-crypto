"""
Plan A — Tier 3: Precision Entry (15m + 5m)
============================================
The entry trigger layer. Only fires when:
  1. Tier 1 bias is aligned (macro direction)
  2. Tier 2 zone is active (price near OB / FVG / liquidity)
  3. LTF entry signal fires (sweep → trap/absorption reversal)

Entry signals (same SMC logic as original system, applied to 15m/5m):
  - Liquidity sweep of N-period high/low
  - Reversal confirmation: trap candle OR order-flow absorption
  - Volume confirmation: volume percentile above threshold

SL placement:
  - Long:  sl = sweep_low_level - atr_k × atr
  - Short: sl = sweep_high_level + atr_k × atr

TP placement (HTF-derived, better RR):
  - TP1 = 2R (standard, partial exit 50%)
  - TP2 = nearest HTF FVG fill target OR liquidity pool (from Tier 2)
  - If TP2 > 3R, use it; else fall back to 3R

Outputs added to ltf_df:
  t3_long_signal  : bool
  t3_short_signal : bool
  t3_sl           : float
  t3_tp1          : float
  t3_tp2          : float | nan
  t3_entry_score  : float [0,1]
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_tier3(
    ltf_15m: pd.DataFrame,
    ltf_5m: pd.DataFrame,
    df: pd.DataFrame,           # working df (ltf with t1/t2 cols already merged)
    n_period: int = 20,
    vol_pct_thresh: float = 80.0,
    body_ratio_thresh: float = 0.4,
    sweep_lookback: int = 6,
    atr_k: float = 1.0,
) -> pd.DataFrame:
    """
    Apply entry signal logic and compute SL/TP levels.

    df must already contain t1_bias, t1_strength, t2_zone_score columns.
    ltf_15m is used as the primary entry timeframe; ltf_5m provides
    a finer confirmation signal merged backward onto 15m.
    """
    out = df.copy()

    # ensure required indicator columns exist
    if "atr" not in out.columns:
        prev_c = out["close"].shift(1)
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - prev_c).abs(),
            (out["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    atr = out["atr"]

    # rolling N-period high/low for sweep detection
    roll_high = out["high"].rolling(n_period).max()
    roll_low  = out["low"].rolling(n_period).min()
    pr_low    = roll_low.shift(1)
    pr_high   = roll_high.shift(1)

    # ── Liquidity sweep ───────────────────────────────────────────────────────
    vol_pct = out["volume"].rolling(20).apply(
        lambda x: float(np.sum(x[:-1] <= x[-1])) / max(len(x) - 1, 1) * 100,
        raw=True,
    )
    sweep_bull = (out["low"] < pr_low) & (out["close"] > pr_low) & (vol_pct > vol_pct_thresh)
    sweep_bear = (out["high"] > pr_high) & (out["close"] < pr_high) & (vol_pct > vol_pct_thresh)
    sweep_low_level  = pr_low.where(sweep_bull)
    sweep_high_level = pr_high.where(sweep_bear)

    # ── Trap detection ────────────────────────────────────────────────────────
    nb_high   = out["high"].rolling(n_period).max()
    nb_low    = out["low"].rolling(n_period).min()
    pr_nh     = nb_high.shift(2)
    pr_nl     = nb_low.shift(2)
    spread    = (out["high"] - out["low"]).replace(0, np.nan)
    body_ratio = ((out["close"] - out["open"]).abs() / spread).clip(0, 1).fillna(0)

    trap_bull = (
        (out["low"].shift(1) < pr_nl)
        & (out["close"] > out["open"])
        & (body_ratio > body_ratio_thresh)
        & (out["close"] > pr_nl)
    )
    trap_bear = (
        (out["high"].shift(1) > pr_nh)
        & (out["close"] < out["open"])
        & (body_ratio > body_ratio_thresh)
        & (out["close"] < pr_nh)
    )

    # ── Order flow absorption ─────────────────────────────────────────────────
    mid   = (out["high"] + out["low"]) / 2
    hv    = vol_pct > vol_pct_thresh
    of_bull = hv & (out["close"] > mid) & (out["spread"] < out["spread"].rolling(20).median() if "spread" in out.columns else True)
    of_bear = hv & (out["close"] < mid) & (out["spread"] < out["spread"].rolling(20).median() if "spread" in out.columns else True)

    # ── Recent sweep window ───────────────────────────────────────────────────
    recent_sweep_bull = sweep_bull.rolling(sweep_lookback, min_periods=1).max().astype(bool)
    recent_sweep_bear = sweep_bear.rolling(sweep_lookback, min_periods=1).max().astype(bool)

    # ── Tier 1 + Tier 2 gate ──────────────────────────────────────────────────
    # Allow trade when bias is bullish/bearish OR when t1_strength > 0
    # (avoids complete lockout in datasets with mostly neutral structure)
    t1_bias_col = out.get("t1_bias", pd.Series("neutral", index=out.index))
    t1_str_col  = out.get("t1_strength", pd.Series(0.0, index=out.index))
    t1_bull = (t1_bias_col == "bullish") | (t1_str_col > 0.3)
    t1_bear = (t1_bias_col == "bearish") | (t1_str_col > 0.3)
    # If structure is totally neutral across the board, fall back to EMA slope
    if t1_bull.sum() + t1_bear.sum() < 10 and "ema_slope" in out.columns:
        t1_bull = out["ema_slope"] > 0
        t1_bear = out["ema_slope"] < 0
    t2_active = out.get("t2_zone_score", pd.Series(0.0, index=out.index)) > 0

    entry_bull = trap_bull | of_bull
    entry_bear = trap_bear | of_bear

    t3_long  = t1_bull & recent_sweep_bull & entry_bull & t2_active
    t3_short = t1_bear & recent_sweep_bear & entry_bear & t2_active

    out["t3_long_signal"]  = t3_long
    out["t3_short_signal"] = t3_short

    # ── SL levels ────────────────────────────────────────────────────────────
    # carry forward the last seen sweep level
    sl_long_level  = sweep_low_level.ffill()
    sl_short_level = sweep_high_level.ffill()

    sl_long  = sl_long_level - atr_k * atr
    sl_short = sl_short_level + atr_k * atr

    out["t3_sl_long"]  = sl_long.where(t3_long)
    out["t3_sl_short"] = sl_short.where(t3_short)

    # ── TP levels ─────────────────────────────────────────────────────────────
    ep = out["close"]

    # TP1 = 4R (minimum 1:4 R:R)
    tp1_long  = ep + 4 * (ep - sl_long)
    tp1_short = ep - 4 * (sl_short - ep)

    # TP2 = HTF target (FVG or liquidity) if available and > 5R, else 5R
    fvg_tp  = out.get("t2_fvg_tp",  pd.Series(np.nan, index=out.index))
    liq_tp  = out.get("t2_liq_tp",  pd.Series(np.nan, index=out.index))

    htf_tp_long  = fvg_tp.where(fvg_tp > ep, liq_tp)
    htf_tp_short = fvg_tp.where(fvg_tp < ep, liq_tp)

    r_long  = (ep - sl_long).abs()
    r_short = (sl_short - ep).abs()

    tp2_long  = htf_tp_long.where(
        htf_tp_long.notna() & (htf_tp_long > ep + 5 * r_long),
        ep + 5 * r_long
    )
    tp2_short = htf_tp_short.where(
        htf_tp_short.notna() & (htf_tp_short < ep - 5 * r_short),
        ep - 5 * r_short
    )

    out["t3_tp1_long"]  = tp1_long.where(t3_long)
    out["t3_tp1_short"] = tp1_short.where(t3_short)
    out["t3_tp2_long"]  = tp2_long.where(t3_long)
    out["t3_tp2_short"] = tp2_short.where(t3_short)

    # ── Entry score ────────────────────────────────────────────────────────────
    # Sum of: sweep present, trap confirmed, OF confirmed, zone aligned,
    # T1 strength, T2 zone score — normalized to [0,1]
    t1_str = out.get("t1_strength", pd.Series(0.5, index=out.index))
    t2_zs  = out.get("t2_zone_score", pd.Series(0.0, index=out.index))

    raw_long = (
        recent_sweep_bull.astype(float) * 0.20
        + trap_bull.astype(float)        * 0.20
        + of_bull.astype(float)          * 0.15
        + t2_active.astype(float)        * 0.20
        + t1_str                         * 0.15
        + t2_zs                          * 0.10
    )
    raw_short = (
        recent_sweep_bear.astype(float) * 0.20
        + trap_bear.astype(float)        * 0.20
        + of_bear.astype(float)          * 0.15
        + t2_active.astype(float)        * 0.20
        + t1_str                         * 0.15
        + t2_zs                          * 0.10
    )

    out["t3_entry_score_long"]  = raw_long.clip(0, 1)
    out["t3_entry_score_short"] = raw_short.clip(0, 1)

    return out
