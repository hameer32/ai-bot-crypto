"""
Plan C ICT — Confluence Scorer
===============================
Multiplicative confidence model: every ICT condition is a gate.
If any critical gate fails → confidence collapses toward zero.

This enforces that we only trade when ALL tiers align.

Score formula:
  confidence = T1_factor × T2_factor × T3_factor × direction_bonus

Factors:
  T1_factor  = f(t1_bias aligned, t1_strength, t1_in_discount/premium, t1_in_ote)
  T2_factor  = f(t2_in_ob, t2_in_fvg, t2_confluence)
  T3_factor  = f(t3_in_killzone, t3_choch, t3_ssl/bsl_swept, t3_in_ote, t3_session_wt)
  direction  = 1.0 if T1+T2+T3 all agree, 0.5 if partial

Final confidence is in [0, 1]. Threshold 0.55+ required to open a trade.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _t1_factor(row: pd.Series, direction: str) -> float:
    """T1 factor: macro bias alignment."""
    bias = row.get("t1_bias", "neutral")
    strength = float(row.get("t1_strength", 0.0))

    if direction == "long":
        if bias != "bullish":
            return 0.0          # hard gate: macro must be bullish for longs
        zone_ok   = bool(row.get("t1_in_discount", False))
        # Hard gate: must be in discount zone for longs (ICT rule)
        if not zone_ok:
            return 0.0
        ote_ok    = bool(row.get("t1_in_ote", False))
        choch_ok  = bool(row.get("t1_choch", False))
    else:
        if bias != "bearish":
            return 0.0
        zone_ok   = bool(row.get("t1_in_premium", False))
        # Hard gate: must be in premium zone for shorts
        if not zone_ok:
            return 0.0
        ote_ok    = bool(row.get("t1_in_ote", False))
        choch_ok  = bool(row.get("t1_choch", False))

    # Strength is the base; OTE and CHoCH add bonus
    base   = 0.50 + 0.50 * strength
    bonus  = 0.10 * ote_ok + 0.10 * choch_ok
    return min(base + bonus, 1.0)


def _t2_factor(row: pd.Series, direction: str) -> float:
    """T2 factor: HTF zone quality — all zone types scored."""
    is_long = direction == "long"

    in_ob  = bool(row.get("t2_in_ob_bull"      if is_long else "t2_in_ob_bear",      False))
    in_fvg = bool(row.get("t2_in_fvg_bull"     if is_long else "t2_in_fvg_bear",     False))
    in_brk = bool(row.get("t2_in_brk_bull"     if is_long else "t2_in_brk_bear",     False)
               or row.get("t2h1_in_brk_bull"   if is_long else "t2h1_in_brk_bear",   False))
    in_rej = bool(row.get("t2h1_in_rej_bull"   if is_long else "t2h1_in_rej_bear",   False))
    in_mit = bool(row.get("t2h1_in_mit_bull"   if is_long else "t2h1_in_mit_bear",   False))
    in_vi  = bool(row.get("t2_in_vi_bull"      if is_long else "t2_in_vi_bear",      False)
               or row.get("t2h1_in_vi_bull"    if is_long else "t2h1_in_vi_bear",    False))
    in_h1  = bool(row.get("t2h1_in_ob_bull"   if is_long else "t2h1_in_ob_bear",    False))

    in_any = in_ob or in_fvg or in_brk or in_rej or in_mit or in_vi or in_h1
    if not in_any:
        return 0.35

    # Base score by zone type (OB is the gold standard)
    base = (0.60 if (in_ob or in_h1) else
            0.55 if in_brk else
            0.50 if in_rej else
            0.45 if in_mit else
            0.40 if in_fvg else
            0.35)   # VI only

    # Overlap bonuses (same logic as advanced_zones.compute_zone_quality)
    conf = float(row.get("t2_confluence", 0.0))
    bonus = (0.15 * (in_ob and in_fvg)    # OB + FVG institutional area
             + 0.10 * (in_ob and in_vi)   # OB + VI strong order flow
             + 0.15 * conf)               # T2 confluence (includes near-liq bonus)

    # Zone quality bonus from precomputed static score
    zq_4h = float(row.get("t2_zone_quality_bull"   if is_long else "t2_zone_quality_bear",   0.0))
    zq_1h = float(row.get("t2h1_zone_quality_bull" if is_long else "t2h1_zone_quality_bear", 0.0))
    zq_bonus = 0.10 * max(zq_4h, zq_1h)  # up to +0.10 from zone quality

    return min(base + bonus + zq_bonus, 1.0)


def _t3_factor(row: pd.Series, direction: str) -> float:
    """T3 factor: LTF confirmation quality."""
    in_kz = bool(row.get("t3_in_killzone", False))
    if not in_kz:
        return 0.0              # hard gate: must be in a killzone

    kz_wt   = float(row.get("t3_session_wt", 0.0))
    in_ote  = bool(row.get("t3_in_ote", False))

    if direction == "long":
        choch  = bool(row.get("t3_choch_bull", False))
        swept  = bool(row.get("t3_ssl_swept",  False))
        signal = bool(row.get("t3_signal_long", False))
    else:
        choch  = bool(row.get("t3_choch_bear", False))
        swept  = bool(row.get("t3_bsl_swept",  False))
        signal = bool(row.get("t3_signal_short", False))

    base   = kz_wt * 0.40
    bonus  = 0.25 * choch + 0.20 * swept + 0.10 * in_ote + 0.05 * signal
    return min(base + bonus, 1.0)


def compute_ict_confluence(
    row: pd.Series,
    direction: str,
) -> float:
    """
    Compute multiplicative ICT confluence score for one bar/direction.

    Parameters
    ----------
    row       : one row of the fully-merged LTF DataFrame (T1+T2+T3 columns)
    direction : 'long' | 'short'

    Returns
    -------
    float in [0, 1]
    """
    f1 = _t1_factor(row, direction)
    f2 = _t2_factor(row, direction)
    f3 = _t3_factor(row, direction)

    # Geometric mean of the three factors — any zero collapses confidence
    if f1 == 0.0 or f3 == 0.0:
        return 0.0

    confidence = (f1 * f2 * f3) ** (1.0 / 3.0)
    return float(np.clip(confidence, 0.0, 1.0))


def compute_ict_confluence_series(
    df: pd.DataFrame,
    direction: str,
) -> pd.Series:
    """
    Vectorised confluence over the entire DataFrame.
    Faster than row-by-row for backtesting.
    """
    t1 = _t1_factor_series(df, direction)
    t2 = _t2_factor_series(df, direction)
    t3 = _t3_factor_series(df, direction)

    # Geometric mean — produces 0 wherever any factor is 0
    raw = np.cbrt(t1.values * t2.values * t3.values)
    raw[(t1.values == 0) | (t3.values == 0)] = 0.0
    return pd.Series(np.clip(raw, 0.0, 1.0), index=df.index)


def _t1_factor_series(df: pd.DataFrame, direction: str) -> pd.Series:
    bias     = df.get("t1_bias",       pd.Series("neutral", index=df.index))
    strength = df.get("t1_strength",   pd.Series(0.0, index=df.index)).astype(float)
    choch    = df.get("t1_choch",      pd.Series(False, index=df.index)).astype(bool)

    if direction == "long":
        aligned  = (bias == "bullish")
        zone_ok  = df.get("t1_in_discount", pd.Series(False, index=df.index)).astype(bool)
    else:
        aligned  = (bias == "bearish")
        zone_ok  = df.get("t1_in_premium",  pd.Series(False, index=df.index)).astype(bool)

    ote_ok = df.get("t1_in_ote", pd.Series(False, index=df.index)).astype(bool)

    base  = 0.50 + 0.50 * strength
    bonus = 0.10 * ote_ok.astype(float) + 0.10 * choch.astype(float)
    result = (base + bonus).clip(0, 1)
    # Hard gate: bias AND zone must align
    result[~aligned]   = 0.0
    result[~zone_ok]   = 0.0
    return result


def _t2_factor_series(df: pd.DataFrame, direction: str) -> pd.Series:
    is_long = direction == "long"

    _b = lambda col: df.get(col, pd.Series(False, index=df.index)).astype(bool)
    _f = lambda col: df.get(col, pd.Series(0.0,  index=df.index)).astype(float)

    in_ob  = _b("t2_in_ob_bull"   if is_long else "t2_in_ob_bear")
    in_fvg = _b("t2_in_fvg_bull"  if is_long else "t2_in_fvg_bear")
    in_brk = (_b("t2_in_brk_bull"  if is_long else "t2_in_brk_bear")
            | _b("t2h1_in_brk_bull" if is_long else "t2h1_in_brk_bear"))
    in_rej = _b("t2h1_in_rej_bull" if is_long else "t2h1_in_rej_bear")
    in_mit = _b("t2h1_in_mit_bull" if is_long else "t2h1_in_mit_bear")
    in_vi  = (_b("t2_in_vi_bull"   if is_long else "t2_in_vi_bear")
            | _b("t2h1_in_vi_bull" if is_long else "t2h1_in_vi_bear"))
    in_h1  = _b("t2h1_in_ob_bull" if is_long else "t2h1_in_ob_bear")
    conf   = _f("t2_confluence")

    in_any = in_ob | in_fvg | in_brk | in_rej | in_mit | in_vi | in_h1

    # Base: zone type priority (OB is the gold standard)
    base = pd.Series(0.35, index=df.index)
    base = base.where(~(in_fvg),              0.40)
    base = base.where(~in_vi,                 0.40)
    base = base.where(~in_mit,                0.45)
    base = base.where(~in_rej,                0.50)
    base = base.where(~in_brk,                0.55)
    base = base.where(~(in_h1 | in_ob),       0.60)
    base = base.where(in_any, pd.Series(0.35, index=df.index))

    # Overlap and confluence bonuses
    bonus = (0.15 * (in_ob & in_fvg).astype(float)
             + 0.10 * (in_ob & in_vi).astype(float)
             + 0.15 * conf)

    # Zone quality add-on (max +0.10)
    zq_4h = _f("t2_zone_quality_bull"   if is_long else "t2_zone_quality_bear")
    zq_1h = _f("t2h1_zone_quality_bull" if is_long else "t2h1_zone_quality_bear")
    zq_bonus = 0.10 * zq_4h.combine(zq_1h, np.maximum).fillna(0.0)

    return (base + bonus + zq_bonus).clip(0.0, 1.0)


def _t3_factor_series(df: pd.DataFrame, direction: str) -> pd.Series:
    in_kz  = df.get("t3_in_killzone",  pd.Series(False, index=df.index)).astype(bool)
    kz_wt  = df.get("t3_session_wt",   pd.Series(0.0, index=df.index)).astype(float)
    in_ote = df.get("t3_in_ote",       pd.Series(False, index=df.index)).astype(bool)

    if direction == "long":
        choch  = df.get("t3_choch_bull",   pd.Series(False, index=df.index)).astype(bool)
        swept  = df.get("t3_ssl_swept",    pd.Series(False, index=df.index)).astype(bool)
        signal = df.get("t3_signal_long",  pd.Series(False, index=df.index)).astype(bool)
    else:
        choch  = df.get("t3_choch_bear",   pd.Series(False, index=df.index)).astype(bool)
        swept  = df.get("t3_bsl_swept",    pd.Series(False, index=df.index)).astype(bool)
        signal = df.get("t3_signal_short", pd.Series(False, index=df.index)).astype(bool)

    base   = kz_wt * 0.40
    bonus  = (0.25 * choch.astype(float)
              + 0.20 * swept.astype(float)
              + 0.10 * in_ote.astype(float)
              + 0.05 * signal.astype(float))
    result = (base + bonus).clip(0, 1)
    result[~in_kz] = 0.0            # hard gate
    return result


CONFIDENCE_THRESHOLD = 0.45  # minimum to open a trade (lower than other strategies — ICT filter handles precision)
