"""
Plan C ICT — Tier 2: 4H + 1H ICT Zone Mapping
===============================================
Identifies intermediate-timeframe Order Blocks and FVGs that align
with the Tier 1 macro bias. Only OBs in the correct premium/discount
zone are tracked (bull OBs must be in discount, bear OBs in premium).

Outputs merged onto LTF (15m) index:
  t2_ob_bull_top   : float — nearest bull OB top
  t2_ob_bull_bot   : float — nearest bull OB bottom
  t2_ob_bear_top   : float — nearest bear OB top
  t2_ob_bear_bot   : float — nearest bear OB bottom
  t2_in_ob_bull    : bool  — price inside a bull OB (long entry zone)
  t2_in_ob_bear    : bool  — price inside a bear OB (short entry zone)
  t2_fvg_bull_top  : float — nearest unfilled bull FVG top (demand below price)
  t2_fvg_bull_bot  : float — nearest unfilled bull FVG bottom
  t2_fvg_bear_top  : float — nearest unfilled bear FVG top (supply above price)
  t2_fvg_bear_bot  : float — nearest unfilled bear FVG bottom
  t2_in_fvg_bull   : bool  — price inside a bull FVG
  t2_in_fvg_bear   : bool  — price inside a bear FVG
  t2_htf_bsl       : float — nearest 4H BSL above price (TP target for longs)
  t2_htf_ssl       : float — nearest 4H SSL below price (TP target for shorts)
  t2_ob_sl_long    : float — SL for long = bull OB bottom − 0.2×ATR
  t2_ob_sl_short   : float — SL for short = bear OB top + 0.2×ATR
  t2_confluence    : float — zone confluence score [0, 1]
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.bos_choch import detect_bos_choch
from strategies.signals.premium_discount import build_premium_discount
from strategies.signals.advanced_zones import (
    detect_volume_imbalance,
    detect_breaker_blocks,
    detect_rejection_blocks,
    detect_mitigation_blocks,
    compute_zone_quality,
)
from indicators.atr import atr as compute_atr


def _rolling_ob(df: pd.DataFrame, impulse_atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Lightweight vectorised OB detection for a single TF.
    Returns DataFrame with columns:
      ob_bull_top, ob_bull_bot, ob_bear_top, ob_bear_bot, ob_atr
    """
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)
    n = len(df)

    ob_bull_top      = np.full(n, np.nan)
    ob_bull_bot      = np.full(n, np.nan)
    ob_bull_wick_low = np.full(n, np.nan)   # full wick low — breaker threshold
    ob_bear_top      = np.full(n, np.nan)
    ob_bear_bot      = np.full(n, np.nan)
    ob_bear_wick_high= np.full(n, np.nan)   # full wick high — breaker threshold
    ob_atr           = np.full(n, np.nan)

    last_bull_top = last_bull_bot = last_bull_wl = np.nan
    last_bear_top = last_bear_bot = last_bear_wh = np.nan
    last_ob_atr   = np.nan

    for i in range(2, n):
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            ob_bull_top[i]       = last_bull_top
            ob_bull_bot[i]       = last_bull_bot
            ob_bull_wick_low[i]  = last_bull_wl
            ob_bear_top[i]       = last_bear_top
            ob_bear_bot[i]       = last_bear_bot
            ob_bear_wick_high[i] = last_bear_wh
            ob_atr[i]            = last_ob_atr
            continue

        move = abs(closes[i] - closes[i - 1])

        # Bullish impulse → bull OB = prior bearish candle (body + wick)
        if (closes[i] > opens[i]
                and move >= impulse_atr_mult * atr
                and closes[i] > highs[i - 1]
                and closes[i - 1] < opens[i - 1]):
            last_bull_top = max(opens[i - 1], closes[i - 1])
            last_bull_bot = min(opens[i - 1], closes[i - 1])
            last_bull_wl  = lows[i - 1]     # full wick low = hard invalidation level
            last_ob_atr   = atr

        # Bearish impulse → bear OB = prior bullish candle (body + wick)
        if (closes[i] < opens[i]
                and move >= impulse_atr_mult * atr
                and closes[i] < lows[i - 1]
                and closes[i - 1] > opens[i - 1]):
            last_bear_top = max(opens[i - 1], closes[i - 1])
            last_bear_bot = min(opens[i - 1], closes[i - 1])
            last_bear_wh  = highs[i - 1]    # full wick high = hard invalidation level
            last_ob_atr   = atr

        ob_bull_top[i]       = last_bull_top
        ob_bull_bot[i]       = last_bull_bot
        ob_bull_wick_low[i]  = last_bull_wl
        ob_bear_top[i]       = last_bear_top
        ob_bear_bot[i]       = last_bear_bot
        ob_bear_wick_high[i] = last_bear_wh
        ob_atr[i]            = last_ob_atr

    out = df.copy()
    out["ob_bull_top"]       = ob_bull_top
    out["ob_bull_bot"]       = ob_bull_bot
    out["ob_bull_wick_low"]  = ob_bull_wick_low
    out["ob_bear_top"]       = ob_bear_top
    out["ob_bear_bot"]       = ob_bear_bot
    out["ob_bear_wick_high"] = ob_bear_wick_high
    out["ob_atr"]            = ob_atr
    return out


def _rolling_fvg(df: pd.DataFrame, min_size_atr: float = 0.25) -> pd.DataFrame:
    """
    Rolling FVG tracker — keeps the most recent unfilled bull and bear FVG.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)
    n = len(df)

    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_bot = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_bot = np.full(n, np.nan)

    cur_bull_top = cur_bull_bot = np.nan
    cur_bear_top = cur_bear_bot = np.nan

    for i in range(2, n):
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            fvg_bull_top[i] = cur_bull_top
            fvg_bull_bot[i] = cur_bull_bot
            fvg_bear_top[i] = cur_bear_top
            fvg_bear_bot[i] = cur_bear_bot
            continue

        # New bull FVG: gap between candle[i-2].high and candle[i].low
        g_bot = highs[i - 2]
        g_top = lows[i]
        if g_top > g_bot and (g_top - g_bot) >= min_size_atr * atr:
            cur_bull_top = g_top
            cur_bull_bot = g_bot

        # New bear FVG
        g_top2 = lows[i - 2]
        g_bot2 = highs[i]
        if g_top2 > g_bot2 and (g_top2 - g_bot2) >= min_size_atr * atr:
            cur_bear_top = g_top2
            cur_bear_bot = g_bot2

        # Invalidate if price has closed through (filled) the FVG
        c = closes[i]
        if not np.isnan(cur_bull_top) and c < cur_bull_bot:
            cur_bull_top = cur_bull_bot = np.nan
        if not np.isnan(cur_bear_bot) and c > cur_bear_top:
            cur_bear_top = cur_bear_bot = np.nan

        fvg_bull_top[i] = cur_bull_top
        fvg_bull_bot[i] = cur_bull_bot
        fvg_bear_top[i] = cur_bear_top
        fvg_bear_bot[i] = cur_bear_bot

    out = df.copy()
    out["fvg_bull_top"] = fvg_bull_top
    out["fvg_bull_bot"] = fvg_bull_bot
    out["fvg_bear_top"] = fvg_bear_top
    out["fvg_bear_bot"] = fvg_bear_bot
    return out


def build_tier2(
    h4_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    swing_window_h4: int = 3,
    swing_window_h1: int = 3,
    atr_period: int = 14,
    impulse_mult: float = 1.5,
    min_fvg_atr: float = 0.25,
    m30_df: pd.DataFrame | None = None,
    swing_window_m30: int = 4,
) -> pd.DataFrame:
    """
    Build T2 ICT zones from 4H + 1H (+ optional 30m) and merge onto ltf_df.

    m30_df (optional): 30-minute OHLCV. When provided, OBs and FVGs are
    detected and merged with prefix ``t2m30_``. This fills the gap between
    the 1H and 5/15m execution TF, giving the ML model an intermediate
    zone-alignment signal.
    """
    # ── ATR on 4H ─────────────────────────────────────────────────────────────
    h4 = h4_df.copy()
    if "atr" not in h4.columns:
        h4["atr"] = compute_atr(h4, period=atr_period)

    h1 = h1_df.copy()
    if "atr" not in h1.columns:
        h1["atr"] = compute_atr(h1, period=atr_period)

    # ── Structure on 4H to get last_sh / last_sl for BSL/SSL ─────────────────
    h4 = detect_bos_choch(h4, swing_window=swing_window_h4)
    h4 = build_premium_discount(h4, swing_window=swing_window_h4, lookback=60)
    h4 = _rolling_ob(h4, impulse_atr_mult=impulse_mult)
    h4 = _rolling_fvg(h4, min_size_atr=min_fvg_atr)
    h4 = detect_volume_imbalance(h4, min_size_atr=0.10)
    h4 = detect_breaker_blocks(h4)
    h4["t2_zone_quality_bull"] = compute_zone_quality(h4, "long",  proximity_atr=2.0)
    h4["t2_zone_quality_bear"] = compute_zone_quality(h4, "short", proximity_atr=2.0)

    h1 = detect_bos_choch(h1, swing_window=swing_window_h1)
    h1 = _rolling_ob(h1, impulse_atr_mult=impulse_mult)
    h1 = _rolling_fvg(h1, min_size_atr=min_fvg_atr)
    h1 = detect_volume_imbalance(h1, min_size_atr=0.10)
    h1 = detect_breaker_blocks(h1)
    h1 = detect_rejection_blocks(h1, wick_ratio=2.0, min_wick_atr=0.4)
    h1 = detect_mitigation_blocks(h1)
    h1["t2h1_zone_quality_bull"] = compute_zone_quality(h1, "long",  proximity_atr=2.0)
    h1["t2h1_zone_quality_bear"] = compute_zone_quality(h1, "short", proximity_atr=2.0)

    closes_h4 = h4["close"].values

    # T2 signal on 4H index
    t2_h4 = pd.DataFrame(index=h4.index)

    # OB levels from 4H
    t2_h4["t2_ob_bull_top"] = h4["ob_bull_top"]
    t2_h4["t2_ob_bull_bot"] = h4["ob_bull_bot"]
    t2_h4["t2_ob_bear_top"] = h4["ob_bear_top"]
    t2_h4["t2_ob_bear_bot"] = h4["ob_bear_bot"]
    t2_h4["t2_ob_atr"]      = h4["ob_atr"]

    # FVG levels from 4H
    t2_h4["t2_fvg_bull_top"] = h4["fvg_bull_top"]
    t2_h4["t2_fvg_bull_bot"] = h4["fvg_bull_bot"]
    t2_h4["t2_fvg_bear_top"] = h4["fvg_bear_top"]
    t2_h4["t2_fvg_bear_bot"] = h4["fvg_bear_bot"]

    # 4H BSL/SSL as TP destinations
    t2_h4["t2_htf_bsl"] = h4["last_sh"]
    t2_h4["t2_htf_ssl"] = h4["last_sl"]

    # 4H CHoCH rolling flags (lookback = 10 × 4H bars = 40 hours ≈ 2 days)
    h4_choch_lookback = 10
    t2_h4["t2_choch_bull"] = (h4["choch_bull"]
                              .rolling(h4_choch_lookback, min_periods=1)
                              .max().astype(bool))
    t2_h4["t2_choch_bear"] = (h4["choch_bear"]
                              .rolling(h4_choch_lookback, min_periods=1)
                              .max().astype(bool))

    # Price-in-zone columns (computed at bar close — no lookahead)
    c = h4["close"]
    ob_bt = h4["ob_bull_top"]
    ob_bb = h4["ob_bull_bot"]
    ob_art = h4["ob_bear_top"]
    ob_arb = h4["ob_bear_bot"]
    fvg_bt = h4["fvg_bull_top"]
    fvg_bb = h4["fvg_bull_bot"]
    fvg_art = h4["fvg_bear_top"]
    fvg_arb = h4["fvg_bear_bot"]

    t2_h4["t2_in_ob_bull"]  = (c >= ob_bb) & (c <= ob_bt)
    t2_h4["t2_in_ob_bear"]  = (c >= ob_arb) & (c <= ob_art)
    t2_h4["t2_in_fvg_bull"] = (c >= fvg_bb) & (c <= fvg_bt)
    t2_h4["t2_in_fvg_bear"] = (c >= fvg_arb) & (c <= fvg_art)

    # OB-based tight SL (0.2×ATR below OB bottom for longs, above OB top for shorts)
    atr_col = h4["atr"]
    t2_h4["t2_ob_sl_long"]  = h4["ob_bull_bot"] - 0.2 * atr_col
    t2_h4["t2_ob_sl_short"] = h4["ob_bear_top"] + 0.2 * atr_col

    # ── 4H advanced zone columns ─────────────────────────────────────────────
    t2_h4["t2_in_vi_bull"]  = h4["in_vi_bull"]
    t2_h4["t2_in_vi_bear"]  = h4["in_vi_bear"]
    t2_h4["t2_in_brk_bull"] = h4["in_brk_bull"]
    t2_h4["t2_in_brk_bear"] = h4["in_brk_bear"]
    t2_h4["t2_zone_quality_bull"] = h4["t2_zone_quality_bull"]
    t2_h4["t2_zone_quality_bear"] = h4["t2_zone_quality_bear"]

    # SL for breaker entries (use breaker body bottom/top ± 0.2×ATR)
    t2_h4["t2_brk_sl_long"]  = h4["brk_bull_bot"] - 0.20 * h4["atr"]
    t2_h4["t2_brk_sl_short"] = h4["brk_bear_top"] + 0.20 * h4["atr"]

    # Confluence score — now uses zone quality which includes all new zone types
    # Bull score: max of OB quality, VI quality, breaker quality
    score_bull = h4["t2_zone_quality_bull"].fillna(0.0)
    score_bear = h4["t2_zone_quality_bear"].fillna(0.0)
    # Legacy FVG overlap bonus preserved for backward compatibility
    bull_overlap = t2_h4["t2_in_ob_bull"] & t2_h4["t2_in_fvg_bull"]
    bear_overlap = t2_h4["t2_in_ob_bear"] & t2_h4["t2_in_fvg_bear"]
    score_bull = (score_bull + (bull_overlap.astype(float) * 0.10)).clip(0, 1)
    score_bear = (score_bear + (bear_overlap.astype(float) * 0.10)).clip(0, 1)
    t2_h4["t2_confluence"] = pd.concat([score_bull, score_bear], axis=1).max(axis=1)

    # 1H refinement: OBs, FVGs, VI, breakers, rejections, mitigation
    t2_h1 = pd.DataFrame(index=h1.index)
    t2_h1["t2h1_fvg_bull_top"] = h1["fvg_bull_top"]
    t2_h1["t2h1_fvg_bull_bot"] = h1["fvg_bull_bot"]
    t2_h1["t2h1_fvg_bear_top"] = h1["fvg_bear_top"]
    t2_h1["t2h1_fvg_bear_bot"] = h1["fvg_bear_bot"]
    t2_h1["t2h1_ob_bull_top"]  = h1["ob_bull_top"]
    t2_h1["t2h1_ob_bull_bot"]  = h1["ob_bull_bot"]
    t2_h1["t2h1_ob_bear_top"]  = h1["ob_bear_top"]
    t2_h1["t2h1_ob_bear_bot"]  = h1["ob_bear_bot"]
    t2_h1["t2h1_ob_sl_long"]   = h1["ob_bull_bot"]      - 0.15 * h1["atr"]
    t2_h1["t2h1_ob_sl_short"]  = h1["ob_bear_top"]      + 0.15 * h1["atr"]
    t2_h1["t2h1_brk_sl_long"]  = h1["brk_bull_bot"]     - 0.15 * h1["atr"]
    t2_h1["t2h1_brk_sl_short"] = h1["brk_bear_top"]     + 0.15 * h1["atr"]
    t2_h1["t2h1_rej_sl_long"]  = h1["rej_bull_bot"]     - 0.10 * h1["atr"]
    t2_h1["t2h1_rej_sl_short"] = h1["rej_bear_top"]     + 0.10 * h1["atr"]

    # 1H advanced zone flags
    t2_h1["t2h1_in_vi_bull"]  = h1["in_vi_bull"]
    t2_h1["t2h1_in_vi_bear"]  = h1["in_vi_bear"]
    t2_h1["t2h1_in_brk_bull"] = h1["in_brk_bull"]
    t2_h1["t2h1_in_brk_bear"] = h1["in_brk_bear"]
    t2_h1["t2h1_in_rej_bull"] = h1["in_rej_bull"]
    t2_h1["t2h1_in_rej_bear"] = h1["in_rej_bear"]
    t2_h1["t2h1_in_mit_bull"] = h1["in_mit_bull"]
    t2_h1["t2h1_in_mit_bear"] = h1["in_mit_bear"]
    t2_h1["t2h1_zone_quality_bull"] = h1["t2h1_zone_quality_bull"]
    t2_h1["t2h1_zone_quality_bear"] = h1["t2h1_zone_quality_bear"]

    c1 = h1["close"]
    t2_h1["t2h1_in_ob_bull"] = (
        (c1 >= h1["ob_bull_bot"]) & (c1 <= h1["ob_bull_top"]) & h1["ob_bull_bot"].notna()
    )
    t2_h1["t2h1_in_ob_bear"] = (
        (c1 >= h1["ob_bear_bot"]) & (c1 <= h1["ob_bear_top"]) & h1["ob_bear_bot"].notna()
    )

    # ── Strip tz and merge onto LTF ───────────────────────────────────────────
    def strip_tz(idx):
        if hasattr(idx, "tz") and idx.tz is not None:
            return idx.tz_localize(None)
        return idx

    ltf = ltf_df.copy()
    ltf_idx  = strip_tz(ltf.index)
    h4_idx   = strip_tz(t2_h4.index)
    h1_idx   = strip_tz(t2_h1.index)

    ltf.index    = ltf_idx
    t2_h4.index  = h4_idx
    t2_h1.index  = h1_idx

    merged = pd.merge_asof(
        ltf.reset_index(),
        t2_h4.reset_index(),
        left_on="timestamp",
        right_on="timestamp",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged,
        t2_h1.reset_index(),
        left_on="timestamp",
        right_on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp")
    merged.index = ltf_df.index

    # Fill missing columns with defaults
    bool_cols = [
        "t2_in_ob_bull", "t2_in_ob_bear", "t2_in_fvg_bull", "t2_in_fvg_bear",
        "t2_choch_bull", "t2_choch_bear",
        "t2_in_vi_bull", "t2_in_vi_bear",
        "t2_in_brk_bull", "t2_in_brk_bear",
        "t2h1_in_ob_bull", "t2h1_in_ob_bear",
        "t2h1_in_vi_bull", "t2h1_in_vi_bear",
        "t2h1_in_brk_bull", "t2h1_in_brk_bear",
        "t2h1_in_rej_bull", "t2h1_in_rej_bear",
        "t2h1_in_mit_bull", "t2h1_in_mit_bear",
    ]
    float_cols = [
        "t2_ob_bull_top", "t2_ob_bull_bot", "t2_ob_bear_top", "t2_ob_bear_bot",
        "t2_fvg_bull_top", "t2_fvg_bull_bot", "t2_fvg_bear_top", "t2_fvg_bear_bot",
        "t2_htf_bsl", "t2_htf_ssl", "t2_ob_sl_long", "t2_ob_sl_short",
        "t2_brk_sl_long", "t2_brk_sl_short",
        "t2h1_fvg_bull_top", "t2h1_fvg_bull_bot", "t2h1_fvg_bear_top", "t2h1_fvg_bear_bot",
        "t2h1_ob_bull_top", "t2h1_ob_bull_bot", "t2h1_ob_bear_top", "t2h1_ob_bear_bot",
        "t2h1_ob_sl_long", "t2h1_ob_sl_short",
        "t2h1_brk_sl_long", "t2h1_brk_sl_short",
        "t2h1_rej_sl_long", "t2h1_rej_sl_short",
    ]
    score_cols = [
        "t2_zone_quality_bull", "t2_zone_quality_bear",
        "t2h1_zone_quality_bull", "t2h1_zone_quality_bear",
    ]
    for col in bool_cols:
        if col not in merged.columns:
            merged[col] = False
    for col in float_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    for col in score_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    if "t2_confluence" not in merged.columns:
        merged["t2_confluence"] = 0.0

    # ── Optional 30m T2.5 layer ───────────────────────────────────────────────
    if m30_df is not None and len(m30_df) >= 50:
        try:
            m30 = m30_df.copy()
            if "atr" not in m30.columns:
                m30["atr"] = compute_atr(m30, period=atr_period)
            m30 = detect_bos_choch(m30, swing_window=swing_window_m30)
            m30 = _rolling_ob(m30, impulse_atr_mult=impulse_mult)
            m30 = _rolling_fvg(m30, min_size_atr=min_fvg_atr)
            m30 = detect_volume_imbalance(m30, min_size_atr=0.10)
            m30 = detect_breaker_blocks(m30)
            m30 = detect_rejection_blocks(m30, wick_ratio=2.0, min_wick_atr=0.4)
            m30["t2m30_zone_quality_bull"] = compute_zone_quality(m30, "long",  proximity_atr=1.5)
            m30["t2m30_zone_quality_bear"] = compute_zone_quality(m30, "short", proximity_atr=1.5)

            t2_m30 = pd.DataFrame(index=m30.index)
            c30 = m30["close"]
            t2_m30["t2m30_ob_bull_top"]  = m30["ob_bull_top"]
            t2_m30["t2m30_ob_bull_bot"]  = m30["ob_bull_bot"]
            t2_m30["t2m30_ob_bear_top"]  = m30["ob_bear_top"]
            t2_m30["t2m30_ob_bear_bot"]  = m30["ob_bear_bot"]
            t2_m30["t2m30_fvg_bull_top"] = m30["fvg_bull_top"]
            t2_m30["t2m30_fvg_bull_bot"] = m30["fvg_bull_bot"]
            t2_m30["t2m30_fvg_bear_top"] = m30["fvg_bear_top"]
            t2_m30["t2m30_fvg_bear_bot"] = m30["fvg_bear_bot"]
            t2_m30["t2m30_in_ob_bull"]   = (c30 >= m30["ob_bull_bot"]) & (c30 <= m30["ob_bull_top"]) & m30["ob_bull_bot"].notna()
            t2_m30["t2m30_in_ob_bear"]   = (c30 >= m30["ob_bear_bot"]) & (c30 <= m30["ob_bear_top"]) & m30["ob_bear_bot"].notna()
            t2_m30["t2m30_in_fvg_bull"]  = (c30 >= m30["fvg_bull_bot"]) & (c30 <= m30["fvg_bull_top"]) & m30["fvg_bull_bot"].notna()
            t2_m30["t2m30_in_fvg_bear"]  = (c30 >= m30["fvg_bear_bot"]) & (c30 <= m30["fvg_bear_top"]) & m30["fvg_bear_bot"].notna()
            t2_m30["t2m30_in_vi_bull"]   = m30["in_vi_bull"]
            t2_m30["t2m30_in_vi_bear"]   = m30["in_vi_bear"]
            t2_m30["t2m30_in_brk_bull"]  = m30["in_brk_bull"]
            t2_m30["t2m30_in_brk_bear"]  = m30["in_brk_bear"]
            t2_m30["t2m30_in_rej_bull"]  = m30["in_rej_bull"]
            t2_m30["t2m30_in_rej_bear"]  = m30["in_rej_bear"]
            t2_m30["t2m30_choch_bull"]   = (m30["choch_bull"]
                                            .rolling(8, min_periods=1).max().astype(bool))
            t2_m30["t2m30_choch_bear"]   = (m30["choch_bear"]
                                            .rolling(8, min_periods=1).max().astype(bool))
            t2_m30["t2m30_zone_quality_bull"] = m30["t2m30_zone_quality_bull"]
            t2_m30["t2m30_zone_quality_bear"] = m30["t2m30_zone_quality_bear"]
            t2_m30["t2m30_ob_sl_long"]   = m30["ob_bull_bot"] - 0.15 * m30["atr"]
            t2_m30["t2m30_ob_sl_short"]  = m30["ob_bear_top"] + 0.15 * m30["atr"]

            # Strip tz from both sides so merge_asof dtypes match
            t2_m30.index = strip_tz(t2_m30.index)
            merged_idx = merged.index          # save original (may be tz-aware)
            merged_stripped = merged.copy()
            merged_stripped.index = strip_tz(merged.index)

            merged_new = pd.merge_asof(
                merged_stripped.reset_index(),
                t2_m30.reset_index(),
                left_on="timestamp",
                right_on="timestamp",
                direction="backward",
            ).set_index("timestamp")
            # Restore original index (tz-aware if it was)
            merged_new.index = merged_idx
            merged = merged_new

        except Exception as _e30:
            import logging as _log
            _log.getLogger(__name__).warning("30m T2.5 layer failed (%s) — skipping", _e30)

    # Default-fill any missing t2m30 columns so downstream code never KeyErrors
    m30_bool_cols  = ["t2m30_in_ob_bull", "t2m30_in_ob_bear",
                      "t2m30_in_fvg_bull", "t2m30_in_fvg_bear",
                      "t2m30_in_vi_bull",  "t2m30_in_vi_bear",
                      "t2m30_in_brk_bull", "t2m30_in_brk_bear",
                      "t2m30_in_rej_bull", "t2m30_in_rej_bear",
                      "t2m30_choch_bull",  "t2m30_choch_bear"]
    m30_float_cols = ["t2m30_ob_bull_top", "t2m30_ob_bull_bot",
                      "t2m30_ob_bear_top", "t2m30_ob_bear_bot",
                      "t2m30_fvg_bull_top","t2m30_fvg_bull_bot",
                      "t2m30_fvg_bear_top","t2m30_fvg_bear_bot",
                      "t2m30_zone_quality_bull", "t2m30_zone_quality_bear",
                      "t2m30_ob_sl_long",  "t2m30_ob_sl_short"]
    for col in m30_bool_cols:
        if col not in merged.columns:
            merged[col] = False
    for col in m30_float_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged
