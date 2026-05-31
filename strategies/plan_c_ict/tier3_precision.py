"""
Plan C ICT — Tier 3: 15m Killzone Precision Entry
==================================================
Confirms entry using LTF signals inside ICT killzones.

Entry checklist (all must pass for full confidence):
  1. In a killzone window (London / NY AM / Silver Bullet)
  2. 15m CHoCH in the trade direction (structure flip confirmation)
  3. SSL or BSL sweep (manipulation phase complete)
  4. Price in OTE zone (61.8–79% Fibonacci retracement)
  5. 15m FVG or OB provides precision entry level

SL placement:
  - Primary: 1H OB low (bull) or 1H OB high (bear) ± 0.15×ATR   [from T2]
  - Fallback: 15m swing low/high ± 0.2×ATR

TP structure:
  - TP1 = 4R (minimum — hit in average moves)
  - TP2 = nearest 4H BSL (bull) or SSL (bear) from T2  [runner target]
  - TP3 = Daily BSL/SSL from T1  [full runner — closes at 30%]

Outputs appended to ltf_df (which already has T1+T2 columns):
  t3_signal_long   : bool  — all conditions met for long entry
  t3_signal_short  : bool  — all conditions met for short entry
  t3_in_killzone   : bool  — inside an ICT session window
  t3_session       : str   — killzone name
  t3_session_wt    : float — killzone confidence weight
  t3_choch_bull    : bool  — 15m CHoCH confirmed bullish
  t3_choch_bear    : bool  — 15m CHoCH confirmed bearish
  t3_ssl_swept     : bool  — recent SSL sweep on 15m
  t3_bsl_swept     : bool  — recent BSL sweep on 15m
  t3_in_ote        : bool  — price in 15m OTE zone
  t3_entry_long    : float — limit entry price for long (top of 15m bull FVG/OB)
  t3_entry_short   : float — limit entry for short (bottom of 15m bear FVG/OB)
  t3_sl_long       : float — stop loss for long trade
  t3_sl_short      : float — stop loss for short trade
  t3_tp1_long      : float — TP1 long (4R)
  t3_tp1_short     : float — TP1 short (4R)
  t3_tp2_long      : float — TP2 long (4H BSL from T2)
  t3_tp2_short     : float — TP2 short (4H SSL from T2)
  t3_atr           : float — 15m ATR at signal bar
  t3_confidence    : float — tier-3 sub-confidence [0, 1]
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.bos_choch import detect_bos_choch
from strategies.signals.premium_discount import build_premium_discount
from strategies.signals.killzone import build_killzone
from strategies.signals.inducement import build_inducement
from indicators.atr import atr as compute_atr


def build_tier3(
    m15_df: pd.DataFrame,
    atr_period: int = 14,
    swing_window: int = 3,
    choch_lookback: int = 20,
    sweep_lookback: int = 10,
    tp1_rr: float = 4.0,
    market: str = "global",
) -> pd.DataFrame:
    """
    Build T3 precision entry signals.
    m15_df can be 15m (for T3 confirmation before 5m execution) or 1h
    (for markets where 5m history is limited — India, Commodity, Forex).
    market: 'global' (crypto/forex/commodity) or 'india' (NSE stocks/indices)
    """
    df = m15_df.copy()

    # ── ATR ───────────────────────────────────────────────────────────────────
    if "atr" not in df.columns:
        df["atr"] = compute_atr(df, period=atr_period)

    # ── Killzone (market-aware) ───────────────────────────────────────────────
    df = build_killzone(df, market=market)

    # ── BOS/CHoCH on 15m ─────────────────────────────────────────────────────
    df = detect_bos_choch(df, swing_window=swing_window)

    # ── Premium/Discount + OTE on 15m ─────────────────────────────────────────
    df = build_premium_discount(df, swing_window=swing_window, lookback=40)

    # ── Inducement (SSL/BSL sweep) on 15m ─────────────────────────────────────
    df = build_inducement(df, swing_window=swing_window, sweep_lookback=sweep_lookback)

    # ── Rolling 15m OB (lightweight) ─────────────────────────────────────────
    # OB = last opposing candle before an impulse move.
    # We track the FULL candle (high/low) for SL accuracy, not just the body.
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values
    n = len(df)

    m15_ob_bull_top = np.full(n, np.nan)  # OB candle high (body top)
    m15_ob_bull_bot = np.full(n, np.nan)  # OB candle body bottom
    m15_ob_bull_low = np.full(n, np.nan)  # OB candle wick low (SL anchor)
    m15_ob_bear_top = np.full(n, np.nan)
    m15_ob_bear_bot = np.full(n, np.nan)
    m15_ob_bear_high = np.full(n, np.nan)  # OB candle wick high (SL anchor)
    cur_bt = cur_bb = cur_bl = cur_art = cur_arb = cur_arh = np.nan
    # Track whether the OB has been "touched" (first entry into zone)
    # Once price closes below the bull OB low, it's mitigated — clear it.
    # Once price closes above the bear OB high, it's mitigated — clear it.

    for i in range(2, n):
        atr = atrs[i]

        # Mitigation check: clear OB if price closes through the wick (fully mitigated)
        if not np.isnan(cur_bl) and closes[i] < cur_bl:
            cur_bt = cur_bb = cur_bl = np.nan
        if not np.isnan(cur_arh) and closes[i] > cur_arh:
            cur_art = cur_arb = cur_arh = np.nan

        if not np.isnan(atr) and atr > 0:
            move = abs(closes[i] - closes[i - 1])
            if (closes[i] > opens[i] and move >= 1.2 * atr
                    and closes[i] > highs[i - 1]
                    and closes[i - 1] < opens[i - 1]):
                cur_bt  = max(opens[i - 1], closes[i - 1])
                cur_bb  = min(opens[i - 1], closes[i - 1])
                cur_bl  = lows[i - 1]       # candle wick low
            if (closes[i] < opens[i] and move >= 1.2 * atr
                    and closes[i] < lows[i - 1]
                    and closes[i - 1] > opens[i - 1]):
                cur_art = max(opens[i - 1], closes[i - 1])
                cur_arb = min(opens[i - 1], closes[i - 1])
                cur_arh = highs[i - 1]      # candle wick high
        m15_ob_bull_top[i]  = cur_bt
        m15_ob_bull_bot[i]  = cur_bb
        m15_ob_bull_low[i]  = cur_bl
        m15_ob_bear_top[i]  = cur_art
        m15_ob_bear_bot[i]  = cur_arb
        m15_ob_bear_high[i] = cur_arh

    df["m15_ob_bull_top"]  = m15_ob_bull_top
    df["m15_ob_bull_bot"]  = m15_ob_bull_bot
    df["m15_ob_bull_low"]  = m15_ob_bull_low
    df["m15_ob_bear_top"]  = m15_ob_bear_top
    df["m15_ob_bear_bot"]  = m15_ob_bear_bot
    df["m15_ob_bear_high"] = m15_ob_bear_high

    # ── Recent CHoCH rolling flags ─────────────────────────────────────────────
    choch_bull_recent = df["choch_bull"].rolling(choch_lookback, min_periods=1).max().astype(bool)
    choch_bear_recent = df["choch_bear"].rolling(choch_lookback, min_periods=1).max().astype(bool)
    ssl_swept_recent  = df["ssl_swept"].rolling(sweep_lookback, min_periods=1).max().astype(bool)
    bsl_swept_recent  = df["bsl_swept"].rolling(sweep_lookback, min_periods=1).max().astype(bool)

    df["t3_in_killzone"]  = df["in_killzone"]
    df["t3_session"]      = df["session"]
    df["t3_session_wt"]   = df["session_weight"]
    df["t3_choch_bull"]   = choch_bull_recent
    df["t3_choch_bear"]   = choch_bear_recent
    df["t3_ssl_swept"]    = ssl_swept_recent
    df["t3_bsl_swept"]    = bsl_swept_recent
    df["t3_in_ote"]       = df["in_ote"]
    df["t3_atr"]          = df["atr"]

    # ── Entry: price must be INSIDE the 15m OB (pullback confirmed) ──────────
    close_s = df["close"]
    atr_s   = df["atr"]

    # Price-in-OB: the core ICT execution model — buy the pullback into OB
    in_ob_bull = (
        close_s.ge(df["m15_ob_bull_bot"])
        & close_s.le(df["m15_ob_bull_top"])
        & df["m15_ob_bull_bot"].notna()
    )
    in_ob_bear = (
        close_s.ge(df["m15_ob_bear_bot"])
        & close_s.le(df["m15_ob_bear_top"])
        & df["m15_ob_bear_bot"].notna()
    )

    df["t3_in_ob_long"]  = in_ob_bull
    df["t3_in_ob_short"] = in_ob_bear

    # Entry = current close (inside OB = at the zone, market order)
    df["t3_entry_long"]  = close_s
    df["t3_entry_short"] = close_s

    # SL: OB candle wick low/high - 0.1×ATR (SL just beyond the wick)
    # Minimum SL = 1.5×ATR to avoid noise-level stops on small OBs
    sl_long_ob_raw  = df["m15_ob_bull_low"]  - 0.1 * atr_s
    sl_short_ob_raw = df["m15_ob_bear_high"] + 0.1 * atr_s

    # Enforce minimum distance: SL at least 2.0×ATR from entry (room to breathe)
    sl_long_min  = close_s - 2.0 * atr_s
    sl_short_min = close_s + 2.0 * atr_s

    # Use whichever is further from entry (more conservative)
    sl_long_ob  = pd.concat([sl_long_ob_raw, sl_long_min], axis=1).min(axis=1)
    sl_short_ob = pd.concat([sl_short_ob_raw, sl_short_min], axis=1).max(axis=1)

    # Fallback: 1.5×ATR when no OB available
    # Standard fallback: 2×ATR below/above close
    sl_long_fallback  = sl_long_min
    sl_short_fallback = sl_short_min

    df["t3_sl_long"]  = sl_long_ob.where(in_ob_bull, sl_long_fallback)
    df["t3_sl_short"] = sl_short_ob.where(in_ob_bear, sl_short_fallback)

    # TP1 = tp1_rr × risk (configurable, default 4R)
    r_long  = (close_s - df["t3_sl_long"]).clip(lower=0.0001)
    r_short = (df["t3_sl_short"] - close_s).clip(lower=0.0001)
    df["t3_tp1_long"]  = close_s + tp1_rr * r_long
    df["t3_tp1_short"] = close_s - tp1_rr * r_short

    # TP2 = 4H BSL/SSL from T2 (will be populated after outer merge — placeholder)
    df["t3_tp2_long"]  = np.nan
    df["t3_tp2_short"] = np.nan

    # ── T3 sub-confidence score ───────────────────────────────────────────────
    kz_wt = df["t3_session_wt"].fillna(0)
    conf = (
        kz_wt * 0.25                              # killzone quality
        + choch_bull_recent.astype(float) * 0.25  # structure confirmation (long path)
        + ssl_swept_recent.astype(float)  * 0.20  # manipulation swept
        + df["t3_in_ote"].astype(float)   * 0.20  # OTE discount zone
        + (~df["m15_ob_bull_top"].isna()).astype(float) * 0.10  # has precision OB
    )
    # Short path confidence (bear side weights)
    conf_short = (
        kz_wt * 0.25
        + choch_bear_recent.astype(float) * 0.25
        + bsl_swept_recent.astype(float)  * 0.20
        + df["t3_in_ote"].astype(float)   * 0.20
        + (~df["m15_ob_bear_top"].isna()).astype(float) * 0.10
    )
    # Use the higher of the two (direction is chosen by T1 bias later)
    df["t3_confidence"] = pd.concat([conf, conf_short], axis=1).max(axis=1).clip(0, 1)

    # ── Final signal flags ────────────────────────────────────────────────────
    # Long signal: killzone + bull CHoCH (recent) + price IN 15m bull OB
    # The OB entry = pullback into demand zone after structure flip
    df["t3_signal_long"] = (
        df["t3_in_killzone"]
        & choch_bull_recent
        & in_ob_bull
    )
    # Short signal: killzone + bear CHoCH + price IN 15m bear OB
    df["t3_signal_short"] = (
        df["t3_in_killzone"]
        & choch_bear_recent
        & in_ob_bear
    )

    return df
