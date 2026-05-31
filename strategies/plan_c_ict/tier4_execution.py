"""
Plan C ICT — Tier 4: 5m Execution Entry
=========================================
Execution tier — enters the trade on the 5m chart after 15m T3 has confirmed.

ICT cascade:
  T1  Weekly/Daily  → macro bias
  T2  4H / 1H       → OB / FVG zone identification
  T3  15m           → CHoCH confirmation + sweep + OTE (killzone filter)
  T4  5m  (this)    → find the 5m OB inside the confirmed zone, tighter SL

Why separate T3 and T4?
  The 15m CHoCH tells you a structure shift happened and direction is confirmed.
  The 5m OB gives you a tighter stop-loss (smaller candle body) = better R:R.
  Executing on 5m after 15m confirmation is the standard ICT entry model.

Pre-condition:
  m5_df must already contain T3 confirmation columns merged down from the 15m
  signal DataFrame (via pd.merge_asof backward fill):
    t3_signal_long, t3_signal_short
    t3_choch_bull,  t3_choch_bear
    t3_in_killzone, t3_session,     t3_session_wt
    t3_ssl_swept,   t3_bsl_swept
    t3_in_ote
    t3_confidence
    conf_long,      conf_short      (ICT confluence from T3)
    t3_sl_long,     t3_sl_short     (15m OB SL — used as fallback)
    t3_tp2_long,    t3_tp2_short    (runner target from T2)

Outputs added to m5_df:
  t4_signal_long   bool  — T3 confirmed AND price in 5m bull OB
  t4_signal_short  bool  — T3 confirmed AND price in 5m bear OB
  t4_entry_long    float — entry price (close inside 5m OB)
  t4_entry_short   float
  t4_sl_long       float — SL below 5m OB wick − 0.1×ATR
  t4_sl_short      float — SL above 5m OB wick + 0.1×ATR
  t4_tp1_long      float — TP1 = tp1_rr × R (5m risk)
  t4_tp1_short     float
  t4_atr           float — 5m ATR at signal bar
  t4_ob_bull_top   float — active 5m bull OB top
  t4_ob_bull_bot   float — active 5m bull OB bottom
  t4_ob_bear_top   float
  t4_ob_bear_bot   float
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from indicators.atr import atr as compute_atr


def build_tier4(
    m5_df: pd.DataFrame,
    atr_period: int = 14,
    tp1_rr: float = 4.0,
    impulse_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Build T4 5m execution signals.

    m5_df must already have T3 confirmation columns merged in from 15m.
    """
    df = m5_df.copy()

    # ── ATR on 5m ────────────────────────────────────────────────────────────
    if "atr" not in df.columns:
        df["atr"] = compute_atr(df, period=atr_period)
    df["t4_atr"] = df["atr"]

    # ── 5m OB detection (same algorithm as T3, running on 5m bars) ───────────
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values
    n = len(df)

    ob_bull_top  = np.full(n, np.nan)
    ob_bull_bot  = np.full(n, np.nan)
    ob_bull_low  = np.full(n, np.nan)   # wick low — SL anchor
    ob_bear_top  = np.full(n, np.nan)
    ob_bear_bot  = np.full(n, np.nan)
    ob_bear_high = np.full(n, np.nan)  # wick high — SL anchor

    cur_bt = cur_bb = cur_bl = np.nan
    cur_art = cur_arb = cur_arh = np.nan

    for i in range(2, n):
        atr = atrs[i]

        # Mitigation: clear if price closes through the wick
        if not np.isnan(cur_bl)  and closes[i] < cur_bl:
            cur_bt = cur_bb = cur_bl = np.nan
        if not np.isnan(cur_arh) and closes[i] > cur_arh:
            cur_art = cur_arb = cur_arh = np.nan

        if not np.isnan(atr) and atr > 0:
            move = abs(closes[i] - closes[i - 1])
            # Bull OB: bearish candle[i-1] followed by strong bullish impulse[i]
            if (closes[i] > opens[i]
                    and move >= impulse_mult * atr
                    and closes[i] > highs[i - 1]
                    and closes[i - 1] < opens[i - 1]):
                cur_bt  = max(opens[i - 1], closes[i - 1])
                cur_bb  = min(opens[i - 1], closes[i - 1])
                cur_bl  = lows[i - 1]
            # Bear OB: bullish candle[i-1] followed by strong bearish impulse[i]
            if (closes[i] < opens[i]
                    and move >= impulse_mult * atr
                    and closes[i] < lows[i - 1]
                    and closes[i - 1] > opens[i - 1]):
                cur_art = max(opens[i - 1], closes[i - 1])
                cur_arb = min(opens[i - 1], closes[i - 1])
                cur_arh = highs[i - 1]

        ob_bull_top[i]  = cur_bt
        ob_bull_bot[i]  = cur_bb
        ob_bull_low[i]  = cur_bl
        ob_bear_top[i]  = cur_art
        ob_bear_bot[i]  = cur_arb
        ob_bear_high[i] = cur_arh

    df["t4_ob_bull_top"]  = ob_bull_top
    df["t4_ob_bull_bot"]  = ob_bull_bot
    df["t4_ob_bull_low"]  = ob_bull_low
    df["t4_ob_bear_top"]  = ob_bear_top
    df["t4_ob_bear_bot"]  = ob_bear_bot
    df["t4_ob_bear_high"] = ob_bear_high

    close_s = df["close"]
    atr_s   = df["atr"]

    # ── Price-in-5m-OB ───────────────────────────────────────────────────────
    in_ob_bull = (
        close_s.ge(df["t4_ob_bull_bot"])
        & close_s.le(df["t4_ob_bull_top"])
        & df["t4_ob_bull_bot"].notna()
    )
    in_ob_bear = (
        close_s.ge(df["t4_ob_bear_bot"])
        & close_s.le(df["t4_ob_bear_top"])
        & df["t4_ob_bear_bot"].notna()
    )

    # ── T3 confirmation flags (merged from 15m) ───────────────────────────────
    t3_long  = df.get("t3_signal_long",  pd.Series(False, index=df.index))
    t3_short = df.get("t3_signal_short", pd.Series(False, index=df.index))

    # Ensure boolean
    t3_long  = t3_long.astype(bool)
    t3_short = t3_short.astype(bool)

    # ── T4 final signal: T3 confirmed + price in 5m OB ───────────────────────
    df["t4_signal_long"]  = t3_long  & in_ob_bull
    df["t4_signal_short"] = t3_short & in_ob_bear

    # ── Entry = close inside OB ───────────────────────────────────────────────
    df["t4_entry_long"]  = close_s
    df["t4_entry_short"] = close_s

    # ── SL: 5m OB wick ± 0.1×ATR (tighter than 15m OB SL) ──────────────────
    sl_long_5m_raw  = df["t4_ob_bull_low"]  - 0.1 * atr_s
    sl_short_5m_raw = df["t4_ob_bear_high"] + 0.1 * atr_s

    # Minimum distance: SL at least 1.5×ATR from entry
    sl_long_min  = close_s - 1.5 * atr_s
    sl_short_min = close_s + 1.5 * atr_s

    sl_long_5m  = pd.concat([sl_long_5m_raw, sl_long_min],  axis=1).min(axis=1)
    sl_short_5m = pd.concat([sl_short_5m_raw, sl_short_min], axis=1).max(axis=1)

    # Fallback to 15m SL when no 5m OB (or use 2×ATR)
    fallback_sl_long  = close_s - 2.0 * atr_s
    fallback_sl_short = close_s + 2.0 * atr_s

    df["t4_sl_long"]  = sl_long_5m.where(in_ob_bull,  fallback_sl_long)
    df["t4_sl_short"] = sl_short_5m.where(in_ob_bear, fallback_sl_short)

    # ── TP1 = tp1_rr × R (based on 5m risk = tighter) ────────────────────────
    r_long  = (close_s - df["t4_sl_long"]).clip(lower=0.0001)
    r_short = (df["t4_sl_short"] - close_s).clip(lower=0.0001)
    df["t4_tp1_long"]  = close_s + tp1_rr * r_long
    df["t4_tp1_short"] = close_s - tp1_rr * r_short

    return df
