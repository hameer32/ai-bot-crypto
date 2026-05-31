"""
Plan A — Tier 1: Macro Bias (Weekly + Daily)
=============================================
Determines the macro trend direction from Weekly and Daily OHLCV data.

Outputs per bar (aligned to LTF via backward merge):
  t1_bias     : 'bullish' | 'bearish' | 'neutral'
  t1_strength : float [0, 1]
  t1_sr_high  : nearest weekly/daily resistance above price
  t1_sr_low   : nearest weekly/daily support below price
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.structure import htf_structure, htf_trend, align_htf_series


def build_tier1(
    weekly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    ema_period: int = 50,
    swing_window: int = 3,
) -> pd.DataFrame:
    df = ltf_df.copy()

    # ── Weekly signals ────────────────────────────────────────────────────────
    w_sw = max(swing_window, 2)
    w_struct = htf_structure(weekly_df, swing_window=w_sw)
    w_trend  = htf_trend(weekly_df, ema_period=min(ema_period, max(len(weekly_df) // 3, 5)))

    # ── Daily signals ─────────────────────────────────────────────────────────
    d_struct = htf_structure(daily_df, swing_window=swing_window)
    d_trend  = htf_trend(daily_df, ema_period=ema_period)

    # ── Merge weekly onto daily ───────────────────────────────────────────────
    daily_merged = align_htf_series(daily_df, w_struct, "w_struct")
    daily_merged = align_htf_series(daily_merged, w_trend, "w_trend")
    daily_merged["d_struct"] = d_struct.values
    daily_merged["d_trend"]  = d_trend.values

    # ── Vectorized bias computation ───────────────────────────────────────────
    ws = daily_merged["w_struct"]
    wt = daily_merged["w_trend"]
    ds = daily_merged["d_struct"]
    dt = daily_merged["d_trend"]

    w_dir = pd.Series("neutral", index=daily_merged.index)
    w_dir[wt >  0.05] = "bullish"
    w_dir[wt < -0.05] = "bearish"

    d_dir = pd.Series("neutral", index=daily_merged.index)
    d_dir[dt >  0.05] = "bullish"
    d_dir[dt < -0.05] = "bearish"

    votes_bull = (
        (ws == "bullish").astype(int)
        + (w_dir == "bullish").astype(int)
        + (ds == "bullish").astype(int)
        + (d_dir == "bullish").astype(int)
    )
    votes_bear = (
        (ws == "bearish").astype(int)
        + (w_dir == "bearish").astype(int)
        + (ds == "bearish").astype(int)
        + (d_dir == "bearish").astype(int)
    )

    bias     = pd.Series("neutral",  index=daily_merged.index)
    strength = pd.Series(0.0,        index=daily_merged.index)

    bull_mask = votes_bull >= 3
    bear_mask = votes_bear >= 3
    mild_bull = (votes_bull == 2) & (votes_bear == 0)
    mild_bear = (votes_bear == 2) & (votes_bull == 0)

    bias[bull_mask]  = "bullish"
    bias[bear_mask]  = "bearish"
    bias[mild_bull]  = "bullish"
    bias[mild_bear]  = "bearish"

    strength[bull_mask]  = votes_bull[bull_mask] / 4
    strength[bear_mask]  = votes_bear[bear_mask] / 4
    strength[mild_bull]  = 0.5
    strength[mild_bear]  = 0.5

    daily_merged["t1_bias"]     = bias
    daily_merged["t1_strength"] = strength

    # ── Key S/R from weekly + daily swings ───────────────────────────────────
    def swing_levels(df_h, w=3):
        highs = df_h["high"].values
        lows  = df_h["low"].values
        sh, sl = [], []
        for i in range(w, len(df_h) - w):
            if highs[i] == max(highs[i - w: i + w + 1]):
                sh.append(highs[i])
            if lows[i] == min(lows[i - w: i + w + 1]):
                sl.append(lows[i])
        return sh[-5:], sl[-5:]

    w_sh, w_sl = swing_levels(weekly_df, w=w_sw)
    d_sh, d_sl = swing_levels(daily_df,  w=swing_window)
    all_highs = sorted(set(w_sh + d_sh), reverse=True)
    all_lows  = sorted(set(w_sl + d_sl))

    def nearest_above(price, levels):
        above = [l for l in levels if l > price]
        return min(above) if above else np.nan

    def nearest_below(price, levels):
        below = [l for l in levels if l < price]
        return max(below) if below else np.nan

    daily_merged["t1_sr_high"] = daily_merged["close"].apply(
        lambda p: nearest_above(p, all_highs)
    )
    daily_merged["t1_sr_low"] = daily_merged["close"].apply(
        lambda p: nearest_below(p, all_lows)
    )

    # ── Merge tier1 columns onto LTF ─────────────────────────────────────────
    for col in ["t1_bias", "t1_strength", "t1_sr_high", "t1_sr_low"]:
        df = align_htf_series(df, daily_merged[col], col)

    df["t1_bias"]     = df["t1_bias"].fillna("neutral")
    df["t1_strength"] = pd.to_numeric(df["t1_strength"], errors="coerce").fillna(0.0)

    return df
