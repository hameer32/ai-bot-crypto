"""
Plan B — Tier 1: Intraday Bias (4H + 1H)
==========================================
Vectorized bias from 4H + 1H structure and trend.
Outputs: t1_bias, t1_strength, t1_sr_high, t1_sr_low merged onto LTF.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.structure import htf_structure, htf_trend, align_htf_series


def build_tier1(
    h4_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    ema_period: int = 50,
    swing_window: int = 5,
) -> pd.DataFrame:
    df = ltf_df.copy()

    h4_struct = htf_structure(h4_df, swing_window=swing_window)
    h4_tr     = htf_trend(h4_df, ema_period=ema_period)
    h1_struct = htf_structure(h1_df, swing_window=swing_window)
    h1_tr     = htf_trend(h1_df, ema_period=min(ema_period, 20))

    # merge h4 onto h1
    h1_merged = align_htf_series(h1_df, h4_struct, "h4_struct")
    h1_merged = align_htf_series(h1_merged, h4_tr, "h4_trend")
    h1_merged["h1_struct"] = h1_struct.values
    h1_merged["h1_trend"]  = h1_tr.values

    h4s = h1_merged["h4_struct"]
    h4t = h1_merged["h4_trend"]
    h1s = h1_merged["h1_struct"]
    h1t = h1_merged["h1_trend"]

    h4_dir = pd.Series("neutral", index=h1_merged.index)
    h4_dir[h4t >  0.05] = "bullish"
    h4_dir[h4t < -0.05] = "bearish"

    h1_dir = pd.Series("neutral", index=h1_merged.index)
    h1_dir[h1t >  0.05] = "bullish"
    h1_dir[h1t < -0.05] = "bearish"

    votes_bull = (
        (h4s == "bullish").astype(int)
        + (h4_dir == "bullish").astype(int)
        + (h1s == "bullish").astype(int)
        + (h1_dir == "bullish").astype(int)
    )
    votes_bear = (
        (h4s == "bearish").astype(int)
        + (h4_dir == "bearish").astype(int)
        + (h1s == "bearish").astype(int)
        + (h1_dir == "bearish").astype(int)
    )

    bias     = pd.Series("neutral", index=h1_merged.index)
    strength = pd.Series(0.0,       index=h1_merged.index)

    bull_mask = votes_bull >= 3
    bear_mask = votes_bear >= 3
    mild_bull = (votes_bull == 2) & (votes_bear == 0)
    mild_bear = (votes_bear == 2) & (votes_bull == 0)

    bias[bull_mask]     = "bullish"
    bias[bear_mask]     = "bearish"
    bias[mild_bull]     = "bullish"
    bias[mild_bear]     = "bearish"
    strength[bull_mask] = votes_bull[bull_mask] / 4
    strength[bear_mask] = votes_bear[bear_mask] / 4
    strength[mild_bull] = 0.5
    strength[mild_bear] = 0.5

    h1_merged["t1_bias"]     = bias
    h1_merged["t1_strength"] = strength

    # S/R levels
    def swing_levels(df_h, w=5):
        highs = df_h["high"].values
        lows  = df_h["low"].values
        sh, sl = [], []
        for i in range(w, len(df_h) - w):
            if highs[i] == max(highs[i - w: i + w + 1]):
                sh.append(highs[i])
            if lows[i] == min(lows[i - w: i + w + 1]):
                sl.append(lows[i])
        return sh[-8:], sl[-8:]

    h4_sh, h4_sl = swing_levels(h4_df, w=swing_window)
    h1_sh, h1_sl = swing_levels(h1_df, w=swing_window)
    all_highs = sorted(set(h4_sh + h1_sh), reverse=True)
    all_lows  = sorted(set(h4_sl + h1_sl))

    def nearest_above(price, levels):
        above = [l for l in levels if l > price]
        return min(above) if above else np.nan

    def nearest_below(price, levels):
        below = [l for l in levels if l < price]
        return max(below) if below else np.nan

    h1_merged["t1_sr_high"] = h1_merged["close"].apply(lambda p: nearest_above(p, all_highs))
    h1_merged["t1_sr_low"]  = h1_merged["close"].apply(lambda p: nearest_below(p, all_lows))

    for col in ["t1_bias", "t1_strength", "t1_sr_high", "t1_sr_low"]:
        df = align_htf_series(df, h1_merged[col], col)

    df["t1_bias"]     = df["t1_bias"].fillna("neutral")
    df["t1_strength"] = pd.to_numeric(df["t1_strength"], errors="coerce").fillna(0.0)

    return df
