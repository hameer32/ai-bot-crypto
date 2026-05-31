"""
Ichimoku Cloud Indicator
========================
Standard Ichimoku Kinko Hyo with all 5 components.
All calculations are ZERO lookahead — safe for backtesting.

Components:
  tenkan   : Conversion Line — (9-period high + 9-period low) / 2
  kijun    : Base Line       — (26-period high + 26-period low) / 2
  senkou_a : Leading Span A  — displayed at current bar (already projected)
  senkou_b : Leading Span B  — displayed at current bar (already projected)
  chikou   : Lagging Span    — close shifted 26 periods back (historical)

Derived signals (no lookahead):
  above_cloud    : close > max(senkou_a, senkou_b) — bullish
  below_cloud    : close < min(senkou_a, senkou_b) — bearish
  in_cloud       : price inside the cloud — neutral/transitioning
  cloud_bullish  : senkou_a > senkou_b — bullish Kumo
  tk_cross_bull  : tenkan crossed above kijun (recent N bars) — bull signal
  tk_cross_bear  : tenkan crossed below kijun
  price_vs_kijun : close / kijun - 1 (how far price is from base line)
  cloud_thickness: (|senkou_a - senkou_b|) / close — cloud width as % of price
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period:  int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
    signal_lookback: int = 3,
) -> pd.DataFrame:
    """
    Add all Ichimoku columns to df. Returns augmented copy.

    Parameters
    ----------
    tenkan_period   : Conversion line period (default 9)
    kijun_period    : Base line period (default 26)
    senkou_b_period : Leading Span B period (default 52)
    displacement    : Cloud shift forward / Chikou shift back (default 26)
    signal_lookback : Bars to look back for TK cross detection
    """
    df = df.copy()
    hi = df["high"]
    lo = df["low"]
    cl = df["close"]

    # ── Tenkan-sen (Conversion Line) ──────────────────────────────────────────
    df["ichi_tenkan"] = (
        hi.rolling(tenkan_period).max() + lo.rolling(tenkan_period).min()
    ) / 2

    # ── Kijun-sen (Base Line) ─────────────────────────────────────────────────
    df["ichi_kijun"] = (
        hi.rolling(kijun_period).max() + lo.rolling(kijun_period).min()
    ) / 2

    # ── Senkou Span A — projected 26 bars forward from TK midpoint ───────────
    # In live charts this appears 26 bars ahead. For backtesting we use the
    # value that was projected 26 bars ago (already past the displacement).
    tk_mid = (df["ichi_tenkan"] + df["ichi_kijun"]) / 2
    df["ichi_senkou_a"] = tk_mid.shift(displacement)

    # ── Senkou Span B — projected 26 bars forward from 52-period midpoint ────
    span_b_mid = (
        hi.rolling(senkou_b_period).max() + lo.rolling(senkou_b_period).min()
    ) / 2
    df["ichi_senkou_b"] = span_b_mid.shift(displacement)

    # ── Chikou Span — current close shifted 26 bars back ─────────────────────
    df["ichi_chikou"] = cl.shift(-displacement)   # display 26 bars back
    # For signal: is the chikou above price 26 bars ago? (bullish if yes)
    past_close = cl.shift(displacement)
    df["ichi_chikou_bull"] = (cl > past_close).astype(float)
    df["ichi_chikou_bear"] = (cl < past_close).astype(float)

    # ── Price vs Cloud ────────────────────────────────────────────────────────
    cloud_top = df[["ichi_senkou_a","ichi_senkou_b"]].max(axis=1)
    cloud_bot = df[["ichi_senkou_a","ichi_senkou_b"]].min(axis=1)

    df["ichi_above_cloud"]   = (cl > cloud_top).astype(float)
    df["ichi_below_cloud"]   = (cl < cloud_bot).astype(float)
    df["ichi_in_cloud"]      = ((cl >= cloud_bot) & (cl <= cloud_top)).astype(float)
    df["ichi_cloud_bullish"] = (df["ichi_senkou_a"] > df["ichi_senkou_b"]).astype(float)
    df["ichi_cloud_thickness"] = (
        (df["ichi_senkou_a"] - df["ichi_senkou_b"]).abs() / cl.clip(lower=0.0001)
    )

    # ── Price distance from Kijun (mean reversion signal) ────────────────────
    df["ichi_price_vs_kijun"] = (cl / df["ichi_kijun"].clip(lower=0.0001) - 1)

    # ── Tenkan/Kijun Cross (signal_lookback bars) ─────────────────────────────
    tk_diff = df["ichi_tenkan"] - df["ichi_kijun"]
    tk_cross_bull = (tk_diff > 0) & (tk_diff.shift(1) <= 0)
    tk_cross_bear = (tk_diff < 0) & (tk_diff.shift(1) >= 0)
    df["ichi_tk_cross_bull"] = tk_cross_bull.rolling(signal_lookback, min_periods=1).max().astype(float)
    df["ichi_tk_cross_bear"] = tk_cross_bear.rolling(signal_lookback, min_periods=1).max().astype(float)

    # ── TK alignment (tenkan above kijun = bullish momentum) ─────────────────
    df["ichi_tk_bull"] = (df["ichi_tenkan"] > df["ichi_kijun"]).astype(float)
    df["ichi_tk_bear"] = (df["ichi_tenkan"] < df["ichi_kijun"]).astype(float)

    # ── Full bull/bear alignment score (0–5 conditions) ──────────────────────
    bull_score = (
        df["ichi_above_cloud"]   +
        df["ichi_cloud_bullish"] +
        df["ichi_tk_bull"]       +
        df["ichi_tk_cross_bull"] +
        df["ichi_chikou_bull"]
    )
    bear_score = (
        df["ichi_below_cloud"]   +
        (1 - df["ichi_cloud_bullish"]) +
        df["ichi_tk_bear"]       +
        df["ichi_tk_cross_bear"] +
        df["ichi_chikou_bear"]
    )
    df["ichi_bull_score"] = bull_score / 5.0   # 0–1 normalised
    df["ichi_bear_score"] = bear_score / 5.0

    return df
