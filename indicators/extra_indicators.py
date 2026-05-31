"""
Extra Indicators for Plan D Hybrid Strategy
============================================
RSI, MACD, Bollinger Bands, EMA trend, Volume-based signals.
All zero-lookahead, vectorised.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column."""
    df = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.clip(lower=1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_ob"]  = (df["rsi"] > 70).astype(float)   # overbought
    df["rsi_os"]  = (df["rsi"] < 30).astype(float)   # oversold
    df["rsi_mid"] = df["rsi"] / 100.0                  # normalised 0–1

    # RSI divergence (simplified): price making new low but RSI not
    n = period
    price_new_low = df["close"].rolling(n).min() == df["close"]
    rsi_higher    = df["rsi"] > df["rsi"].shift(n // 2)
    df["rsi_bull_div"] = (price_new_low & rsi_higher).astype(float)

    price_new_high = df["close"].rolling(n).max() == df["close"]
    rsi_lower      = df["rsi"] < df["rsi"].shift(n // 2)
    df["rsi_bear_div"] = (price_new_high & rsi_lower).astype(float)

    return df


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD, signal, histogram columns."""
    df = df.copy()
    cl  = df["close"]
    ema_fast  = cl.ewm(span=fast,   adjust=False).mean()
    ema_slow  = cl.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    hist      = macd_line - sig_line

    df["macd"]       = macd_line
    df["macd_signal"]= sig_line
    df["macd_hist"]  = hist

    df["macd_bull"]  = (hist > 0).astype(float)   # histogram above zero
    df["macd_cross_bull"] = ((hist > 0) & (hist.shift(1) <= 0)).astype(float)
    df["macd_cross_bear"] = ((hist < 0) & (hist.shift(1) >= 0)).astype(float)

    # Normalise histogram to ATR for cross-asset comparability
    atr_proxy = cl.rolling(14).std().clip(lower=1e-9)
    df["macd_hist_norm"] = hist / atr_proxy

    return df


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Band columns."""
    df = df.copy()
    cl  = df["close"]
    mid = cl.rolling(period).mean()
    std = cl.rolling(period).std()

    df["bb_upper"] = mid + std_mult * std
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - std_mult * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.clip(lower=1e-9)

    # Position: 0 = at lower band, 1 = at upper band
    band_range = (df["bb_upper"] - df["bb_lower"]).clip(lower=1e-9)
    df["bb_pct"]  = (cl - df["bb_lower"]) / band_range

    # Squeeze: current width vs rolling 20-period percentile
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)).astype(float)

    # Price at extremes
    df["bb_at_upper"] = (cl >= df["bb_upper"]).astype(float)
    df["bb_at_lower"] = (cl <= df["bb_lower"]).astype(float)

    return df


def ema_trend(
    df: pd.DataFrame,
    fast: int = 21,
    slow: int = 50,
    long: int = 200,
) -> pd.DataFrame:
    """Add EMA trend columns."""
    df = df.copy()
    cl = df["close"]

    df["ema_fast"] = cl.ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = cl.ewm(span=slow, adjust=False).mean()
    df["ema_long"] = cl.ewm(span=long, adjust=False).mean()

    # Trend alignment
    df["ema_bull_align"] = (
        (df["ema_fast"] > df["ema_slow"]) &
        (df["ema_slow"] > df["ema_long"])
    ).astype(float)
    df["ema_bear_align"] = (
        (df["ema_fast"] < df["ema_slow"]) &
        (df["ema_slow"] < df["ema_long"])
    ).astype(float)

    # Slope of slow EMA (momentum direction)
    df["ema_slope"] = (df["ema_slow"] / df["ema_slow"].shift(5).clip(lower=1e-9) - 1)

    # Price vs EMA — how extended?
    df["price_vs_ema50"] = cl / df["ema_slow"].clip(lower=1e-9) - 1

    return df


def volume_signals(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume-based confirmation signals."""
    df = df.copy()
    if "volume" not in df.columns or df["volume"].isna().all():
        df["vol_above_avg"] = 0.0
        df["vol_ratio"]     = 1.0
        return df

    vol_avg = df["volume"].rolling(period).mean().clip(lower=1)
    df["vol_ratio"]     = df["volume"] / vol_avg
    df["vol_above_avg"] = (df["vol_ratio"] > 1.5).astype(float)

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all Plan D indicators to df in one call."""
    from indicators.ichimoku import ichimoku
    df = ichimoku(df)
    df = rsi(df)
    df = macd(df)
    df = bollinger_bands(df)
    df = ema_trend(df)
    df = volume_signals(df)
    return df
