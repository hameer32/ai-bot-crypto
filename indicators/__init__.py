from __future__ import annotations
import pandas as pd

from .atr import atr, atr_percentile
from .structure import rolling_high, rolling_low, swing_highs, swing_lows
from .volume import volume_percentile, spread_percentile
from .trend import ema, ema_slope
from .candle import candle_spread, body_ratio


def build_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    r = cfg.rules

    df["atr"] = atr(df, r.atr_period)
    df["atr_pct"] = atr_percentile(df["atr"], window=100)
    df["roll_high"] = rolling_high(df, r.n_period)
    df["roll_low"] = rolling_low(df, r.n_period)
    df["swing_high"] = swing_highs(df)
    df["swing_low"] = swing_lows(df)
    df["vol_pct"] = volume_percentile(df, r.vol_window)
    df["spread_pct"] = spread_percentile(df, r.vol_window)
    df["ema"] = ema(df, r.ema_period)
    df["ema_slope"] = ema_slope(df, r.ema_period)
    df["spread"] = candle_spread(df)
    df["body_ratio"] = body_ratio(df)

    return df
