from __future__ import annotations
import pandas as pd


def order_flow_proxy(
    df: pd.DataFrame,
    vol_pct_thresh: float = 80.0,
    spread_pct_thresh: float = 40.0,
) -> pd.DataFrame:
    df = df.copy()
    midpoint = (df["high"] + df["low"]) / 2
    high_vol = df["vol_pct"] > vol_pct_thresh
    low_spread = df["spread_pct"] < spread_pct_thresh

    # Bullish absorption: high volume, compressed range, close above midpoint
    df["of_bull"] = high_vol & low_spread & (df["close"] > midpoint)

    # Bearish absorption: high volume, compressed range, close below midpoint
    df["of_bear"] = high_vol & low_spread & (df["close"] < midpoint)

    return df
