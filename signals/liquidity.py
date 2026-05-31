from __future__ import annotations
import pandas as pd


def liquidity_sweep(
    df: pd.DataFrame,
    vol_pct_thresh: float = 80.0,
) -> pd.DataFrame:
    df = df.copy()
    prev_roll_low = df["roll_low"].shift(1)
    prev_roll_high = df["roll_high"].shift(1)

    # Bullish sweep: low wicks below prior N-period low then closes back above it
    df["sweep_bull"] = (
        (df["low"] < prev_roll_low)
        & (df["close"] > prev_roll_low)
        & (df["vol_pct"] > vol_pct_thresh)
    )

    # Bearish sweep: high wicks above prior N-period high then closes back below it
    df["sweep_bear"] = (
        (df["high"] > prev_roll_high)
        & (df["close"] < prev_roll_high)
        & (df["vol_pct"] > vol_pct_thresh)
    )

    # Store the swept levels for use in SL calculation
    df["sweep_low_level"] = prev_roll_low.where(df["sweep_bull"])
    df["sweep_high_level"] = prev_roll_high.where(df["sweep_bear"])

    return df
