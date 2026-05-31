from __future__ import annotations
import numpy as np
import pandas as pd


def market_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sh_vals = df["high"].where(df["swing_high"])
    sl_vals = df["low"].where(df["swing_low"])

    # Build "previous swing" series: for each bar, what was the swing value
    # *before* the most recent one? Use cumcount on non-NaN entries.
    def prev_swing(series: pd.Series) -> pd.Series:
        """At each swing point, the value of the swing before this one."""
        swing_idx = series.dropna().index
        if len(swing_idx) < 2:
            return pd.Series(np.nan, index=series.index)

        prev_vals = pd.Series(np.nan, index=series.index)
        swing_values = series.dropna().values
        for i, idx in enumerate(swing_idx[1:], start=1):
            prev_vals.loc[idx] = swing_values[i - 1]

        return prev_vals.ffill()

    sh_current = sh_vals.ffill()          # most recent swing high value
    sh_previous = prev_swing(sh_vals)     # the swing high before that

    sl_current = sl_vals.ffill()
    sl_previous = prev_swing(sl_vals)

    valid = sh_previous.notna() & sl_previous.notna()

    bullish = valid & (sh_current > sh_previous) & (sl_current > sl_previous)
    bearish = valid & (sh_current < sh_previous) & (sl_current < sl_previous)

    conditions = [bullish, bearish]
    choices = ["bullish", "bearish"]
    df["structure"] = np.select(conditions, choices, default="neutral")

    return df
