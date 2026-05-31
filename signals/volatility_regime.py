from __future__ import annotations
import numpy as np
import pandas as pd


def volatility_regime(
    df: pd.DataFrame,
    high_pct: float = 70.0,
    low_pct: float = 30.0,
) -> pd.DataFrame:
    df = df.copy()
    conditions = [
        df["atr_pct"] >= high_pct,
        df["atr_pct"] <= low_pct,
    ]
    choices = ["high", "low"]
    df["vol_regime"] = np.select(conditions, choices, default="normal")
    return df
