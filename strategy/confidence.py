from __future__ import annotations
import pandas as pd

SCORE_WEIGHTS = {
    "htf_alignment": 2,
    "liquidity_sweep": 2,
    "trap": 3,
    "order_flow": 2,
    "volatility": 1,
}
MAX_SCORE = sum(SCORE_WEIGHTS.values())  # 10


def compute_confidence(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Compute rules-based confidence score ∈ [0, 1] for each row.
    direction: 'long' or 'short'
    """
    score = pd.Series(0.0, index=df.index)

    if direction == "long":
        htf_aligned = df.get("htf_structure", pd.Series("neutral", index=df.index)) == "bullish"
        sweep = df.get("sweep_bull", pd.Series(False, index=df.index))
        trap = df.get("trap_bull", pd.Series(False, index=df.index))
        of_sig = df.get("of_bull", pd.Series(False, index=df.index))
    else:
        htf_aligned = df.get("htf_structure", pd.Series("neutral", index=df.index)) == "bearish"
        sweep = df.get("sweep_bear", pd.Series(False, index=df.index))
        trap = df.get("trap_bear", pd.Series(False, index=df.index))
        of_sig = df.get("of_bear", pd.Series(False, index=df.index))

    vol_ok = df.get("vol_regime", pd.Series("normal", index=df.index)).isin(["normal", "high"])

    score += htf_aligned.astype(float) * SCORE_WEIGHTS["htf_alignment"]
    score += sweep.astype(float) * SCORE_WEIGHTS["liquidity_sweep"]
    score += trap.astype(float) * SCORE_WEIGHTS["trap"]
    score += of_sig.astype(float) * SCORE_WEIGHTS["order_flow"]
    score += vol_ok.astype(float) * SCORE_WEIGHTS["volatility"]

    return (score / MAX_SCORE).clip(0, 1).rename("P_rules")


def filter_by_confidence(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    df = df.copy()
    df["confidence_long"] = compute_confidence(df, "long")
    df["confidence_short"] = compute_confidence(df, "short")
    df.loc[df["confidence_long"] < threshold, "long_signal"] = False
    df.loc[df["confidence_short"] < threshold, "short_signal"] = False
    return df
