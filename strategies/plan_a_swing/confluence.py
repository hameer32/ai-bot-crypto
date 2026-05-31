"""
Plan A — Confluence Scorer
===========================
Combines Tier 1 + Tier 2 + Tier 3 scores into a single final_confidence [0,1].

Weighting:
  Tier 1 (macro bias)    : 30% — direction must be right
  Tier 2 (zone quality)  : 35% — must be near a valid HTF zone
  Tier 3 (entry trigger) : 35% — the actual signal quality

A trade is only taken when final_confidence >= threshold.

Additionally computes:
  rr_ratio : expected RR if TP2 is used (from Tier 3 HTF targets)
             Used to size position more aggressively on high-RR setups.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


TIER1_WEIGHT = 0.30
TIER2_WEIGHT = 0.35
TIER3_WEIGHT = 0.35


def compute_confluence(
    df: pd.DataFrame,
    direction: str,   # 'long' | 'short'
    conf_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Add conf_{direction} and rr_ratio_{direction} columns to df.

    Expects columns: t1_strength, t2_zone_score, t3_entry_score_{direction},
                     close, t3_sl_{direction}, t3_tp2_{direction}
    """
    out = df.copy()
    d = direction

    t1 = out.get("t1_strength",                  pd.Series(0.0, index=out.index))
    t2 = out.get("t2_zone_score",                pd.Series(0.0, index=out.index))
    t3 = out.get(f"t3_entry_score_{d}",          pd.Series(0.0, index=out.index))

    confidence = (
        TIER1_WEIGHT * t1
        + TIER2_WEIGHT * t2
        + TIER3_WEIGHT * t3
    ).clip(0, 1)

    out[f"conf_{d}"] = confidence

    # RR ratio using TP2 / SL distance
    ep  = out["close"]
    sl  = out.get(f"t3_sl_{d}",  pd.Series(np.nan, index=out.index))
    tp2 = out.get(f"t3_tp2_{d}", pd.Series(np.nan, index=out.index))

    r = (ep - sl).abs().replace(0, np.nan)
    if d == "long":
        reward = (tp2 - ep)
    else:
        reward = (ep - tp2)

    out[f"rr_{d}"] = (reward / r).clip(0, 10).fillna(2.0)

    # gate: zero out confidence where signal is not active
    sig_col = f"t3_{d}_signal"
    if sig_col in out.columns:
        out[f"conf_{d}"] = out[f"conf_{d}"].where(out[sig_col], 0.0)
        out[f"rr_{d}"]   = out[f"rr_{d}"].where(out[sig_col], 0.0)

    return out


def above_threshold(df: pd.DataFrame, direction: str, threshold: float) -> pd.Series:
    return df.get(f"conf_{direction}", pd.Series(0.0, index=df.index)) >= threshold
