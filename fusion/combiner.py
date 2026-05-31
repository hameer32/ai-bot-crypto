from __future__ import annotations
import numpy as np
import pandas as pd


def compute_final_confidence(
    P_rules: pd.Series | None,
    P_quant: pd.Series | None,
    P_ml: pd.Series | None,
    cfg,
) -> pd.Series:
    """
    Combine active layer scores into final_confidence ∈ [0, 1].

    Methods:
      weighted_average: normalised weighted mean of active scores
      min_score:        conservative — minimum of active scores
      product:          geometric mean of active scores
    """
    index = None
    for s in [P_rules, P_quant, P_ml]:
        if s is not None:
            index = s.index
            break
    if index is None:
        raise ValueError("All score series are None")

    active_scores: list[tuple[float, pd.Series]] = []
    if P_rules is not None and cfg.rules.enabled:
        active_scores.append((cfg.rules.weight, P_rules.reindex(index).fillna(0.5)))
    if P_quant is not None and cfg.quant.enabled:
        active_scores.append((cfg.quant.weight, P_quant.reindex(index).fillna(0.5)))
    if P_ml is not None and cfg.ml.enabled:
        active_scores.append((cfg.ml.weight, P_ml.reindex(index).fillna(0.5)))

    if not active_scores:
        return pd.Series(0.0, index=index, name="final_confidence")

    method = cfg.fusion.method

    if method == "weighted_average":
        total_weight = sum(w for w, _ in active_scores)
        result = sum(w * s for w, s in active_scores) / total_weight

    elif method == "min_score":
        stacked = pd.concat([s for _, s in active_scores], axis=1)
        result = stacked.min(axis=1)

    elif method == "product":
        n = len(active_scores)
        stacked = pd.concat([s for _, s in active_scores], axis=1)
        result = stacked.prod(axis=1) ** (1 / n)

    else:
        raise ValueError(f"Unknown fusion method: {method}")

    result = result.clip(0, 1)
    result.name = "final_confidence"
    return result


def should_trade(final_confidence: pd.Series, cfg) -> pd.Series:
    return final_confidence >= cfg.fusion.final_threshold
