"""
Plan B — Tier 3: Scalp Entry (5m + 1m)
========================================
Same entry logic as Plan A Tier 3 but applied to 5m/1m data.
Parameters are tighter to suit scalp timeframes:
  - Smaller n_period (faster signals)
  - Lower body_ratio_thresh (scalp candles are smaller)
  - Tighter atr_k (smaller SL buffer)

TP levels:
  - TP1 = 1.5R (lower for scalp — take profit faster)
  - TP2 = nearest 15m/30m FVG or liquidity level
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.plan_a_swing.tier3_entry import build_tier3 as _build_tier3_base


def build_tier3(
    ltf_5m: pd.DataFrame,
    ltf_1m: pd.DataFrame,
    df: pd.DataFrame,
    n_period: int = 15,
    vol_pct_thresh: float = 75.0,
    body_ratio_thresh: float = 0.3,
    sweep_lookback: int = 4,
    atr_k: float = 0.8,
) -> pd.DataFrame:
    """
    Thin wrapper: calls Plan A's tier3 with scalp-tuned defaults,
    then adjusts TP1 to 1.5R instead of 2R for faster profit-taking.
    """
    out = _build_tier3_base(
        ltf_5m, ltf_1m, df,
        n_period=n_period,
        vol_pct_thresh=vol_pct_thresh,
        body_ratio_thresh=body_ratio_thresh,
        sweep_lookback=sweep_lookback,
        atr_k=atr_k,
    )

    # TP1 = 4R minimum (1:4 R:R)
    ep = out["close"]
    for direction in ("long", "short"):
        sl_col  = f"t3_sl_{direction}"
        tp1_col = f"t3_tp1_{direction}"
        if sl_col in out.columns:
            sl = out[sl_col]
            r  = (ep - sl).abs()
            if direction == "long":
                out[tp1_col] = ep + 4 * r
            else:
                out[tp1_col] = ep - 4 * r

    return out
