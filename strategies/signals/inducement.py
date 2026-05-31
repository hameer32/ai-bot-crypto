"""
Inducement / Liquidity Sweep Detection
---------------------------------------
ICT's "manipulation" phase: before the real move, smart money sweeps
the opposing liquidity pool (stop losses) then reverses.

Bullish sweep (for long entry):
  - Equal lows / swing lows = Sell-Side Liquidity (SSL)
  - Price wicks BELOW the SSL level, then CLOSES back above it
  - This "takes out" stop losses sitting below equal lows
  - The wick below = inducement; the close back above = confirmation

Bearish sweep (for short entry):
  - Equal highs / swing highs = Buy-Side Liquidity (BSL)
  - Price wicks ABOVE the BSL level, then CLOSES back below it

Outputs per bar
---------------
  ssl_level      : float — nearest SSL (equal lows cluster)
  bsl_level      : float — nearest BSL (equal highs cluster)
  ssl_swept      : bool  — SSL taken out within sweep_lookback bars
  bsl_swept      : bool  — BSL taken out within sweep_lookback bars
  sweep_low      : float — the actual wick low during SSL sweep
  sweep_high     : float — the actual wick high during BSL sweep
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _find_equal_levels(
    levels: np.ndarray,
    tolerance_pct: float = 0.002,
) -> list[float]:
    """Cluster nearby price levels (within tolerance_pct) → return cluster means."""
    if len(levels) == 0:
        return []
    sorted_l = np.sort(levels)
    clusters: list[list[float]] = [[sorted_l[0]]]
    for v in sorted_l[1:]:
        if abs(v - clusters[-1][-1]) / max(abs(clusters[-1][-1]), 1e-9) <= tolerance_pct:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # Only clusters with ≥2 members qualify as liquidity pools
    return [float(np.mean(c)) for c in clusters if len(c) >= 2]


def build_inducement(
    df: pd.DataFrame,
    swing_window: int = 5,
    sweep_lookback: int = 8,
    equal_tol: float = 0.002,
) -> pd.DataFrame:
    """
    Detect SSL/BSL liquidity pools and sweeps for each bar.
    """
    df = df.copy()
    n = len(df)

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    ssl_level  = np.full(n, np.nan)
    bsl_level  = np.full(n, np.nan)
    ssl_swept  = np.zeros(n, dtype=bool)
    bsl_swept  = np.zeros(n, dtype=bool)
    sweep_low  = np.full(n, np.nan)
    sweep_high = np.full(n, np.nan)

    lookback_bars = 50  # bars to look back for swing levels

    for i in range(swing_window * 2, n):
        start = max(0, i - lookback_bars)

        # Find swing lows in lookback window → SSL candidates
        window_lows  = []
        window_highs = []
        for j in range(start, i - swing_window):
            if lows[j] == min(lows[max(0, j - swing_window):j + swing_window + 1]):
                window_lows.append(lows[j])
            if highs[j] == max(highs[max(0, j - swing_window):j + swing_window + 1]):
                window_highs.append(highs[j])

        # Find equal low clusters (SSL) and equal high clusters (BSL)
        ssl_clusters = _find_equal_levels(np.array(window_lows), equal_tol)
        bsl_clusters = _find_equal_levels(np.array(window_highs), equal_tol)

        # Nearest SSL below current close, nearest BSL above
        c = closes[i]
        below_ssl = [s for s in ssl_clusters if s < c]
        above_bsl = [b for b in bsl_clusters if b > c]

        if below_ssl:
            ssl_level[i] = max(below_ssl)   # closest SSL below price
        if above_bsl:
            bsl_level[i] = min(above_bsl)   # closest BSL above price

        # Check for sweep in last sweep_lookback bars
        lo_window = lows[max(0, i - sweep_lookback):i + 1]
        hi_window = highs[max(0, i - sweep_lookback):i + 1]
        cl_window = closes[max(0, i - sweep_lookback):i + 1]

        # SSL sweep: wick below SSL, close back above
        if not np.isnan(ssl_level[i]):
            sl = ssl_level[i]
            swept_idx = np.where(lo_window < sl)[0]
            if len(swept_idx) > 0:
                k = swept_idx[-1]  # most recent sweep bar
                if cl_window[k] >= sl:   # close recovered above SSL
                    ssl_swept[i] = True
                    sweep_low[i] = lo_window[k]

        # BSL sweep: wick above BSL, close back below
        if not np.isnan(bsl_level[i]):
            bl = bsl_level[i]
            swept_idx = np.where(hi_window > bl)[0]
            if len(swept_idx) > 0:
                k = swept_idx[-1]
                if cl_window[k] <= bl:   # close recovered below BSL
                    bsl_swept[i] = True
                    sweep_high[i] = hi_window[k]

    df["ssl_level"]  = ssl_level
    df["bsl_level"]  = bsl_level
    df["ssl_swept"]  = ssl_swept
    df["bsl_swept"]  = bsl_swept
    df["sweep_low"]  = sweep_low
    df["sweep_high"] = sweep_high

    return df
