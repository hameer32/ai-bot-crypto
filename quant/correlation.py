from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_correlation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Rolling Pearson correlation of log returns between two assets.
    Assets must share a common timestamp index or be aligned before calling.
    """
    ret_a = np.log(df_a["close"] / df_a["close"].shift(1))
    ret_b = np.log(df_b["close"] / df_b["close"].shift(1))
    ret_a, ret_b = ret_a.align(ret_b, join="inner")
    result = ret_a.rolling(window).corr(ret_b)
    result.name = "btc_eth_corr"
    return result


def correlation_divergence_score(
    corr_series: pd.Series,
    thresh: float = 0.7,
) -> pd.Series:
    """
    Score ∈ [0, 1]: higher when assets are less correlated (diverging).
    corr < thresh → potential independent move → stronger individual signal.
    Maps: corr = 1.0 → score = 0.0, corr = -1.0 → score = 1.0.
    """
    result = ((1.0 - corr_series.clip(-1, 1)) / 2).fillna(0.5)
    result.name = "corr_divergence_score"
    return result
