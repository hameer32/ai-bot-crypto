from __future__ import annotations
import numpy as np
import pandas as pd


def kyle_lambda(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Kyle's Lambda proxy: |Δclose| / volume (rolling mean).
    High lambda = thin market, price moves more per unit volume.
    """
    delta_price = df["close"].diff().abs()
    vol = df["volume"].replace(0, np.nan)
    raw = delta_price / vol
    result = raw.rolling(window).mean()
    result.name = "kyle_lambda"
    return result


def amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Amihud (2002): |return| / dollar_volume, rolling mean.
    High = price is sensitive to order flow.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).abs()
    dollar_vol = df["close"] * df["volume"]
    dollar_vol = dollar_vol.replace(0, np.nan)
    raw = log_ret / dollar_vol
    result = raw.rolling(window).mean()
    result.name = "amihud"
    return result


def roll_spread_estimate(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Roll (1984): effective bid-ask spread proxy.
    spread = 2 * sqrt(max(-cov(Δclose[t], Δclose[t-1]), 0))
    """
    delta = df["close"].diff()
    delta_lag = delta.shift(1)

    def neg_cov(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        half = len(x) // 2
        a, b = x[:half], x[half:]
        if len(a) != len(b):
            b = b[: len(a)]
        cov = np.cov(a, b)[0, 1]
        return float(max(-cov, 0.0))

    combined = pd.concat([delta, delta_lag], axis=1)
    result = combined.apply(lambda row: 2 * np.sqrt(max(-np.cov(row.dropna())[0, 1] if len(row.dropna()) >= 2 else 0, 0)), axis=1)
    result = result.rolling(window).mean()
    result.name = "roll_spread"
    return result


def microstructure_score(df: pd.DataFrame, kyle_window: int = 20, amihud_window: int = 20) -> pd.Series:
    """
    Combines Kyle lambda and Amihud illiquidity into a score ∈ [0, 1].
    Higher score = better order flow conditions (price impact is significant → signals are real).
    """
    kl = kyle_lambda(df, kyle_window)
    am = amihud_illiquidity(df, amihud_window)

    def normalize(s: pd.Series) -> pd.Series:
        min_v = s.rolling(100, min_periods=20).min()
        max_v = s.rolling(100, min_periods=20).max()
        rng = (max_v - min_v).replace(0, np.nan)
        return ((s - min_v) / rng).clip(0, 1).fillna(0.5)

    score = (normalize(kl) + normalize(am)) / 2
    score.name = "microstructure_score"
    return score
