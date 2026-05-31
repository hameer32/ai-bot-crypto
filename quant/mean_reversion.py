from __future__ import annotations
import numpy as np
import pandas as pd


def zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    mean = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    result = (df["close"] - mean) / std.replace(0, np.nan)
    result.name = "zscore"
    return result


def bollinger_signal(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    """
    Returns float in [0, 1].
    close near lower band → 1.0 (bullish mean-revert),
    close near upper band → 0.0 (bearish mean-revert),
    near midline → 0.5.
    """
    mid = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    band_width = upper - lower
    band_width = band_width.replace(0, np.nan)

    # Normalized position within bands: 0 = at lower, 1 = at upper
    pos = (df["close"] - lower) / band_width
    pos = pos.clip(0, 1)

    # Invert so high value = bullish setup (close near bottom of bands)
    result = 1.0 - pos
    result.name = "bb_signal"
    return result


def mean_reversion_halflife(price_series: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life via OLS: Δprice = α + β*price_lag + ε
    half_life = -ln(2) / β
    Returns float (periods). Negative β required for mean reversion.
    """
    try:
        import statsmodels.api as sm
        delta = price_series.diff().dropna()
        lag = price_series.shift(1).dropna()
        lag, delta = lag.align(delta, join="inner")
        X = sm.add_constant(lag)
        res = sm.OLS(delta, X).fit()
        beta = res.params.iloc[1]
        if beta >= 0:
            return float("inf")
        return float(-np.log(2) / beta)
    except Exception:
        return float("inf")
