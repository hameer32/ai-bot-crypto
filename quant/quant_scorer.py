from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from .mean_reversion import bollinger_signal
from .microstructure import microstructure_score
from .momentum import tsm_score
from .regime import HMMRegimeDetector

logger = logging.getLogger(__name__)

# Module-level HMM detector instances (one per symbol, lazy-fitted)
_detectors: dict[str, HMMRegimeDetector] = {}


def get_detector(symbol: str, n_states: int = 2) -> HMMRegimeDetector:
    if symbol not in _detectors:
        _detectors[symbol] = HMMRegimeDetector(n_states=n_states)
    return _detectors[symbol]


def compute_P_quant(
    df: pd.DataFrame,
    cfg,
    direction: str,
    symbol: str = "BTCUSDT",
    df_other: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Combine active quant model scores into P_quant ∈ [0, 1].

    direction: 'long' or 'short'
    df_other:  counterpart asset df for correlation score (e.g. ETH when trading BTC)
    """
    qcfg = cfg.quant
    scores: list[pd.Series] = []

    # Mean reversion: for long, high bb_signal (near lower band) is bullish
    if qcfg.mean_reversion.enabled:
        bb = bollinger_signal(df, period=qcfg.mean_reversion.bb_period, std=qcfg.mean_reversion.bb_std)
        if direction == "short":
            bb = 1.0 - bb
        scores.append(bb)

    # Regime: trending regime favors momentum/SMC signals
    if qcfg.regime.enabled:
        detector = get_detector(symbol, qcfg.regime.n_states)
        if detector.model is None:
            logger.info("Fitting HMM for %s ...", symbol)
            detector.fit(df, qcfg.regime.lookback)
        proba = detector.state_proba(df)
        scores.append(proba["p_trending"])

    # Microstructure: high score = good conditions
    if qcfg.microstructure.enabled:
        ms = microstructure_score(df, qcfg.microstructure.kyle_window, qcfg.microstructure.amihud_window)
        scores.append(ms)

    # Momentum: for long, high TSM score (positive trend) supports the trade
    if qcfg.momentum.enabled:
        tsm = tsm_score(df, qcfg.momentum.tsm_lookback, qcfg.momentum.tsm_short_window)
        if direction == "short":
            tsm = 1.0 - tsm
        scores.append(tsm)

    # Correlation divergence: low correlation = stronger individual signal
    if qcfg.correlation.enabled and df_other is not None:
        try:
            from .correlation import rolling_correlation, correlation_divergence_score
            corr = rolling_correlation(df, df_other, qcfg.correlation.corr_window)
            corr_aligned = corr.reindex(df.index).ffill().fillna(0.5)
            div_score = correlation_divergence_score(corr_aligned, qcfg.correlation.corr_filter_thresh)
            scores.append(div_score)
        except Exception as e:
            logger.warning("Correlation score failed: %s", e)

    if not scores:
        return pd.Series(0.5, index=df.index, name="P_quant")

    stacked = pd.concat(scores, axis=1)
    result = stacked.mean(axis=1).clip(0, 1).fillna(0.5)
    result.name = "P_quant"
    return result
