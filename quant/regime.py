from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    2-state Hidden Markov Model on (log_return, realized_vol, volume).
    State labels are assigned after fitting: higher-vol state = 'trending',
    lower-vol state = 'ranging'.
    """

    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.model = None
        self.trending_state: int = 1  # determined after fit

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        vol = log_ret.rolling(10).std().fillna(0)
        volume_norm = (df["volume"] - df["volume"].mean()) / (df["volume"].std() + 1e-9)
        X = np.column_stack([log_ret.values, vol.values, volume_norm.values])
        return X

    def fit(self, df: pd.DataFrame, lookback: int = 200) -> None:
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed; regime detection disabled")
            return

        data = df.tail(lookback) if len(df) > lookback else df
        X = self._build_features(data)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.model.fit(X)

        # Identify which state has higher volatility → trending state
        means_vol = self.model.means_[:, 1]  # realized vol feature
        self.trending_state = int(np.argmax(means_vol))
        logger.info("HMM fitted. Trending state index: %d", self.trending_state)

    def predict_state(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series("unknown", index=df.index, name="regime_state")
        X = self._build_features(df)
        states = self.model.predict(X)
        labels = ["trending" if s == self.trending_state else "ranging" for s in states]
        return pd.Series(labels, index=df.index, name="regime_state")

    def state_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with P(trending) and P(ranging) for each row.
        P(trending) is used as the quant regime confidence score.
        """
        if self.model is None:
            n = len(df)
            return pd.DataFrame(
                {"p_trending": np.full(n, 0.5), "p_ranging": np.full(n, 0.5)},
                index=df.index,
            )
        X = self._build_features(df)
        proba = self.model.predict_proba(X)
        p_trending = proba[:, self.trending_state]
        p_ranging = 1.0 - p_trending
        return pd.DataFrame(
            {"p_trending": p_trending, "p_ranging": p_ranging},
            index=df.index,
        )
