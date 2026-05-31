from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "atr", "atr_pct", "vol_pct", "spread_pct", "body_ratio", "ema_slope",
    "zscore", "bb_signal", "tsm_score", "microstructure_score",
    "sweep_bull", "sweep_bear", "trap_bull", "trap_bear", "of_bull", "of_bear",
]


class TradingEnv:
    """
    Minimal Gymnasium-compatible trading environment for RL agent training.
    Enable via config: ml.rl.enabled = true

    Action space: Discrete(3) → 0=hold, 1=long, 2=short
    Observation: fixed-size window of normalized feature values
    Reward: realized PnL per step, penalty for drawdown
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, cfg, window: int = 20):
        self.df = df.reset_index(drop=False)
        self.cfg = cfg
        self.window = window
        self.n_features = len(FEATURE_COLS)
        self.action_space_n = 3

        # Observation: window × n_features flattened
        self.obs_shape = (window * self.n_features,)

        self._pos = window
        self._capital = cfg.risk.initial_capital
        self._peak = cfg.risk.initial_capital
        self._in_trade = False
        self._entry_price = 0.0
        self._direction = 0

    def reset(self, seed=None):
        self._pos = self.window
        self._capital = self.cfg.risk.initial_capital
        self._peak = self._capital
        self._in_trade = False
        self._entry_price = 0.0
        self._direction = 0
        return self._get_obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self._pos]
        price = float(row["close"])

        reward = 0.0
        terminated = False

        if self._in_trade:
            # Close trade
            if self._direction == 1:
                reward = (price - self._entry_price) / self._entry_price
            else:
                reward = (self._entry_price - price) / self._entry_price
            self._capital *= (1 + reward * self.cfg.risk.risk_pct_per_trade)
            self._in_trade = False

            # Drawdown penalty
            if self._capital < self._peak:
                dd = (self._peak - self._capital) / self._peak
                reward -= dd * 2.0
            else:
                self._peak = self._capital

        if action in (1, 2) and not self._in_trade:
            self._in_trade = True
            self._entry_price = price
            self._direction = action

        self._pos += 1
        if self._pos >= len(self.df) - 1:
            terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _get_obs(self) -> np.ndarray:
        window_df = self.df.iloc[self._pos - self.window: self._pos]
        obs = []
        for col in FEATURE_COLS:
            if col in window_df.columns:
                vals = window_df[col].fillna(0).values.astype(float)
            else:
                vals = np.zeros(self.window)
            obs.extend(vals.tolist())
        arr = np.array(obs, dtype=np.float32)
        # Normalize to [-1, 1]
        rng = np.abs(arr).max()
        if rng > 0:
            arr = arr / rng
        return arr
