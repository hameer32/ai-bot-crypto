from __future__ import annotations
import logging
import os

logger = logging.getLogger(__name__)

MODEL_DIR = "ml/models"


class RLTrader:
    """
    RL agent wrapper around stable-baselines3 PPO/DQN.
    Enable via config: ml.rl.enabled = true
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.algo_name = cfg.ml.rl.algorithm
        self.model = None

    def train(self, env, timesteps: int | None = None) -> None:
        try:
            if self.algo_name == "ppo":
                from stable_baselines3 import PPO
                self.model = PPO("MlpPolicy", env, verbose=0)
            elif self.algo_name == "dqn":
                from stable_baselines3 import DQN
                self.model = DQN("MlpPolicy", env, verbose=0)
            else:
                raise ValueError(f"Unknown RL algorithm: {self.algo_name}")
        except ImportError:
            logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3 gymnasium")
            return

        steps = timesteps or self.cfg.ml.rl.timesteps
        logger.info("Training %s for %d timesteps ...", self.algo_name.upper(), steps)
        self.model.learn(total_timesteps=steps)

        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f"rl_{self.algo_name}.zip")
        self.model.save(path)
        logger.info("RL model saved to %s", path)

    def load(self) -> bool:
        path = os.path.join(MODEL_DIR, f"rl_{self.algo_name}.zip")
        if not os.path.exists(path):
            return False
        try:
            if self.algo_name == "ppo":
                from stable_baselines3 import PPO
                self.model = PPO.load(path)
            elif self.algo_name == "dqn":
                from stable_baselines3 import DQN
                self.model = DQN.load(path)
            return True
        except Exception as e:
            logger.error("Failed to load RL model: %s", e)
            return False

    def predict(self, obs) -> int:
        """Return action (0=hold, 1=long, 2=short) given current observation."""
        if self.model is None:
            return 0
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
