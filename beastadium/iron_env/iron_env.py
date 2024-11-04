import beastadium.iron_env.IrononeEnv as IrononeEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional


class IronEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        super(IronEnv, self).__init__()
        self.env = IrononeEnv.IrononeEnv("beastadium/iron_env/Ironone.project")

        # Define action and observation space
        self.action_space = spaces.Box(-1, 1, shape=(8,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(42,))

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)
        result = self.env.step(action)
        terminated = False
        return np.array(result.observation), result.reward, terminated, result.done, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        result = self.env.reset()
        return np.array(result.observation), {}

    def close(self):
        pass
