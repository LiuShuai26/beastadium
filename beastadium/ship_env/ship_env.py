import beastadium.ship_env.FlappyBirdEnv as FlappyBirdEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional


class ShipEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        super(ShipEnv, self).__init__()
        self.env = FlappyBirdEnv.FlappyBirdEnv("beastadium/ship_env/FlappyBird.project")

        # Define action and observation space
        self.action_space = spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,))

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
