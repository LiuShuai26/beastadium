import beastadium.simplecat_env.SimpleCatEnv as SimpleCatEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional


class SimpleCat(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        super(SimpleCat, self).__init__()
        self.env = SimpleCatEnv.SimpleCatEnv("beastadium/simplecat_env/SimpleCat.project")

        # Define action and observation space
        self.action_space = spaces.Box(
            np.array([-1]).astype(np.float32),
            np.array([1]).astype(np.float32),
        )
        low = np.array(
            [
                -20.0,
                -20.0,
                -20.0,])
        high = np.array(
            [
                20.0,
                20.0,
                20.0,])

        self.observation_space = spaces.Box(low, high)

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
