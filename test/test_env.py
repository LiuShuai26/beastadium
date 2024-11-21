import numpy as np

import gymnasium as gym
import beastadium

env = gym.make("StarEnv")
# env = gym.make("IronEnv")
# env = gym.make("ShipEnv")
# env = gym.make("SimpleCat")

print(env.observation_space.shape)
print(env.action_space)

for i in range(10):
    obs, _ = env.reset()

    for i in range(500):
        # a = env.action_space.sample()
        a = np.array([0.0, 0.0])
        obs, reward, t, done, info = env.step(a)
        # print(f"Step {i}: obs={obs}, reward={reward}, done={done}, info={info}")
        print(f"Step {i}: obs={obs[-7]*25}, reward={reward}, done={done}")
        if done:
            break
