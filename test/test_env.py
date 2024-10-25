import numpy as np

import gymnasium as gym
import beastadium

# env = gym.make("ShipEnv")
env = gym.make("SimpleCat")

print(env.observation_space)
print(env.action_space)

obs, _ = env.reset()

for i in range(20):
    a = env.action_space.sample()
    obs, reward, t, done, info = env.step(a)
    print(f"Step {i}: obs={obs}, reward={reward}, done={done}, info={info}")
    if done:
        break
