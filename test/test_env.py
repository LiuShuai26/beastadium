import numpy as np

import gymnasium as gym
import beastadium


env = gym.make("HumanoidE")
# env = gym.make("StarEnv")
# env = gym.make("IronEnv")
# env = gym.make("ShipEnv")
# env = gym.make("SimpleCat")

print(env.observation_space.shape)
print(env.action_space)

for i in range(1):
    obs, _ = env.reset()
    print(f"Reset {i}: obs={obs}")

    for i in range(200):
        a = env.action_space.sample()
        # a = np.array([0.0, 0.0])
        obs, reward, t, done, info = env.step(a)
        # print(f"Step {i}: obs={obs}, reward={reward}, done={done}, info={info}")
        # print(f"Step {i}: obs={obs}, reward={reward}, done={done}")
        print(f"Step {i}: obs shape={obs.shape}, reward={reward}, done={done}")
        if done:
            break
