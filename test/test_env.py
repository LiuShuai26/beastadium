import numpy as np

import gymnasium as gym
import beastadium


# env = gym.make("HumanoidE")
# env = gym.make("SimpleCat")
env = gym.make("ROM")

print(env.observation_space.shape)
print(env.action_space)

for i in range(3):
    obs, _ = env.reset()
    print(f"Reset {i}: obs={obs}")

    for i in range(50):
        a = env.action_space.sample()
        # a = np.array([0.0])
        obs, reward, t, done, info = env.step(a)
        print(f"action={a}")
        print(f"Step {i}: obs={obs}, reward={reward}, done={done}, info={info}")
        # print(f"Step {i}: obs={obs}, reward={reward}, done={done}")
        # print(f"Step {i}: obs shape={obs.shape}, reward={reward}, done={done}")
        if done:
            break
