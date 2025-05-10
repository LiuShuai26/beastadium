from gymnasium.envs.registration import register


register(
    id="StarEnv",
    entry_point="beastadium.star_env.star_env:StarEnv",
    max_episode_steps=1000,
    reward_threshold=100,
)

register(
    id="ShipEnv",
    entry_point="beastadium.ship_env.ship_env:ShipEnv",
    max_episode_steps=3000,
    reward_threshold=100,
)

register(
    id="SimpleCat",
    entry_point="beastadium.cat_env.cat_env:CatEnv",
    max_episode_steps=3000,
    reward_threshold=100,
)

register(
    id="IronEnv",
    entry_point="beastadium.iron_env.iron_env:IronEnv",
    max_episode_steps=3000,
    reward_threshold=100,
)

register(
    id="HumanoidE",
    entry_point="beastadium.humanoid_env.humanoid_env:HumanoidE",
    max_episode_steps=3000,
    reward_threshold=100,
)