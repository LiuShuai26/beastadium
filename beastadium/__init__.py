from gymnasium.envs.registration import register

register(
    id="SimpleCat",
    entry_point="beastadium.simplecat_env.cat_env:SimpleCat",
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id="HumanoidE",
    entry_point="beastadium.humanoid_env.humanoid_env:HumanoidE",
    max_episode_steps=2000,
    reward_threshold=3000,
)

register(
    id="ROM",
    entry_point="beastadium.rom_env.rom_env:ROM",
    max_episode_steps=2000,
    reward_threshold=3000,
)
