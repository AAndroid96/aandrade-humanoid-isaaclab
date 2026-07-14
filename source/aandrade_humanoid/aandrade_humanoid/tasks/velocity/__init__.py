import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-AAndrade-Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:AAndradeHumanoidFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:AAndradeHumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-AAndrade-Humanoid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:AAndradeHumanoidFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:AAndradeHumanoidFlatPPORunnerCfg",
    },
)
