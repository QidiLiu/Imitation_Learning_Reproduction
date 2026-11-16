import gymnasium as gym


class StateOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["state"]

    def observation(self, obs):
        return obs["state"]


def make_state_env(cfg=None, seed: int = 0):
    from .push_arm import PushArmConfig, PushArmEnv

    env = PushArmEnv(cfg=cfg or PushArmConfig(obs_mode="state"))
    env = StateOnlyWrapper(env)
    env.reset(seed=seed)
    return env

