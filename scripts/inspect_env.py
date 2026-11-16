import imageio.v2 as imageio
import numpy as np

from imitation_learning_reproduction.envs.push_arm import PushArmConfig, PushArmEnv

env = PushArmEnv(PushArmConfig(obs_mode="full"))
obs, info = env.reset(seed=0)
frames = []
for _ in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(obs["image"])
    if terminated or truncated:
        break
env.close()
imageio.mimsave("env_preview.gif", frames, fps=10)
print("saved env_preview.gif", np.asarray(frames).shape)

