import numpy as np
import pytest

from imitation_learning_reproduction.envs.push_arm import (
    ACTION_DIM, STATE_DIM, PushArmConfig, PushArmEnv,
)


@pytest.mark.parametrize("obs_mode", ["state", "image", "point_cloud", "full"])
def test_observation_shapes(obs_mode):
    env = PushArmEnv(PushArmConfig(obs_mode=obs_mode, image_size=64, n_points=128))
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (STATE_DIM,)
    assert obs["state"].dtype == np.float32
    if obs_mode in {"image", "full"}:
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["image"].std() > 1.0
    if obs_mode in {"point_cloud", "full"}:
        assert obs["point_cloud"].shape == (128, 3)
        assert np.isfinite(obs["point_cloud"]).all()
    env.close()


def test_step_and_bounds():
    env = PushArmEnv(PushArmConfig(obs_mode="state"))
    obs, info = env.reset(seed=1)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.ones(ACTION_DIM))
        assert obs["state"].shape == (STATE_DIM,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()


def test_reset_determinism():
    env = PushArmEnv(PushArmConfig(obs_mode="state"))
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    np.testing.assert_allclose(a["state"], b["state"])
    env.close()


def test_task_distribution_is_reachable():
    env = PushArmEnv(PushArmConfig(obs_mode="state"))
    for seed in range(50):
        _, info = env.reset(seed=seed)
        assert np.linalg.norm(info["goal"] - np.array([-0.38, 0.0])) < 0.63
        assert np.linalg.norm(info["puck"] - info["goal"]) > 0.10
    env.close()


def test_point_cloud_contains_objects():
    env = PushArmEnv(PushArmConfig(obs_mode="point_cloud", image_size=64, n_points=256))
    obs, info = env.reset(seed=3)
    pc = obs["point_cloud"]
    assert pc.shape == (256, 3)
    # The puck top is around z=0.05; the arm is around z=0.06; the goal post is taller.
    assert pc[:, 2].max() > 0.04
    assert (pc[:, 2] > 0.004).sum() > 20
    env.close()

