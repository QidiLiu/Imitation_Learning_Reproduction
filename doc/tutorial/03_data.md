# 03 · Expert Demonstrations, HDF5 Storage, and Window Sampling

> **Goal.** Write a scripted expert that solves the pushing task often enough to
> produce useful demonstrations, record RGB/depth/state/action episodes in a
> validated HDF5 dataset, split episodes without leakage, compute normalization
> statistics, and build the window sampler that every later algorithm uses.
>
> **Prerequisites.** Chapter 02.
>
> **Time.** 1–2 days.
>
> **Artifacts.** `envs/expert.py`, `data/store.py`, `data/dataset.py`,
> `data/stats.py`, `scripts/collect_demos.py`, `scripts/inspect_dataset.py`,
> `tests/test_dataset.py`.

---

## 1. What makes a good demonstration set?

A demonstration is a sequence of observations and the actions that were actually
executed. For this project, each episode is a successful push from a reset state
to the goal. The learned policies do not need the expert's internal plan; they
need the state-action mapping along a competent trajectory.

Three properties matter more than the number of episodes:

1. **Success.** A failed demonstration teaches the policy to fail. Store failures
   for analysis, but train on successes.
2. **Coverage.** Vary the puck position, goal position and initial arm pose. If
   every demonstration pushes straight along +x, BC will fail on the evaluation
   distribution.
3. **Smoothness.** Jerky actions are hard to imitate and produce jittery
   policies. A position-controlled arm driven by a smooth target sequence is
   easier to learn than one driven by bang-bang actions.

The scripted expert below is not meant to be optimal. It is meant to be
*competent, varied, and cheap*. It is a data generator, not a research
contribution.

> **A note on “expert”.** A scripted controller can produce demonstrations that a
> human would not. That is fine for algorithm reproduction, but it changes the
> multimodal structure of the data: a scripted expert tends to be unimodal,
> while human teleoperation is multimodal. Chapters 10–11 discuss why diffusion
> and CVAE policies are designed for multimodal data. If you later record human
> demonstrations, keep the same dataset interface.

---

## 2. The scripted expert

The expert is a small state machine:

```text
reset: arm at a point 7 cm behind the puck along the goal direction
  │
  ├─ approach: move the paddle to the "behind" point
  │       │  (within 1.5 cm)
  │       ▼
  └─ push: drive the paddle 12 cm beyond the puck along the goal direction
          │  (if the paddle loses the puck or drifts sideways)
          └──────────────► approach
```

The two-link IK is analytic (chapter 02). The controller uses the current joint
angles from the observation, converts the IK solution to a normalized action,
and relies on the position actuators.

### `src/imitation_learning_reproduction/envs/expert.py`

```python
from __future__ import annotations

import numpy as np

from .push_arm import ACTION_DIM, BASE_XY, LINK_LENGTHS, STATE_SLICES, ik2


class ScriptedPusher:
    """A small, deterministic pushing controller used only to generate data.

    The controller assumes the environment starts the arm near the "behind"
    point. It re-approaches if the paddle loses contact or drifts sideways.
    """

    def __init__(self, action_scale: float = 0.06, noise: float = 0.0, seed: int = 0):
        self.action_scale = action_scale
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.mode = "approach"

    def reset(self) -> None:
        self.mode = "approach"

    def act(self, state: np.ndarray) -> np.ndarray:
        q = state[STATE_SLICES["q"]]
        ee = state[STATE_SLICES["ee"]]
        puck = state[STATE_SLICES["puck"]]
        goal = state[STATE_SLICES["goal"]]

        delta = goal - puck
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        u = delta / dist

        behind = puck - u * 0.07
        rel = ee - puck
        cross = u[0] * rel[1] - u[1] * rel[0]

        if self.mode == "approach" and np.linalg.norm(ee - behind) < 0.015:
            self.mode = "push"
        if self.mode == "push" and (np.linalg.norm(rel) > 0.11 or abs(cross) > 0.035):
            self.mode = "approach"

        target = behind if self.mode == "approach" else puck + u * 0.12
        q_target = ik2(target, BASE_XY, *LINK_LENGTHS, elbow=-1)
        action = (q_target - q) / self.action_scale
        action = np.clip(action, -1.0, 1.0)

        if self.noise > 0.0:
            action = np.clip(action + self.rng.normal(0.0, self.noise, ACTION_DIM), -1.0, 1.0)
        return action.astype(np.float32)
```

### Tuning the expert

Run 50 episodes and print the success rate before collecting the real dataset:

```python
import numpy as np
from imitation_learning_reproduction.envs.push_arm import PushArmConfig, PushArmEnv
from imitation_learning_reproduction.envs.expert import ScriptedPusher

env = PushArmEnv(PushArmConfig(obs_mode="state"))
expert = ScriptedPusher()
successes, lengths, min_dists = [], [], []
for seed in range(50):
    obs, info = env.reset(seed=seed)
    expert.reset()
    for t in range(env.cfg.episode_steps):
        obs, reward, terminated, truncated, info = env.step(expert.act(obs["state"]))
        if terminated or truncated:
            break
    successes.append(info["success"])
    lengths.append(t + 1)
    min_dists.append(info["min_dist"])
env.close()
print("success", np.mean(successes), "mean length", np.mean(lengths),
      "mean min dist", np.mean(min_dists))
```

A reasonable reference result for this exact model is **55–75%** success with
successful episodes around 20–40 steps. If you get less than 40%:

- Verify that the paddle (not the arm link) is the only colliding robot geom.
- Verify the IK elbow branch is `-1` at reset and in the controller.
- Check that `_reset_arm_behind` puts the paddle 7 cm behind the puck, not on top
  of it.
- Increase `action_scale` if the arm is too slow, or decrease it if it
  overshoots and loses the puck.
- Check the success threshold; 5 cm is intentional. Do not make it 1 cm to
  “improve” the expert.

Do **not** filter demonstrations by “looks nice”. Filter by the success flag.

### Optional: diversity without sacrificing success

Add small action noise (`noise=0.01`) or randomize the behind distance
(`0.06–0.09 m`) per episode. This produces slightly different approach
trajectories and makes BC more robust. Keep the noise small; the success rate is
the constraint.

---

## 3. Dataset format

We use HDF5 because it handles variable-length episodes, compression, and partial
loading with one dependency. A directory of `.npz` files would also work; the
important thing is a schema that is written down and tested.

```text
data/push_arm/
├── demos.hdf5
├── meta.json
└── splits.json
```

### HDF5 tree

```text
demos.hdf5
├── attrs: format_version = 1
└── episodes/
    ├── ep_000000/
    │   ├── attrs: seed, success, length, goal_xy, puck_xy0
    │   ├── obs/
    │   │   ├── state       (T, 18) float32
    │   │   ├── image       (T, 96, 96, 3) uint8
    │   │   └── point_cloud (T, 512, 3) float32
    │   ├── next_state      (T, 18) float32
    │   ├── action          (T, 2)  float32
    │   ├── reward          (T,)    float32
    │   ├── done            (T,)    bool
    │   ├── success         (T,)    bool
    │   └── dist            (T,)    float32
    └── ep_000001/
        ...
```

`next_state[t]` is the state after executing `action[t]`. For a terminal step it
is a copy of `state[t]` (and `done[t]` is true). AIRL's shaping term uses
`(1 - done) * h(next_state)` so the terminal copy does not matter, but a
well-defined value avoids NaNs and makes the transition dataset uniform.

`image` is stored in HWC uint8 exactly as rendered. Do not store normalized
floats; normalization belongs to the training pipeline, and different policies
use different image preprocessing.

### `meta.json`

```json
{
  "format_version": 1,
  "created_at": "2026-11-12T21:30:00",
  "env": "PlanarPush-v1",
  "state_dim": 18,
  "action_dim": 2,
  "image_size": 96,
  "n_points": 512,
  "fps": 20,
  "control_dt": 0.05,
  "episode_steps": 60,
  "success_threshold": 0.05,
  "num_episodes": 200,
  "num_success": 124,
  "success_rate": 0.62,
  "stats": {
    "state_mean": [0.0, 0.0, "... 18 values"],
    "state_std": [1.0, 1.0, "... 18 values"],
    "action_low": [-0.9, -0.8],
    "action_high": [0.9, 0.8],
    "pc_mean": [0.0, 0.0, 0.05],
    "pc_std": [0.2, 0.2, 0.04],
    "image_mean": [0.35, 0.35, 0.38],
    "image_std": [0.18, 0.18, 0.20]
  }
}
```

### `splits.json`

```json
{
  "train": ["ep_000000", "ep_000002", "..."],
  "val":   ["ep_000017", "..."],
  "test":  ["ep_000003", "..."]
}
```

Only successful episodes are split. The test split is **not** touched until the
final benchmark. If you tune on test, you no longer have a test set.

### Statistics

Compute statistics from the **training split only**. Use them for all splits and
all algorithms.

- State: mean and standard deviation per dimension. Floor the standard deviation
  at `1e-6`.
- Action: min and max per dimension; clip to a range that contains the observed
  actions. The environment action is already bounded, so simple min/max is fine.
- Point cloud: mean and standard deviation per xyz axis. The point cloud is in
  world coordinates; do not center per frame.
- Image: mean and std over the training frames in `[0,1]`, for diagnostics.
  Policies normalize images with `x/127.5 - 1`; the dataset stats are not used
  for that transform.

Why train-only? A normalizer fitted on all episodes leaks test information. The
leak is small for mean/std, but the habit is not.

---

## 4. Dataset reader and window sampler

### `data/stats.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    pc_mean: np.ndarray
    pc_std: np.ndarray
    image_mean: np.ndarray
    image_std: np.ndarray

    @classmethod
    def compute(cls, episodes: list[dict]) -> "DatasetStats":
        states = np.concatenate([ep["state"] for ep in episodes], axis=0)
        actions = np.concatenate([ep["action"] for ep in episodes], axis=0)
        clouds = np.concatenate([ep["point_cloud"] for ep in episodes], axis=0)
        images = np.concatenate([ep["image"] for ep in episodes], axis=0)

        state_mean = states.mean(axis=0).astype(np.float32)
        state_std = states.std(axis=0).astype(np.float32)
        state_std = np.maximum(state_std, 1e-6)

        action_low = actions.min(axis=0).astype(np.float32)
        action_high = actions.max(axis=0).astype(np.float32)
        # Guard against a constant action dimension.
        low = np.minimum(action_low, action_high - 1e-3)
        high = np.maximum(action_high, low + 1e-3)

        pc_mean = clouds.reshape(-1, 3).mean(axis=0).astype(np.float32)
        pc_std = clouds.reshape(-1, 3).std(axis=0).astype(np.float32)
        pc_std = np.maximum(pc_std, 0.01)

        image_float = images.astype(np.float32) / 255.0
        image_mean = image_float.mean(axis=(0, 1, 2)).astype(np.float32)
        image_std = image_float.std(axis=(0, 1, 2)).astype(np.float32)
        image_std = np.maximum(image_std, 1e-3)

        return cls(state_mean, state_std, low, high, pc_mean, pc_std,
                   image_mean, image_std)

    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "pc_mean": self.pc_mean.tolist(),
            "pc_std": self.pc_std.tolist(),
            "image_mean": self.image_mean.tolist(),
            "image_std": self.image_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetStats":
        return cls(
            state_mean=np.asarray(d["state_mean"], dtype=np.float32),
            state_std=np.asarray(d["state_std"], dtype=np.float32),
            action_low=np.asarray(d["action_low"], dtype=np.float32),
            action_high=np.asarray(d["action_high"], dtype=np.float32),
            pc_mean=np.asarray(d["pc_mean"], dtype=np.float32),
            pc_std=np.asarray(d["pc_std"], dtype=np.float32),
            image_mean=np.asarray(d["image_mean"], dtype=np.float32),
            image_std=np.asarray(d["image_std"], dtype=np.float32),
        )
```

### `data/store.py`

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .stats import DatasetStats


def write_episode(f: h5py.File, name: str, episode: dict, attrs: dict) -> None:
    group = f.create_group(f"episodes/{name}")
    for key, value in attrs.items():
        group.attrs[key] = value

    obs = group.create_group("obs")
    obs.create_dataset("state", data=episode["state"], compression="gzip", compression_opts=4)
    obs.create_dataset(
        "image", data=episode["image"], compression="gzip", compression_opts=4,
        chunks=(1, *episode["image"].shape[1:]),
    )
    obs.create_dataset(
        "point_cloud", data=episode["point_cloud"], compression="gzip", compression_opts=4,
        chunks=(1, *episode["point_cloud"].shape[1:]),
    )
    group.create_dataset("next_state", data=episode["next_state"], compression="gzip",
                         compression_opts=4)
    group.create_dataset("action", data=episode["action"])
    group.create_dataset("reward", data=episode["reward"])
    group.create_dataset("done", data=episode["done"])
    group.create_dataset("success", data=episode["success"])
    group.create_dataset("dist", data=episode["dist"])


def load_episodes(path: str | Path, split: list[str] | None = None) -> list[dict]:
    episodes = []
    with h5py.File(path, "r") as f:
        names = sorted(f["episodes"].keys())
        if split is not None:
            names = [n for n in names if n in set(split)]
        for name in names:
            g = f[f"episodes/{name}"]
            ep = {
                "name": name,
                "state": g["obs/state"][:],
                "image": g["obs/image"][:],
                "point_cloud": g["obs/point_cloud"][:],
                "next_state": g["next_state"][:],
                "action": g["action"][:],
                "reward": g["reward"][:],
                "done": g["done"][:],
                "success": g["success"][:],
                "dist": g["dist"][:],
                "success_episode": bool(g.attrs["success"]),
                "seed": int(g.attrs["seed"]),
                "length": int(g.attrs["length"]),
            }
            episodes.append(ep)
    return episodes


def write_metadata(root: str | Path, meta: dict, splits: dict) -> None:
    root = Path(root)
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    (root / "splits.json").write_text(json.dumps(splits, indent=2))


def make_splits(successful_names: list[str], seed: int = 0,
                fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict:
    names = list(successful_names)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n = len(names)
    n_val = max(1, int(round(n * fractions[1])))
    n_test = max(1, int(round(n * fractions[2])))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough successful episodes for a train split")
    return {
        "train": sorted(names[:n_train]),
        "val": sorted(names[n_train:n_train + n_val]),
        "test": sorted(names[n_train + n_val:]),
    }
```

### `data/dataset.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .store import load_episodes


class DemoDataset(Dataset):
    """Windowed demonstration dataset.

    Shapes returned for one item:
        obs["state"]       (T_o, 18)
        obs["image"]       (T_o, H, W, 3) uint8   (if requested)
        obs["point_cloud"] (T_o, N, 3) float32   (if requested)
        action             (T_p, 2) float32
        action_pad         (T_p,) bool
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        obs_horizon: int = 1,
        action_horizon: int = 1,
        keys: tuple[str, ...] = ("state",),
        stride: int = 1,
    ):
        self.root = Path(root)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.keys = keys

        import json
        splits = json.loads((self.root / "splits.json").read_text())
        meta = json.loads((self.root / "meta.json").read_text())
        if split not in splits:
            raise KeyError(f"split {split!r} not in {list(splits)}")
        self.episodes = load_episodes(self.root / "demos.hdf5", splits[split])
        self.episodes = [ep for ep in self.episodes if ep["success_episode"]]
        if not self.episodes:
            raise RuntimeError(f"no successful episodes in split {split!r}")

        self.index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            for t in range(0, len(ep["action"]), stride):
                self.index.append((ep_idx, t))

        # Training statistics come from meta, computed on the train split.
        from .stats import DatasetStats
        self.stats = DatasetStats.from_dict(meta["stats"])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]
        length = len(ep["action"])

        obs_idx = np.clip(
            np.arange(t - self.obs_horizon + 1, t + 1), 0, length - 1
        )
        act_idx = np.arange(t, t + self.action_horizon)
        act_mask = act_idx < length
        act_idx = np.clip(act_idx, 0, length - 1)

        obs = {}
        if "state" in self.keys:
            obs["state"] = ep["state"][obs_idx]
        if "image" in self.keys:
            obs["image"] = ep["image"][obs_idx]
        if "point_cloud" in self.keys:
            obs["point_cloud"] = ep["point_cloud"][obs_idx]

        return {
            "obs": obs,
            "action": ep["action"][act_idx],
            "action_pad": act_mask,
            "episode": ep_idx,
            "t": t,
        }


class TransitionDataset(Dataset):
    """Flat (s, a, s', done) transitions for AIRL/GAIL expert sampling."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        import json
        splits = json.loads((self.root / "splits.json").read_text())
        episodes = load_episodes(self.root / "demos.hdf5", splits[split])
        episodes = [ep for ep in episodes if ep["success_episode"]]
        self.state = np.concatenate([ep["state"] for ep in episodes], axis=0)
        self.action = np.concatenate([ep["action"] for ep in episodes], axis=0)
        self.next_state = np.concatenate([ep["next_state"] for ep in episodes], axis=0)
        self.done = np.concatenate([ep["done"] for ep in episodes], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, idx: int):
        return {
            "state": torch.from_numpy(self.state[idx]),
            "action": torch.from_numpy(self.action[idx]),
            "next_state": torch.from_numpy(self.next_state[idx]),
            "done": torch.tensor(self.done[idx]),
        }
```

The dataset preloads all episodes into memory. For 120 successful episodes of
about 40 steps each, the images are roughly 120×40×96×96×3 ≈ 130 MB compressed
and somewhat more uncompressed; this is fine. For a larger dataset, switch to
lazy `h5py` reads or a memory-mapped array; the sampler interface does not change.

### Padding semantics

For a window starting at `t`:

- Observation indices are clipped to `[0, T-1]`, so the first observation is
  repeated at the beginning of an episode. This is standard and avoids a separate
  mask for the encoder.
- Action indices beyond the episode end are clipped to the last action and
  flagged by `action_pad=False`.
- The loss masks padded action positions. Never train on the repeated last action
  as if it were a real target.
- For `T_o=1` and `T_p=1` (BC), no padding is needed except at the episode end,
  where the last transition is still valid.

---

## 5. Collection script

`scripts/collect_demos.py`:

```python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from imitation_learning_reproduction.data.stats import DatasetStats
from imitation_learning_reproduction.data.store import make_splits, write_episode, write_metadata
from imitation_learning_reproduction.envs.expert import ScriptedPusher
from imitation_learning_reproduction.envs.push_arm import PushArmConfig, PushArmEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/push_arm")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--expert-noise", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    h5_path = root / "demos.hdf5"
    if h5_path.exists() and not args.overwrite:
        raise SystemExit(f"{h5_path} exists; pass --overwrite or choose another --out")

    cfg = PushArmConfig(
        obs_mode="full",
        image_size=args.image_size,
        n_points=args.n_points,
    )
    env = PushArmEnv(cfg)
    expert = ScriptedPusher(action_scale=cfg.action_scale, noise=args.expert_noise,
                            seed=args.seed)

    successful: list[dict] = []
    successful_names: list[str] = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["format_version"] = 1
        f.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")

        for i in tqdm(range(args.num_episodes), desc="collect"):
            seed = args.seed + i
            obs, info = env.reset(seed=seed)
            expert.reset()
            states, next_states, actions, rewards = [], [], [], []
            images, clouds, dones, successes, dists = [], [], [], [], []

            for _ in range(cfg.episode_steps):
                action = expert.act(obs["state"])
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                states.append(obs["state"])
                images.append(obs["image"])
                clouds.append(obs["point_cloud"])
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_obs["state"])
                dones.append(done)
                successes.append(bool(info["success"]))
                dists.append(float(info["dist"]))
                obs = next_obs
                if done:
                    break

            name = f"ep_{i:06d}"
            episode = {
                "state": np.asarray(states, dtype=np.float32),
                "image": np.asarray(images, dtype=np.uint8),
                "point_cloud": np.asarray(clouds, dtype=np.float32),
                "next_state": np.asarray(next_states, dtype=np.float32),
                "action": np.asarray(actions, dtype=np.float32),
                "reward": np.asarray(rewards, dtype=np.float32),
                "done": np.asarray(dones, dtype=bool),
                "success": np.asarray(successes, dtype=bool),
                "dist": np.asarray(dists, dtype=np.float32),
            }
            attrs = {
                "seed": seed,
                "success": bool(info["success"]),
                "length": len(actions),
                "goal_xy": np.asarray(info["goal"]).tolist(),
                "puck_xy0": episode["state"][0, 6:8].tolist(),
            }
            write_episode(f, name, episode, attrs)
            if attrs["success"]:
                successful.append(episode)
                successful_names.append(name)

    env.close()

    if len(successful) < 10:
        raise SystemExit(f"only {len(successful)} successful episodes; tune the expert first")

    splits = make_splits(successful_names, seed=args.seed)
    train_names = set(splits["train"])
    train_episodes = [
        ep for ep, name in zip(successful, successful_names) if name in train_names
    ]
    stats = DatasetStats.compute(train_episodes)

    meta = {
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env": "PlanarPush-v1",
        "state_dim": 18,
        "action_dim": 2,
        "image_size": args.image_size,
        "n_points": args.n_points,
        "fps": 20,
        "control_dt": cfg.frame_skip * 0.002,
        "episode_steps": cfg.episode_steps,
        "success_threshold": cfg.success_threshold,
        "num_episodes": args.num_episodes,
        "num_success": len(successful),
        "success_rate": len(successful) / args.num_episodes,
        "stats": stats.to_dict(),
    }
    write_metadata(root, meta, splits)
    print(f"wrote {len(successful)}/{args.num_episodes} successful episodes to {root}")
    print(f"splits: {len(splits['train'])} train, {len(splits['val'])} val, "
          f"{len(splits['test'])} test")


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python scripts/collect_demos.py --out data/push_arm --num-episodes 200 --overwrite
```

Expected: a `demos.hdf5` of a few hundred MB, `meta.json`, `splits.json`, and a
success rate around 55–75%. Collection should take a few minutes, dominated by
rendering.

---

## 6. Dataset QA

Write `scripts/inspect_dataset.py` to print a report and save visual artifacts:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from imitation_learning_reproduction.data.store import load_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/push_arm")
    parser.add_argument("--out", default="data/push_arm/qa")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "meta.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    episodes = load_episodes(root / "demos.hdf5", splits["train"])

    lengths = np.array([len(ep["action"]) for ep in episodes])
    states = np.concatenate([ep["state"] for ep in episodes])
    actions = np.concatenate([ep["action"] for ep in episodes])
    images = np.concatenate([ep["image"] for ep in episodes])
    clouds = np.concatenate([ep["point_cloud"] for ep in episodes])

    print("success rate:", meta["num_success"] / meta["num_episodes"])
    print("train episodes:", len(episodes))
    print("length: min/mean/max", lengths.min(), lengths.mean(), lengths.max())
    print("state mean/std", states.mean(0).round(3), states.std(0).round(3))
    print("action min/max", actions.min(0).round(3), actions.max(0).round(3))
    print("action std", actions.std(0).round(3))
    print("image mean/std", images.mean().round(1), images.std().round(1))
    print("cloud z: min/mean/max", clouds[..., 2].min(), clouds[..., 2].mean(),
          clouds[..., 2].max())
    print("object points fraction", (clouds[..., 2] > 0.004).mean().round(3))

    # Contact sheet: first frame of the first 12 training episodes.
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, ep in zip(axes.ravel(), episodes[:12]):
        ax.imshow(ep["image"][0])
        ax.set_title(ep["name"], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "contact_sheet.png", dpi=120)
    plt.close(fig)

    # One episode as a GIF and a point-cloud top view.
    ep = episodes[0]
    imageio.mimsave(out / "episode_0000.gif", list(ep["image"]), fps=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ep["point_cloud"][-1, :, 0], ep["point_cloud"][-1, :, 1],
               c=ep["point_cloud"][-1, :, 2], s=3, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title("final point cloud, top view")
    fig.tight_layout()
    fig.savefig(out / "point_cloud_final.png", dpi=120)
    plt.close(fig)
    print("wrote QA artifacts to", out)


if __name__ == "__main__":
    main()
```

Check the report against the following:

| Check | Expected |
|---|---|
| Success rate | 0.55–0.80 |
| Train episodes | ≥ 80 |
| Length mean | 20–45 steps |
| Action std | > 0.05 per dimension |
| Image std | > 10 (uint8) |
| Object point fraction | 0.05–0.5 |
| Cloud z max | > 0.08 (goal post/top) |
| NaNs/infs | none |

If the action standard deviation is tiny, the expert is nearly constant and BC
has nothing to learn. If the image standard deviation is tiny, the camera may be
pointing at an empty table or the lights are saturating. If the object point
fraction is zero, the point-cloud sampler is filtering out the objects.

---

## 7. Tests

```python
# tests/test_dataset.py
import json

import numpy as np
import pytest

from imitation_learning_reproduction.data.dataset import DemoDataset, TransitionDataset
from imitation_learning_reproduction.data.stats import DatasetStats


def test_dataset_shapes(tmp_path, synthetic_dataset):
    ds = DemoDataset(synthetic_dataset, split="train", obs_horizon=2,
                     action_horizon=4, keys=("state", "image", "point_cloud"))
    item = ds[0]
    assert item["obs"]["state"].shape == (2, 18)
    assert item["obs"]["image"].shape[1:] == (96, 96, 3)
    assert item["obs"]["point_cloud"].shape[1:] == (512, 3)
    assert item["action"].shape == (4, 2)
    assert item["action_pad"].shape == (4,)


def test_action_padding_at_end(synthetic_dataset):
    ds = DemoDataset(synthetic_dataset, split="train", obs_horizon=1,
                     action_horizon=8, keys=("state",))
    ep = ds.episodes[0]
    last_t = len(ep["action"]) - 1
    idx = next(i for i, (e, t) in enumerate(ds.index) if e == 0 and t == last_t)
    item = ds[idx]
    assert item["action_pad"].sum() == 1
    assert not item["action_pad"][1:].any()


def test_no_split_overlap(synthetic_dataset):
    splits = json.loads((synthetic_dataset / "splits.json").read_text())
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["val"]) & set(splits["test"])


def test_stats_roundtrip():
    stats = DatasetStats(
        state_mean=np.zeros(18, np.float32), state_std=np.ones(18, np.float32),
        action_low=-np.ones(2, np.float32), action_high=np.ones(2, np.float32),
        pc_mean=np.zeros(3, np.float32), pc_std=np.ones(3, np.float32),
        image_mean=np.zeros(3, np.float32), image_std=np.ones(3, np.float32),
    )
    restored = DatasetStats.from_dict(stats.to_dict())
    np.testing.assert_allclose(restored.state_mean, stats.state_mean)
```

The `synthetic_dataset` fixture is worth writing: create 6 episodes of length 5
with random arrays and a tiny `meta.json`/`splits.json`, so the dataset tests do
not need a full collection run. A fixture of 30 lines saves minutes per test
iteration.

### A note on HDF5 and multiprocessing

`h5py.File` objects are not safely pickled across DataLoader workers. This
dataset preloads arrays in `__init__`, so worker processes receive NumPy arrays,
not open files. If you switch to lazy loading, open the file inside each worker's
`__getitem__`, or use one worker on CPU. This is a common source of
“file already open” errors.

---

## 8. Checklist

- [ ] The scripted expert succeeds on 55–80% of 50 seeds.
- [ ] 200 collection attempts yield at least 80 successful episodes.
- [ ] `demos.hdf5` opens and has the documented groups/attrs.
- [ ] `meta.json` stats are computed from the train split only.
- [ ] `splits.json` has no overlap and only successful episodes.
- [ ] `DemoDataset` returns the documented shapes for `T_o=1/2` and `T_p=1/16`.
- [ ] Padded actions are masked; no loss is computed on padding.
- [ ] QA report shows varied actions, non-flat images, and object points.
- [ ] Tests pass.

### Exercises

1. Record 50 episodes with action noise and compare the state distribution to the
   noiseless set. Does BC improve or degrade?
2. Add a `--only-success` flag that does not store failed episodes, and measure
   how much disk space it saves.
3. Add a `next_state`-aware window sampler that returns `(s_t, a_t, s_{t+1})` for
   AIRL without a separate `TransitionDataset`.
4. Write a dataset validator that fails if two episodes have identical goal/puck
   positions and identical actions. This catches a broken reset seed.

Next: [04 · Evaluation protocol](04_evaluation.md).
