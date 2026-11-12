# 05 · Behavioural Cloning

> **Goal.** Implement the simplest imitation-learning algorithm — supervised
> regression from states to actions — and use it to validate the dataset,
> normalization, checkpointing and evaluation pipeline. Then implement an image
> BC baseline that ACT and Diffusion Policy must beat.
>
> **Prerequisites.** Chapters 02–04.
>
> **Time.** Half a day.
>
> **Artifacts.** `training/normalizer.py`, `models/bc.py`,
> `scripts/train_bc.py`, `configs/bc.yaml`, `tests/test_bc.py`.

---

## 1. The idea

Given demonstrations `D = {(s_i, a_i)}`, fit a policy `π_θ` by maximum
likelihood. For continuous actions with a Gaussian likelihood and fixed variance,
this is mean-squared error:

```text
L(θ) = E_{(s,a)~D} [ ||π_θ(s) - a||² ]
```

For the normalized actions used here, MSE is well behaved. L1 is also common and
is more robust to occasional abrupt actions; try both, but pick one before
comparing methods.

BC is not “just supervised learning” in the sense of being trivial to make work.
The test distribution is different from the training distribution: at test time,
the policy's own actions determine which states it visits. A small error at step
`t` changes the state at `t+1`, which changes the input distribution, which
produces a larger error. This is **compounding error** or **covariate shift**.

A classical bound (Ross & Bagnell) says that if the policy has 0–1 loss `ε` on
the expert distribution and the task horizon is `T`, the expected cost can grow
as `O(εT²)`. The quadratic term is why a policy with 1% per-step error can fail
badly over 60 steps. It also explains why action chunking (ACT) and
receding-horizon diffusion help: they reduce the number of independent decisions.

### What BC can and cannot learn

- It can learn a deterministic mapping when the demonstrations are consistent.
- It cannot represent genuinely multimodal behaviour with an MSE-trained
  deterministic MLP: the optimum is the conditional mean, which may be an action
  that no expert would take. This is the motivation for CVAE/diffusion policies.
- It cannot recover from states it never saw. DAgger (exercise) addresses this by
  querying the expert on the policy's own states.

---

## 2. Normalization

Normalize states with the training mean/std and actions to `[-1,1]` with the
training min/max. Keep the normalizer inside the policy checkpoint so evaluation
cannot accidentally use a different one.

### `training/normalizer.py`

```python
from __future__ import annotations

import torch
from torch import nn


class StateNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class ActionNormalizer(nn.Module):
    """Map actions from [low, high] to [-1, 1] and back."""

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.register_buffer("low", low.float())
        self.register_buffer("high", high.float().clamp_min(1e-6))
        self.register_buffer("scale", (high - low).float().clamp_min(1e-6))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return 2.0 * (a - self.low) / self.scale - 1.0

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) * 0.5 * self.scale + self.low
```

`clamp_min` on `high` is not a substitute for a correct `low < high` check; add
an assertion in the checkpoint loader. A silently degenerate action range
produces a policy that outputs NaNs or a constant.

---

## 3. The BC model

### `models/common.py`

```python
from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...] = (256, 256)) -> nn.Sequential:
    layers = []
    last = in_dim
    for width in hidden:
        layers += [nn.Linear(last, width), nn.SiLU()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)
```

### `models/bc.py`


```python
from __future__ import annotations

import torch
from torch import nn

from imitation_learning_reproduction.models.common import mlp
from imitation_learning_reproduction.training.normalizer import (
    ActionNormalizer, StateNormalizer,
)


class BCPolicy(nn.Module):
    def __init__(self, state_dim: int = 18, action_dim: int = 2,
                 hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.state_normalizer: StateNormalizer | None = None
        self.action_normalizer: ActionNormalizer | None = None
        self.net = mlp(state_dim, action_dim, hidden)

    def set_normalizers(self, state_norm: StateNormalizer,
                        action_norm: ActionNormalizer) -> None:
        self.state_normalizer = state_norm
        self.action_normalizer = action_norm

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_normalizer is None:
            raise RuntimeError("call set_normalizers before forward")
        z = self.state_normalizer(state)
        return torch.tanh(self.net(z))

    @torch.no_grad()
    def act(self, obs: dict) -> "torch.Tensor":
        state = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        action_norm = self.forward(state)
        assert self.action_normalizer is not None
        action = self.action_normalizer.inverse(action_norm)
        return action.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "BCPolicy":
        cfg = ckpt["model_config"]
        model = cls(**cfg)
        # Set the normalizer submodules before loading: their buffers are part
        # of the saved state_dict.
        model.set_normalizers(
            StateNormalizer(
                torch.tensor(ckpt["stats"]["state_mean"]),
                torch.tensor(ckpt["stats"]["state_std"]),
            ),
            ActionNormalizer(
                torch.tensor(ckpt["stats"]["action_low"]),
                torch.tensor(ckpt["stats"]["action_high"]),
            ),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model
```

The `tanh` keeps normalized actions in `[-1,1]`. The action normalizer maps them
to the environment's action range. For this project the environment range is
also `[-1,1]`, so the mapping is the identity in practice; keep it explicit so
the same code works if you change the environment's action bound.

### Why a separate image BC?

A state-only BC baseline is not a fair comparison for image-based ACT/DP. Add a
small image encoder now:

```python
class ImageBCPolicy(BCPolicy):
    def __init__(self, state_dim: int = 18, action_dim: int = 2,
                 image_channels: int = 3, feature_dim: int = 128):
        super().__init__(state_dim, action_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, feature_dim),
        )
        self.state_proj = nn.Linear(state_dim, feature_dim)
        self.head = mlp(2 * feature_dim, action_dim, (256,))

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W) in [-1, 1]
        z = self.state_normalizer(state)
        h = self.encoder(image) + self.state_proj(z)
        return torch.tanh(self.head(h))

    @torch.no_grad()
    def act(self, obs: dict) -> "torch.Tensor":
        state = torch.as_tensor(obs["state"], dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        image = torch.as_tensor(obs["image"], device=self.device)
        image = image.permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        action_norm = self.forward(state, image)
        assert self.action_normalizer is not None
        action = self.action_normalizer.inverse(action_norm)
        return action.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
```

`GroupNorm` rather than `BatchNorm` because the evaluation batch is one. This is a
recurring source of mysterious train/eval differences in imitation learning.

---

## 4. Training loop

`scripts/train_bc.py` should be short enough to read in one sitting. The important
parts are the loss mask, the validation loop, and the checkpoint contents.

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from imitation_learning_reproduction.data.dataset import DemoDataset
from imitation_learning_reproduction.models.bc import BCPolicy, ImageBCPolicy
from imitation_learning_reproduction.training.normalizer import (
    ActionNormalizer, StateNormalizer,
)
from imitation_learning_reproduction.utils.config import apply_overrides, load_config
from imitation_learning_reproduction.utils.seed import resolve_device, seed_everything


@dataclass
class DataConfig:
    root: str = "data/push_arm"
    batch_size: int = 256
    num_workers: int = 0


@dataclass
class ModelConfig:
    kind: str = "state"          # state | image
    hidden: tuple[int, ...] = (256, 256)


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    out: str = "runs/bc"
    val_every: int = 5
    eval_episodes: int = 20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def make_dataset(cfg: Config, split: str):
    keys = ("state",) if cfg.model.kind == "state" else ("state", "image")
    return DemoDataset(cfg.data.root, split=split, obs_horizon=1,
                       action_horizon=1, keys=keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config, Config), args.set)

    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    train_ds = make_dataset(cfg, "train")
    val_ds = make_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)

    stats = train_ds.stats
    state_normalizer = StateNormalizer(torch.tensor(stats.state_mean),
                                       torch.tensor(stats.state_std))
    action_normalizer = ActionNormalizer(torch.tensor(stats.action_low),
                                         torch.tensor(stats.action_high))

    if cfg.model.kind == "state":
        model = BCPolicy(state_dim=18, action_dim=2, hidden=cfg.model.hidden)
    else:
        model = ImageBCPolicy(state_dim=18, action_dim=2)
    model.set_normalizers(state_normalizer, action_normalizer)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    run_dir = Path(cfg.train.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            state = batch["obs"]["state"][:, 0].to(device)  # (B, 18)
            action = batch["action"][:, 0].to(device)       # (B, 2)
            action_target = action_normalizer(action)
            if cfg.model.kind == "state":
                pred = model(state)
            else:
                image = batch["obs"]["image"][:, 0].to(device)  # (B,H,W,3)
                image = image.permute(0, 3, 1, 2).float() / 127.5 - 1.0
                pred = model(state, image)
            loss = F.mse_loss(pred, action_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total += loss.item() * len(state)
            count += len(state)
        scheduler.step()

        if epoch % cfg.train.val_every == 0 or epoch == cfg.train.epochs:
            model.eval()
            val_total, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    state = batch["obs"]["state"][:, 0].to(device)
                    action = batch["action"][:, 0].to(device)
                    target = action_normalizer(action)
                    if cfg.model.kind == "state":
                        pred = model(state)
                    else:
                        image = batch["obs"]["image"][:, 0].to(device)
                        image = image.permute(0, 3, 1, 2).float() / 127.5 - 1.0
                        pred = model(state, image)
                    val_total += F.mse_loss(pred, target).item() * len(state)
                    val_count += len(state)
            val_loss = val_total / max(1, val_count)
            print(f"epoch {epoch:03d} train {total / count:.5f} val {val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "algorithm": "bc",
                    "model": model.state_dict(),
                    "model_config": {"state_dim": 18, "action_dim": 2,
                                     "hidden": cfg.model.hidden}
                    if cfg.model.kind == "state" else
                    {"state_dim": 18, "action_dim": 2},
                    "kind": cfg.model.kind,
                    "stats": stats.to_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": cfg.__dict__,
                }
                torch.save(ckpt, run_dir / "best.pt")

    print("best val loss", best_val, "checkpoint", run_dir / "best.pt")


if __name__ == "__main__":
    main()
```

This is deliberately explicit rather than a framework. In later chapters, the
train loops differ enough (adversarial updates, diffusion loss, chunk masks) that
a premature abstraction would be harder to read.

### `configs/bc.yaml`

```yaml
data:
  root: data/push_arm
  batch_size: 256
  num_workers: 0

model:
  kind: state
  hidden: [256, 256]

train:
  epochs: 200
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  seed: 0
  device: auto
  out: runs/bc
  val_every: 5
  eval_episodes: 20
```

Run:

```bash
uv run python scripts/train_bc.py --config configs/bc.yaml
```

Expected: train/val MSE drops quickly, then plateaus. State BC should reach
around 50–75% success on the evaluation seeds. If it is near random, debug
before moving on; the later algorithms inherit the data pipeline.

---

## 5. What to look for

### Loss curves

- A healthy BC run drops fast in the first 10 epochs and improves slowly after.
- Train loss much lower than val loss: overfitting, common with 80 demonstrations.
  Add image augmentation, weight decay, or reduce model width.
- Both losses flat and high: normalization bug, wrong state slice, or actions not
  changing across episodes.
- Loss goes to zero but success is low: the policy is fitting the training
  trajectories but not recovering from its own mistakes. This is the expected
  BC failure mode, not necessarily a bug.

### Closed-loop evaluation

```bash
uv run python scripts/evaluate.py --policy checkpoint --checkpoint runs/bc/best.pt \
  --obs-mode state --episodes 50
```

Plot the puck-goal distance over time for a few successes and failures. A policy
that gets close and then stalls may be suffering from action saturation or a
distribution shift near the goal. A policy that moves the arm away from the puck
has likely learned a spurious correlation, e.g. “goal position → joint angles”
without using the puck.

### Overfit one batch

Before trusting any training curve:

```python
batch = next(iter(train_loader))
# train the model on this single batch for 200 steps; loss should go below 1e-3
```

If it does not, the bug is in the model/loss, not the data. This test catches
wrong shapes, missing normalization, and accidental detaching.

---

## 6. Image BC baseline

Set `model.kind: image` and rerun. The model gets one 96×96 image and the state.
It is a fair-ish baseline for ACT/DP, though it lacks action chunking and temporal
context.

Expected: image BC is usually **worse** than state BC on this task, because the
image is a lossy view of a state that is already available. That is not a failure
of the method; it shows that visual policies pay for partial observability. The
interesting comparison is under visual shift, where the image policy may be more
robust than a state policy that never sees colours — but our state policy doesn't
use colours either, so this baseline is mainly about architecture and sample
efficiency.

If image BC is much worse:

- Verify the image channel order and normalization.
- Verify the image is not transposed; `imshow` a batch.
- Check that the image contains the puck; at 96×96 it is only a few pixels wide.
- Train longer or add augmentation; 80 demonstrations is small for a CNN.

---

## 7. Compounding error and DAgger

A useful experiment: evaluate BC with a **perturbed initial state** (nudge the
arm by 2 cm or 0.05 rad). State BC often degrades sharply because it never saw
those states. DAgger fixes this by aggregating the expert's labels on the
policy's visited states:

```text
for iteration:
    roll out current policy
    query expert for action at each visited state
    add (visited state, expert action) to dataset
    retrain
```

DAgger is not required for the main benchmark, but it is a good exercise and a
useful diagnostic. If DAgger dramatically improves BC, your policy is suffering
from covariate shift rather than underfitting.

---

## 8. Tests

```python
# tests/test_bc.py
import torch

from imitation_learning_reproduction.models.bc import BCPolicy
from imitation_learning_reproduction.training.normalizer import (
    ActionNormalizer, StateNormalizer,
)


def test_normalizer_roundtrip():
    mean = torch.randn(18)
    std = torch.rand(18) + 0.1
    norm = StateNormalizer(mean, std)
    x = torch.randn(4, 18)
    torch.testing.assert_close(norm.inverse(norm(x)), x, atol=1e-5)


def test_action_normalizer_roundtrip():
    low = torch.tensor([-0.8, -0.9])
    high = torch.tensor([0.7, 0.9])
    norm = ActionNormalizer(low, high)
    a = torch.rand(5, 2) * (high - low) + low
    torch.testing.assert_close(norm.inverse(norm(a)), a, atol=1e-5)


def test_bc_overfits_one_batch():
    torch.manual_seed(0)
    model = BCPolicy()
    model.set_normalizers(
        StateNormalizer(torch.zeros(18), torch.ones(18)),
        ActionNormalizer(-torch.ones(2), torch.ones(2)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = torch.randn(64, 18)
    action = torch.tanh(torch.randn(64, 2))
    for _ in range(300):
        pred = model(state)
        loss = torch.nn.functional.mse_loss(pred, action)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 1e-3
```

The overfit test is a contract: if it fails, stop and fix the model before
running 200 epochs.

---

## 9. Checklist

- [ ] `StateNormalizer` and `ActionNormalizer` round-trip.
- [ ] BC overfits a single batch to < 1e-3 MSE.
- [ ] Train/val losses are finite and decrease.
- [ ] Checkpoint contains model, model config, stats, epoch and val loss.
- [ ] Evaluation loads the checkpoint and reproduces a fixed seed.
- [ ] State BC success is clearly above random (target 50–75%).
- [ ] Image BC baseline is trained with the same split and protocol.
- [ ] You can explain the expected compounding-error failure.

### Exercises

1. Replace MSE with L1 and compare success; does it change smoothness?
2. Implement DAgger for 3 iterations and measure the improvement on perturbed
   initial states.
3. Add action noise during training (input noise on states) and see whether it
   improves closed-loop success.
4. Train BC on 10, 25, 50, 100 demonstrations and plot success vs dataset size.
5. Add a small L2 penalty on the action and measure smoothness.

Next: [06 · PPO](06_ppo.md).
