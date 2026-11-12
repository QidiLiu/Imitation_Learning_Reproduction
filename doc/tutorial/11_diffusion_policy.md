# 11 · Diffusion Policy

> **Goal.** Implement a conditional diffusion policy over action chunks: a
> DDPM/DDIM noise scheduler, a 1-D U-Net with FiLM conditioning, an
> image/proprioception encoder, and receding-horizon control. Train it on the same
> dataset and compare it with ACT and image BC.
>
> **Prerequisites.** Chapters 03–05. Chapter 10 is helpful for chunking.
>
> **Time.** 2 days.
>
> **Artifacts.** `models/diffusion.py`, `models/dp.py`, `scripts/train_dp.py`,
> `configs/dp.yaml`, `tests/test_diffusion.py`.

---

## 1. Why diffusion for actions?

A deterministic MSE policy predicts the conditional mean. If two experts push the
puck around opposite sides, the mean is “push straight through the puck”, which is
not a valid behaviour. A diffusion policy models the full conditional distribution
`p(a_{t:t+T_p} | o_t)` by learning to denoise samples.

Diffusion Policy (Chi et al., 2023) adds three practical choices:

1. **Action chunks** rather than single actions, reducing compounding error.
2. **Observation history** `T_o` rather than a single frame.
3. **Receding-horizon execution**: sample a chunk, execute a prefix, resample.

The result is a multimodal, stable visuomotor policy that is surprisingly robust
on manipulation tasks.

---

## 2. Diffusion in one page

### Forward process

For a clean action chunk `x_0`:

```text
q(x_t | x_0) = N( sqrt(ᾱ_t) x_0, (1 - ᾱ_t) I )
x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε,   ε ~ N(0, I)
```

with `β_t` a noise schedule and `ᾱ_t = Π_{s≤t}(1-β_s)`.

### Training

Predict the noise with a network `ε_θ(x_t, t, c)` conditioned on observation
features `c`:

```text
L = E_{t,x_0,ε} [ || ε - ε_θ( sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t)ε, t, c ) ||² ]
```

This is a simple regression objective. The network sees a noisy action chunk and
must recover the noise; sampling iteratively removes it.

### Sampling

DDPM ancestral sampling:

```text
x_{t-1} = (1/√α_t) (x_t - β_t/√(1-ᾱ_t) ε_θ) + σ_t z
```

DDIM sampling (deterministic, fewer steps):

```text
x_0^pred = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t
x_{t-1}  = √(ᾱ_{t-1}) x_0^pred + √(1-ᾱ_{t-1}) ε_θ
```

Use 100 training steps and 10 DDIM inference steps. This is the standard
Diffusion Policy configuration and is fast enough for CPU evaluation.

---

## 3. Scheduler

### `models/diffusion.py` (part 1)

```python
from __future__ import annotations

import math

import torch
from torch import nn


class DDPMScheduler:
    def __init__(self, num_train_timesteps: int = 100,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 schedule: str = "cosine"):
        self.num_train_timesteps = num_train_timesteps
        if schedule == "cosine":
            s = 0.008
            steps = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            f = torch.cos((steps / num_train_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.999)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        self.betas = betas.float()
        self.alphas = (1.0 - self.betas).float()
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).float()

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        ab = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def step(self, noise_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """One DDPM reverse step. t is a (B,) tensor of timesteps."""
        ab = self.alpha_bar.to(x_t.device)[t]
        ab_prev = torch.where(
            t > 0,
            self.alpha_bar.to(x_t.device)[(t - 1).clamp_min(0)],
            torch.ones_like(ab),
        )
        alpha = self.alphas.to(x_t.device)[t]
        beta = self.betas.to(x_t.device)[t]
        mean = (x_t - beta / (1.0 - ab).sqrt().view(-1, 1, 1) * noise_pred) \
            / alpha.sqrt().view(-1, 1, 1)
        var = beta * (1.0 - ab_prev) / (1.0 - ab)
        noise = torch.randn_like(x_t)
        return mean + var.sqrt().view(-1, 1, 1) * noise


class DDIMScheduler:
    def __init__(self, base: DDPMScheduler, num_inference_steps: int = 10):
        self.base = base
        self.num_inference_steps = num_inference_steps

    def timesteps(self, device=None):
        # Evenly spaced training timesteps, high to low, ending at 0.
        ts = torch.linspace(0, self.base.num_train_timesteps - 1,
                            self.num_inference_steps).round().long().flip(0)
        return ts.to(device)

    def step(self, noise_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        ab = self.base.alpha_bar.to(x_t.device)[t].view(-1, 1, 1)
        prev = self.base.alpha_bar.to(x_t.device)[(t - 1).clamp_min(0)].view(-1, 1, 1)
        ab_prev = torch.where((t > 0).view(-1, 1, 1), prev, torch.ones_like(ab))
        x0 = (x_t - (1.0 - ab).sqrt() * noise_pred) / ab.sqrt()
        x0 = x0.clamp(-1.0, 1.0)
        return ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * noise_pred
```

`DDIMScheduler` uses the same `alpha_bar` table. With 100 training steps and 10
inference steps, the timesteps are approximately `[99, 88, 77, …, 0]`; the last
step produces `x_0` because `ab_prev=1`.

**A common bug:** using `t-1` in DDIM instead of the previous *inference*
timestep. The formula above uses the previous training timestep's `ᾱ`; for a
strided inference schedule this is an approximation that works in practice. For
an exact implementation, precompute the `ᾱ` sequence of the inference timesteps
and index that.

---

## 4. The 1-D U-Net

### FiLM residual block

```python
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLMResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.film(cond)[:, :, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(torch.nn.functional.silu(h))
        return h + self.skip(x)
```

### U-Net block structure

```python
class Downsample1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim: int = 2, global_cond_dim: int = 320,
                 down_dims: tuple[int, ...] = (64, 128, 256),
                 diffusion_step_embed_dim: int = 128, kernel_size: int = 5,
                 n_groups: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 2, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        self.in_conv = nn.Conv1d(action_dim, down_dims[0], kernel_size,
                                 padding=kernel_size // 2)

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = down_dims[0]
        for i, dim in enumerate(down_dims):
            self.downs.append(nn.ModuleList([
                FiLMResBlock1d(in_ch, dim, cond_dim, n_groups),
                FiLMResBlock1d(dim, dim, cond_dim, n_groups),
            ]))
            in_ch = dim
            if i < len(down_dims) - 1:
                self.downsamples.append(Downsample1d(dim))

        self.mid1 = FiLMResBlock1d(down_dims[-1], down_dims[-1], cond_dim, n_groups)
        self.mid2 = FiLMResBlock1d(down_dims[-1], down_dims[-1], cond_dim, n_groups)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(down_dims))):
            dim = down_dims[i]
            prev_dim = down_dims[i + 1] if i + 1 < len(down_dims) else dim
            self.ups.append(nn.ModuleList([
                FiLMResBlock1d(dim * 2, dim, cond_dim, n_groups),
                FiLMResBlock1d(dim, dim, cond_dim, n_groups),
            ]))
            if i < len(down_dims) - 1:
                self.upsamples.append(Upsample1d(prev_dim, dim))

        self.out_norm = nn.GroupNorm(n_groups, down_dims[0])
        self.out_conv = nn.Conv1d(down_dims[0], action_dim, 1)

    def forward(self, sample, timestep, global_cond):
        # sample: (B, action_dim, Tp); global_cond: (B, global_cond_dim)
        cond = torch.cat([self.time_mlp(timestep), global_cond], dim=-1)
        x = self.in_conv(sample)
        skips = []

        for i, blocks in enumerate(self.downs):
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        for up_idx, blocks in enumerate(self.ups):
            if up_idx > 0:
                x = self.upsamples[up_idx - 1](x)
            x = torch.cat([x, skips.pop()], dim=1)
            for block in blocks:
                x = block(x, cond)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(x)))
```

### Channel arithmetic

For `down_dims=(64,128,256)` and `T_p=16`:

| Stage | Channels | Length |
|---|---|---|
| input | 2 | 16 |
| down 0 | 64 | 16 → 8 |
| down 1 | 128 | 8 → 4 |
| down 2 | 256 | 4 |
| mid | 256 | 4 |
| up 2 | 256 + 256 skip → 512 → 256 | 4 |
| up 1 | 256 → 128, +128 skip → 256 → 128 | 8 |
| up 0 | 128 → 64, +64 skip → 128 → 64 | 16 |
| output | 2 | 16 |

The `Upsample1d(prev_dim, dim)` changes channels, then the skip connection
concatenates `dim` more, so the first up block always receives `2 * dim`
channels. That is why its constructor is `FiLMResBlock1d(dim * 2, dim, ...)`.
Getting this arithmetic wrong produces a runtime shape error, which is preferable
to a silently wrong architecture.

---

## 5. Observation encoder and policy

### Image/proprio encoder

```python
class ImageObsEncoder(nn.Module):
    def __init__(self, state_dim: int = 18, obs_horizon: int = 2,
                 feature_dim: int = 128, state_feature: int = 64,
                 global_cond_dim: int = 320):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, feature_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(),
            nn.Linear(128, state_feature),
        )
        self.out = nn.Linear(obs_horizon * (feature_dim + state_feature),
                             global_cond_dim)

    def forward(self, state, image):
        # state: (B, To, 18) normalized; image: (B, To, 3, H, W) in [-1, 1]
        b, to = state.shape[:2]
        img = self.image_encoder(image.reshape(b * to, *image.shape[2:]))
        img = img.reshape(b, to, -1)
        st = self.state_encoder(state)
        return self.out(torch.cat([img, st], dim=-1).reshape(b, -1))
```

With `T_o=2`, `feature_dim=128`, `state_feature=64`, the concatenated input is
`2 * (128 + 64) = 384`, projected to 320. Keep these numbers in one config and
assert them in the policy constructor.

### Diffusion policy module

```python
class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int = 18, action_dim: int = 2,
                 obs_horizon: int = 2, horizon: int = 16,
                 down_dims: tuple[int, ...] = (64, 128, 256),
                 diffusion_step_embed_dim: int = 128,
                 num_train_timesteps: int = 100,
                 num_inference_steps: int = 10):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_normalizer = None
        self.action_normalizer = None
        self.encoder = ImageObsEncoder(state_dim, obs_horizon, global_cond_dim=320)
        self.unet = ConditionalUnet1D(
            action_dim, global_cond_dim=320, down_dims=down_dims,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
        )
        self.ddpm = DDPMScheduler(num_train_timesteps, schedule="cosine")
        self.ddim = DDIMScheduler(self.ddpm, num_inference_steps)

    def set_normalizers(self, state_norm, action_norm):
        self.state_normalizer = state_norm
        self.action_normalizer = action_norm

    def _prepare(self, obs):
        state = obs["state"].float()
        if self.state_normalizer is not None:
            state = self.state_normalizer(state)
        image = obs["image"]
        if image.dtype == torch.uint8:
            image = image.float() / 127.5 - 1.0
        else:
            image = image.float()
        return state, image

    def compute_loss(self, obs, actions, action_pad):
        state, image = self._prepare(obs)
        cond = self.encoder(state, image)
        x0 = self.action_normalizer(actions) if self.action_normalizer else actions
        b = x0.shape[0]
        t = torch.randint(0, self.ddpm.num_train_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.ddpm.add_noise(x0, noise, t)
        pred = self.unet(x_t.permute(0, 2, 1), t, cond).permute(0, 2, 1)
        err = (pred - noise) ** 2
        mask = action_pad.unsqueeze(-1).float()
        return (err * mask).sum() / (mask.sum() * self.action_dim + 1e-8)

    @torch.no_grad()
    def sample(self, obs, num_steps: int | None = None):
        state, image = self._prepare(obs)
        cond = self.encoder(state, image)
        b = state.shape[0]
        x = torch.randn(b, self.horizon, self.action_dim, device=state.device)
        steps = num_steps or self.ddim.num_inference_steps
        if steps != self.ddim.num_inference_steps:
            self.ddim.num_inference_steps = steps
        for t in self.ddim.timesteps(device=state.device):
            t_batch = t.expand(b)
            eps = self.unet(x.permute(0, 2, 1), t_batch, cond).permute(0, 2, 1)
            x = self.ddim.step(eps, x, t_batch)
        if self.action_normalizer is not None:
            x = self.action_normalizer.inverse(x)
        return x.clamp(-1.0, 1.0)
```

`sample` mutates `self.ddim.num_inference_steps` if a different step count is
requested. That is acceptable for a single-threaded evaluation but is a footgun
in a multi-run loop; pass a fresh scheduler or store per-call state instead. The
tutorial keeps the simple version and flags it.

### EMA

Diffusion training is noisy; an exponential moving average of the weights is
standard:

```python
import copy


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow.state_dict())
```

Evaluate the EMA copy, not the raw model.

---

## 6. Training

```python
dataset = DemoDataset(root, split="train", obs_horizon=2, action_horizon=16,
                      keys=("state", "image"))
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

model = DiffusionPolicy().to(device)
model.set_normalizers(
    StateNormalizer(torch.tensor(stats.state_mean), torch.tensor(stats.state_std)),
    ActionNormalizer(torch.tensor(stats.action_low), torch.tensor(stats.action_high)),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
ema = EMA(model, 0.999)

for step in range(1, steps + 1):
    batch = next(iter(loader))
    obs = {
        "state": batch["obs"]["state"].to(device),
        "image": augment_images(batch["obs"]["image"]).to(device),
    }
    loss = model.compute_loss(obs, batch["action"].to(device),
                              batch["action_pad"].to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema.update(model)
```

`augment_images` is the same helper as in chapter 10: random resized crop to
96×96 and mild colour jitter, applied to uint8 CHW tensors. Validate without
augmentation.

Save a checkpoint that the evaluation registry can rebuild:

```python
import dataclasses
import torch

torch.save({
    "algorithm": "dp",
    "model": ema.shadow.state_dict(),
    "model_config": dataclasses.asdict(cfg.model),
    "stats": stats.to_dict(),
    "obs_horizon": cfg.model.obs_horizon,
    "execute_horizon": cfg.eval.execute_horizon,
    "num_inference_steps": cfg.model.num_inference_steps,
}, run_dir / "best.pt")
```

Evaluating the EMA weights, not the raw model, is the default in Diffusion
Policy.

`configs/dp.yaml`:

```yaml
data:
  root: data/push_arm
  batch_size: 64
  num_workers: 0
model:
  obs_horizon: 2
  horizon: 16
  action_dim: 2
  down_dims: [64, 128, 256]
  diffusion_step_embed_dim: 128
  num_train_timesteps: 100
  num_inference_steps: 10
train:
  steps: 4000
  lr: 0.0001
  weight_decay: 0.000001
  grad_clip: 1.0
  ema_decay: 0.999
  seed: 0
  device: auto
  out: runs/dp
  val_every: 500
eval:
  episodes: 50
  execute_horizon: 8
```

Expected: **70–90%** success. A well-trained DP is usually close to ACT on this
task; the differences show up in multimodality, smoothness and robustness.

---

## 7. Inference wrapper

```python
# eval/policies.py
from collections import deque

import numpy as np
import torch

from imitation_learning_reproduction.training.normalizer import (
    ActionNormalizer, StateNormalizer,
)


class DPPolicy:
    def __init__(self, model, stats, obs_horizon=2, execute_horizon=8,
                 num_inference_steps=10, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.obs_horizon = obs_horizon
        self.execute_horizon = execute_horizon
        self.num_inference_steps = num_inference_steps
        self.history = deque(maxlen=obs_horizon)
        self.queue = []

        self.state_norm = StateNormalizer(
            torch.tensor(stats["state_mean"]), torch.tensor(stats["state_std"])
        ).to(device)
        self.action_norm = ActionNormalizer(
            torch.tensor(stats["action_low"]), torch.tensor(stats["action_high"])
        ).to(device)
        self.model.set_normalizers(self.state_norm, self.action_norm)

    def reset(self):
        self.history.clear()
        self.queue.clear()

    @torch.no_grad()
    def _sample(self, obs):
        self.history.append((obs["state"], obs["image"]))
        while len(self.history) < self.obs_horizon:
            self.history.append(self.history[0])
        states = np.stack([h[0] for h in self.history])
        images = np.stack([h[1] for h in self.history])
        state = torch.as_tensor(states, device=self.device).unsqueeze(0)
        image = torch.as_tensor(images, device=self.device)
        image = image.permute(0, 3, 1, 2).float().unsqueeze(0) / 127.5 - 1.0
        return self.model.sample({"state": state, "image": image},
                                 self.num_inference_steps)[0]

    def act(self, obs):
        if not self.queue:
            chunk = self._sample(obs)
            self.queue = list(chunk[: self.execute_horizon].cpu().numpy())
        return self.queue.pop(0).astype(np.float32)
```

Sampling is stochastic because the initial `x_T` is random. For reproducible
evaluation, seed the sampler per episode or fix `x_T`; report which choice you
made.

---

## 8. What to expect and how to debug

| Symptom | Likely cause |
|---|---|
| Loss NaN | Action normalization degenerate; logvar/ab indexing; image scale |
| Loss decreases, samples are noise | `alpha_bar` indexing wrong, or DDIM step uses wrong `t` |
| Samples are all the same | Condition ignored; check `cond` shape and FiLM |
| Good training loss, bad rollouts | Action horizon mismatch or unnormalization bug |
| Jitter at chunk boundaries | Reduce `execute_horizon`, add temporal smoothing |
| Slow evaluation | DDIM 100 steps; use 10 and report latency |
| EMA worse than raw model | Decay too high for the number of steps |

### Sanity checks

1. **Noise prediction on a toy linear dataset.** Generate `x0 ~ N(0, I)` and
   train the U-Net to predict noise with a constant condition. The loss should
   approach the irreducible noise variance.
2. **Scheduler inverse.** For a fixed `x0` and `ε`, compute `x_t`; at `t=0`,
   `x_t ≈ x0`.
3. **Overfit four action chunks.** The loss should approach zero; sampling should
   reproduce the chunks approximately.
4. **Condition sensitivity.** Change the goal position in the observation and
   check that the sampled action direction changes. If not, the encoder is not
   connected.

### Ablations

| Change | Expected effect |
|---|---|
| `num_inference_steps=100` | Slightly better samples, ~10× slower |
| `horizon=8` | Less smooth, faster; may be enough for this task |
| `horizon=32` | More context, harder to train with 100 demos |
| `obs_horizon=1` | Worse near contact; less temporal information |
| No EMA | Noisier evaluation |
| No image augmentation | Overfits the training backgrounds |

---

## 9. Tests

```python
# tests/test_diffusion.py
import torch

from imitation_learning_reproduction.models.diffusion import (
    ConditionalUnet1D, DDPMScheduler, DDIMScheduler,
)


def test_alpha_bar_monotonic():
    s = DDPMScheduler(num_train_timesteps=100)
    assert torch.all(s.alpha_bar[1:] <= s.alpha_bar[:-1])
    assert s.alpha_bar[0] > 0.99
    assert s.alpha_bar[-1] < 0.1


def test_add_noise_shapes():
    s = DDPMScheduler(num_train_timesteps=100)
    x0 = torch.randn(4, 16, 2)
    eps = torch.randn_like(x0)
    t = torch.tensor([0, 10, 50, 99])
    xt = s.add_noise(x0, eps, t)
    assert xt.shape == x0.shape
    assert torch.isfinite(xt).all()


def test_unet_shape():
    model = ConditionalUnet1D(action_dim=2, global_cond_dim=320,
                              down_dims=(16, 32, 64), n_groups=8)
    x = torch.randn(3, 2, 16)
    t = torch.randint(0, 100, (3,))
    cond = torch.randn(3, 320)
    assert model(x, t, cond).shape == (3, 2, 16)


def test_ddim_timesteps_descending():
    s = DDPMScheduler(num_train_timesteps=100)
    d = DDIMScheduler(s, num_inference_steps=10)
    ts = d.timesteps()
    assert len(ts) == 10
    assert torch.all(ts[1:] < ts[:-1])
    assert ts[-1] == 0
```

Add an overfit test on a small batch of real action chunks. If the loss does not
fall below a small threshold, stop before training the full model.

---

## 10. Checklist

- [ ] Scheduler `alpha_bar` is monotone and indexing is correct.
- [ ] U-Net output shape matches the action chunk.
- [ ] DDIM inference runs 10 steps and produces finite actions.
- [ ] Action normalization is inverted before the environment.
- [ ] Image normalization is consistent between training and evaluation.
- [ ] EMA is updated and used for evaluation.
- [ ] The model overfits a small batch.
- [ ] DP beats image BC on the same seeds.
- [ ] Latency is reported for 10 vs 100 inference steps.

### Exercises

1. Replace the 1-D U-Net with a transformer denoiser (Diffusion Transformer) and
   compare.
2. Add classifier-free guidance with a condition-dropout probability and test
   whether guidance improves success.
3. Implement flow matching and compare sample quality/latency with DDIM.
4. Add temporal ensembling across diffusion samples (sample twice and average).
5. Train with 25, 50, 100 demonstrations and plot sample efficiency.

Next: [12 · DP3](12_dp3.md).
