# 10 · ACT: Action Chunking with Transformers

> **Goal.** Implement ACT (Zhao et al., 2023): a transformer policy that observes
> a short history of images and proprioception and predicts a **chunk** of future
> actions. Add a CVAE latent to model demonstration multimodality, and use
> temporal ensembling at inference. Train it on the same dataset and evaluate it
> with the same protocol.
>
> **Prerequisites.** Chapters 03–05. Chapter 06 is not required, but you should
> understand action distributions.
>
> **Time.** 2 days.
>
> **Artifacts.** `models/act.py`, `scripts/train_act.py`, `configs/act.yaml`,
> `tests/test_act.py`.

---

## 1. Why action chunks?

Behaviour cloning predicts one action from one observation. If the policy makes a
small error, the next observation is off-distribution, the next prediction is
worse, and the error compounds. Action chunking changes the unit of prediction:
the policy predicts the next `T_p` actions at once, and the controller executes a
prefix of the chunk before replanning. This reduces the number of independent
decisions by roughly `T_p`, which reduces compounding error.

ACT combines three ideas:

1. **Transformer** over image tokens and proprioception, with learned action
   queries.
2. **CVAE** latent `z` that captures demonstration style/multimodality.
3. **Action chunking + temporal ensembling** at inference.

The original ACT uses a ResNet18 backbone and a DETR-style transformer. We use a
small convolutional tokenizer and a compact transformer so the model trains on a
CPU. The algorithmic pieces are the same.

---

## 2. The model

### 2.1 Shapes

Use `T_o = 2`, `T_p = 16`, image `96×96`, state 18, action 2.

```text
image (B, 2, 3, 96, 96)
  └─ conv tokenizer → (B, 2, 144, 128)
state (B, 2, 18)
  └─ linear → (B, 2, 128)
memory = image tokens + state tokens + positional embeddings
vae encoder: [CLS, state_t, action tokens] → μ, logσ  (training)
latent token z → memory
transformer encoder → transformer decoder (16 queries) → actions (B, 16, 2)
```

### 2.2 Convolutional tokenizer

Use three stride-2 convolutions and one stride-1 convolution:

```python
class ConvTokenizer(nn.Module):
    def __init__(self, in_ch: int = 3, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, d_model, 5, stride=2, padding=2),
            nn.GroupNorm(8, d_model), nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.GroupNorm(8, d_model), nn.SiLU(),
        )

    def forward(self, image):
        # image: (B*T_o, 3, 96, 96)
        h = self.net(image)              # (B*T_o, d, 12, 12)
        return h.flatten(2).transpose(1, 2)  # (B*T_o, 144, d)
```

At 96×96, the output is 12×12 = 144 tokens per frame. With two frames, the
transformer memory is 288 tokens plus two state tokens. That is small enough for
CPU attention.

### 2.3 Full model

```python
# models/act.py
from __future__ import annotations

import math

import torch
from torch import nn

from imitation_learning_reproduction.training.normalizer import (
    ActionNormalizer, StateNormalizer,
)


class ACT(nn.Module):
    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 2,
        obs_horizon: int = 2,
        chunk_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dim_feedforward: int = 512,
        latent_dim: int = 32,
        dropout: float = 0.1,
        image_size: int = 96,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.state_normalizer: StateNormalizer | None = None
        self.action_normalizer: ActionNormalizer | None = None

        self.tokenizer = ConvTokenizer(3, d_model)
        n_tokens = (image_size // 8) ** 2
        self.n_tokens = n_tokens
        self.image_pos = nn.Parameter(torch.zeros(1, obs_horizon * n_tokens, d_model))
        self.state_pos = nn.Parameter(torch.zeros(1, obs_horizon, d_model))
        nn.init.trunc_normal_(self.image_pos, std=0.02)
        nn.init.trunc_normal_(self.state_pos, std=0.02)

        self.state_proj = nn.Linear(state_dim, d_model)
        self.memory_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers, enable_nested_tensor=False)

        self.query_embed = nn.Embedding(chunk_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.action_head = nn.Linear(d_model, action_dim)

        # CVAE encoder: [CLS, state, action tokens]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_pos = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        self.vae_state_proj = nn.Linear(state_dim, d_model)
        vae_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.vae_encoder = nn.TransformerEncoder(vae_layer, enc_layers, enable_nested_tensor=False)
        self.vae_head = nn.Linear(d_model, latent_dim * 2)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.action_pos, std=0.02)

    def set_normalizers(self, state_norm: StateNormalizer,
                        action_norm: ActionNormalizer) -> None:
        self.state_normalizer = state_norm
        self.action_normalizer = action_norm

    def _encode_obs(self, obs):
        state = obs["state"]                     # (B, T_o, 18)
        image = obs["image"]                     # (B, T_o, 3, H, W)
        b, to = state.shape[:2]
        if self.state_normalizer is not None:
            state = self.state_normalizer(state)

        tokens = self.tokenizer(image.reshape(b * to, *image.shape[2:]))
        tokens = tokens.reshape(b, to * self.n_tokens, self.d_model)
        tokens = tokens + self.image_pos[:, : to * self.n_tokens]

        state_tokens = self.state_proj(state) + self.state_pos[:, :to]
        memory = torch.cat([tokens, state_tokens], dim=1)
        return self.memory_norm(memory)

    def _encode_vae(self, obs, actions, action_pad=None):
        state = obs["state"][:, -1]              # (B, 18)
        if self.state_normalizer is not None:
            state = self.state_normalizer(state)
        if self.action_normalizer is not None:
            actions = self.action_normalizer(actions)

        b, tp = actions.shape[:2]
        action_tokens = self.action_proj(actions) + self.action_pos[:, :tp]
        cls = self.cls_token.expand(b, 1, -1)
        state_token = self.vae_state_proj(state).unsqueeze(1)
        seq = torch.cat([cls, state_token, action_tokens], dim=1)  # (B, 2+Tp, d)

        if action_pad is not None:
            pad = torch.cat([
                torch.zeros(b, 2, dtype=torch.bool, device=actions.device),
                action_pad,
            ], dim=1)
        else:
            pad = None
        h = self.vae_encoder(seq, src_key_padding_mask=pad)
        mu_logvar = self.vae_head(h[:, 0])
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

    def forward(self, obs, actions=None, action_pad=None):
        memory = self._encode_obs(obs)
        b = memory.shape[0]

        if actions is not None:
            mu, logvar = self._encode_vae(obs, actions, action_pad)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            mu = logvar = None
            z = torch.zeros(b, self.latent_dim, device=memory.device)

        memory = torch.cat([memory, self.latent_proj(z).unsqueeze(1)], dim=1)
        memory = self.encoder(memory)

        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        h = self.decoder(queries, memory)
        pred = self.action_head(h)
        return pred, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
```

The CVAE encoder is a transformer over `[CLS, state, action tokens]`. Taking the
CLS output is a design choice; the original ACT encoder uses a similar pattern.
The latent is 32-D, small enough to avoid collapse but large enough to capture
style.

### 2.4 Why the CVAE?

A deterministic MSE/L1 policy trained on multimodal demonstrations predicts the
conditional mean, which can be an action no expert takes. The CVAE models a
distribution over latent styles; at training time the encoder sees the action
chunk, at inference the latent is set to zero (the prior mean). The KL term keeps
the latent distribution close to the prior.

If you omit the CVAE, ACT still works on this unimodal scripted dataset. Keep it
because it is part of the algorithm, and because it matters when you record human
data.

---

## 3. Data and training

Use `DemoDataset` with:

```python
dataset = DemoDataset(
    root, split="train",
    obs_horizon=2, action_horizon=16,
    keys=("state", "image"),
)
```

Batch shapes:

```text
obs["state"]       (B, 2, 18)
obs["image"]       (B, 2, 96, 96, 3) uint8
action             (B, 16, 2)
action_pad         (B, 16) bool
```

Image preprocessing:

```python
image = batch["obs"]["image"].permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0
# (B, 2, 3, 96, 96)
```

Apply augmentation per frame:

```python
import torchvision.transforms.v2 as v2

aug = v2.Compose([
    v2.RandomResizedCrop((96, 96), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])

def augment_images(images_uint8):
    # (B, T_o, H, W, 3) -> (B*T_o, 3, H, W) tensor, apply, reshape back
    b, to = images_uint8.shape[:2]
    x = images_uint8.permute(0, 1, 4, 2, 3).reshape(b * to, 3, *images_uint8.shape[2:4])
    x = aug(x)
    x = x.reshape(b, to, 3, 96, 96)
    return x.float() / 127.5 - 1.0
```

`RandomResizedCrop` needs a tensor; `v2` handles uint8 CHW. Do not augment the
validation set.

### Loss

```python
pred, mu, logvar = model(obs, action, action_pad)
err = (pred - model.action_normalizer(action)).abs()  # (B, Tp, 2)
mask = action_pad.unsqueeze(-1).float()
l1 = (err * mask).sum() / (mask.sum() * 2 + 1e-8)
kl = kl_divergence(mu, logvar)
loss = l1 + kl_weight * kl
```

Normalize the target actions before computing the loss, and keep the policy's
output normalized. At inference, unnormalize.

### KL annealing

Start with `kl_weight=0` and linearly increase to `max_kl=10` over the first
1,000 steps. This avoids the latent collapsing to the prior before the decoder
has learned anything useful. If the KL is huge and the L1 does not decrease,
reduce `max_kl` or delay the ramp.

### Training script skeleton

```python
model.set_normalizers(
    StateNormalizer(torch.tensor(stats.state_mean), torch.tensor(stats.state_std)),
    ActionNormalizer(torch.tensor(stats.action_low), torch.tensor(stats.action_high)),
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

for step in range(1, steps + 1):
    batch = next(iter(loader))
    obs = {
        "state": batch["obs"]["state"].to(device),
        "image": augment_images(batch["obs"]["image"]).to(device),
    }
    action = batch["action"].to(device)
    action_pad = batch["action_pad"].to(device)

    pred, mu, logvar = model(obs, action, action_pad)
    target = model.action_normalizer(action)
    err = (pred - target).abs()
    mask = action_pad.unsqueeze(-1).float()
    l1 = (err * mask).sum() / (mask.sum() * 2 + 1e-8)
    kl = kl_divergence(mu, logvar)
    kl_weight = min(1.0, step / 1000.0) * cfg.max_kl
    loss = l1 + kl_weight * kl

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

Use a batch size of 32–64 on CPU. For 100 demonstrations, 2,000–4,000 steps is
enough. Save the checkpoint with the model config, stats, and step:

```python
import dataclasses
import torch

torch.save({
    "algorithm": "act",
    "model": model.state_dict(),
    "model_config": dataclasses.asdict(cfg.model),
    "stats": stats.to_dict(),
    "obs_horizon": cfg.model.obs_horizon,
    "execute_horizon": cfg.eval.execute_horizon,
    "temporal_ensemble": cfg.eval.temporal_ensemble,
}, run_dir / "best.pt")
```

The evaluation registry in chapter 13 relies on `model_config` and `stats`, not
on guessing the architecture from the state dict.

The training script must set the normalizers before the first forward pass;
otherwise `model.action_normalizer` is `None`. The same normalizers are saved in
the checkpoint and restored by `ACTPolicy`.

`configs/act.yaml`:

```yaml
data:
  root: data/push_arm
  batch_size: 32
  num_workers: 0
model:
  obs_horizon: 2
  chunk_size: 16
  d_model: 128
  nhead: 4
  enc_layers: 2
  dec_layers: 2
  latent_dim: 32
  dropout: 0.1
train:
  steps: 3000
  lr: 0.0001
  weight_decay: 0.0001
  grad_clip: 1.0
  max_kl: 10.0
  kl_anneal_steps: 1000
  seed: 0
  device: auto
  out: runs/act
  val_every: 250
eval:
  episodes: 50
  execute_horizon: 8
  temporal_ensemble: true
```

---

## 4. Inference: chunks and temporal ensembling

### 4.1 Chunk execution

The simplest inference executes the first `k` actions of a predicted chunk, then
replans:

```python
if not self.queue:
    chunk = self.predict_chunk(obs)   # (Tp, 2)
    self.queue = list(chunk[: self.execute_horizon])
action = self.queue.pop(0)
```

This is fast: one forward pass every `k` steps. It is also less reactive; choose
`k` based on how fast the task changes. `k=8` is a reasonable default for 16-step
chunks.

### 4.2 Temporal ensembling

Temporal ensembling averages overlapping chunk predictions with exponential
weights. At every step, predict a new chunk and average all chunks that cover the
current time:

```python
# eval/policies.py
import math


class TemporalEnsembler:
    def __init__(self, k: float = 0.01):
        self.k = k
        self.buffer: list[tuple[int, "torch.Tensor"]] = []

    def reset(self):
        self.buffer.clear()

    def update(self, chunk, t: int):
        self.buffer.append((t, chunk))
        self.buffer = [(s, c) for s, c in self.buffer if s + len(c) > t]

    def action(self, t: int):
        num = None
        den = 0.0
        for start, chunk in self.buffer:
            idx = t - start
            if 0 <= idx < len(chunk):
                w = math.exp(-self.k * (t - start))
                contribution = w * chunk[idx]
                num = contribution if num is None else num + contribution
                den += w
        if num is None:
            raise RuntimeError("no chunk covers the current step")
        return num / den
```

Temporal ensembling is more expensive (one forward pass per step) and smoother.
It is a strong default for evaluation. Report which mode you used.

### 4.3 `ACTPolicy` wrapper

The evaluation wrapper owns the observation history, normalizers, and ensembler:

```python
# eval/policies.py
from collections import deque

import numpy as np
import torch

from imitation_learning_reproduction.training.normalizer import (
    ActionNormalizer, StateNormalizer,
)


class ACTPolicy:
    def __init__(self, model, stats, obs_horizon=2, execute_horizon=8,
                 temporal_ensemble=True, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.obs_horizon = obs_horizon
        self.execute_horizon = execute_horizon
        self.use_te = temporal_ensemble
        self.history = deque(maxlen=obs_horizon)
        self.queue = []
        self.ensemble = TemporalEnsembler()
        self.t = 0

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
        self.ensemble.reset()
        self.t = 0

    @torch.no_grad()
    def _predict(self, obs):
        self.history.append((obs["state"], obs["image"]))
        while len(self.history) < self.obs_horizon:
            self.history.append(self.history[0])
        states = np.stack([h[0] for h in self.history])
        images = np.stack([h[1] for h in self.history])
        state = torch.as_tensor(states, device=self.device).unsqueeze(0)
        image = torch.as_tensor(images, device=self.device)
        image = image.permute(0, 3, 1, 2).float().unsqueeze(0) / 127.5 - 1.0
        pred, _, _ = self.model({"state": state, "image": image})
        action = self.action_norm.inverse(pred[0])
        return action.clamp(-1.0, 1.0)

    def act(self, obs):
        if self.use_te:
            chunk = self._predict(obs)
            self.ensemble.update(chunk, self.t)
            action = self.ensemble.action(self.t)
            self.t += 1
            return action.cpu().numpy().astype(np.float32)

        if not self.queue:
            self.queue = list(self._predict(obs)[: self.execute_horizon])
        return self.queue.pop(0).cpu().numpy().astype(np.float32)
```

The history is filled by repeating the first observation when the episode starts;
this matches the dataset's left-padding convention.

---

## 5. Evaluation and expected results

```bash
uv run python scripts/train_act.py --config configs/act.yaml
uv run python scripts/evaluate.py --policy checkpoint --checkpoint runs/act/best.pt \
  --obs-mode image --episodes 50
```

Expected: **70–90%** success. ACT usually beats image BC clearly and is
competitive with Diffusion Policy. Temporal ensembling typically adds a few
points and smoother actions at the cost of inference time.

### What to inspect

- **L1 loss** should decrease; a flat L1 with decreasing KL means the decoder is
  ignoring the latent.
- **KL** should not collapse to zero immediately; if it does, the latent is
  unused. If it explodes, reduce `max_kl`.
- **Chunk boundaries**: execute chunks and plot the action sequence; look for
  jumps at replanning boundaries. Temporal ensembling should reduce them.
- **Action saturation**: normalized actions near ±1 for long stretches indicate
  the policy is pushing against the joint limits or the action scale is wrong.
- **Evaluation images**: save a GIF; check that the arm actually approaches the
  puck, not just the goal.

### Ablations

| Change | Expected effect |
|---|---|
| `chunk_size=1` | Degenerates to image BC; worse success |
| `chunk_size=32` | May improve smoothness, but harder to fit with 100 demos |
| `obs_horizon=1` | Less temporal information; worse near contact |
| No image augmentation | Overfits the 80 training episodes |
| No CVAE | Similar on scripted data; worse on multimodal human data |
| `execute_horizon=1` + temporal ensemble | Smoother, slower |
| `execute_horizon=16` | Faster, less reactive |

---

## 6. Debugging

| Symptom | Likely cause |
|---|---|
| Loss NaN | `logvar` overflow; clamp logvar or use `logsigmoid`; check action normalization |
| KL exactly zero | Decoder ignores latent; increase KL weight or check VAE input |
| Train loss low, success low | Compounding error or wrong image normalization |
| All actions constant | Action normalizer degenerate; check dataset action range |
| Chunk boundary jitter | No temporal ensembling; reduce execute horizon |
| Model overfits 4 demos but not 80 | Too few demos for the model size; shrink d_model or add augmentation |
| Inference very slow | Replanning every step with a large transformer; use chunk execution |

### Overfit test

Train on four demonstrations for 200 steps and check that the training L1 goes
below 0.02 (normalized action units). If it does not, the model or loss is wrong.
Then train on the full set; if the overfit test passes but full training does
not, the problem is data/augmentation, not the architecture.

---

## 7. Tests

```python
# tests/test_act.py
import torch

from imitation_learning_reproduction.models.act import ACT, kl_divergence


def test_forward_shapes():
    model = ACT(state_dim=18, action_dim=2, obs_horizon=2, chunk_size=16,
                d_model=64, nhead=4, enc_layers=1, dec_layers=1,
                dim_feedforward=128, image_size=96)
    obs = {
        "state": torch.randn(3, 2, 18),
        "image": torch.randn(3, 2, 3, 96, 96),
    }
    actions = torch.randn(3, 16, 2)
    pad = torch.zeros(3, 16, dtype=torch.bool)
    pred, mu, logvar = model(obs, actions, pad)
    assert pred.shape == (3, 16, 2)
    assert mu.shape == (3, 32)
    assert logvar.shape == (3, 32)
    pred2, mu2, logvar2 = model(obs)
    assert pred2.shape == (3, 16, 2)
    assert mu2 is None and logvar2 is None


def test_kl_nonnegative():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    assert kl_divergence(mu, logvar).item() == 0.0
```

Also test the temporal ensembler with two constant chunks and check the weighted
average; and test that padded positions do not contribute to the loss.

---

## 8. Checklist

- [ ] Model forward/backward works for `T_o=2`, `T_p=16`.
- [ ] Action normalization is applied to targets and inverted at inference.
- [ ] Image normalization is `[-1,1]` and consistent between train/eval.
- [ ] Padding mask excludes padded actions from the L1 loss.
- [ ] KL annealing is implemented and logged.
- [ ] Temporal ensembling is correct and resets between episodes.
- [ ] Overfit four demonstrations.
- [ ] ACT success beats image BC on the same seeds.
- [ ] Latency is reported with and without temporal ensembling.

### Exercises

1. Replace the conv tokenizer with a pretrained ResNet18 and compare sample
   efficiency.
2. Add a second camera and concatenate its tokens.
3. Predict an episode-end / padding flag as an auxiliary head (as in the original
   ACT) and see whether it helps.
4. Compare chunk execution and temporal ensembling on smoothness and latency.
5. Train on 25, 50, 100 demonstrations and plot success vs data.

Next: [11 · Diffusion Policy](11_diffusion_policy.md).
