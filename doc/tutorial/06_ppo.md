# 06 · PPO: The RL Oracle Behind IRL, Apprenticeship Learning and GAIL

> **Goal.** Implement PPO well enough to serve as a reliable policy optimiser for
> the next three chapters. Validate it on the environment's hand-designed
> reward, then reuse the same trainer with learned rewards.
>
> **Prerequisites.** Chapters 02–05; comfortable with policy gradients.
>
> **Time.** 1–2 days.
>
> **Artifacts.** `rl/ppo.py`, `scripts/train_ppo.py`, `configs/ppo.yaml`,
> `tests/test_ppo.py`.

---

## 1. Why PPO appears in an imitation-learning tutorial

IRL, Apprenticeship Learning and GAIL all alternate between two problems:

1. **Evaluate or improve a policy** under a reward (explicit or learned).
2. **Update the reward or discriminator** using expert and policy samples.

The inner policy optimiser must be stable, off-the-shelf, and sample-efficient
enough to run several times inside an outer loop. PPO is a good default: it is
on-policy and simple, it handles continuous actions, and its clipped objective
is forgiving of a reward that changes between iterations.

PPO is **not** the imitation-learning contribution. If the PPO implementation is
broken, chapters 7–9 will fail in ways that look like algorithm failures. Test
PPO on the environment reward first.

---

## 2. The objective

PPO maximises a clipped surrogate objective:

```text
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
L_clip(θ) = E_t [ min( r_t(θ) A_t,
                       clip(r_t(θ), 1-ε, 1+ε) A_t ) ]
```

with an entropy bonus and a value loss:

```text
L = -L_clip + c_v * E[(V_θ(s_t) - R_t)²] - c_e * H[π_θ(·|s_t)]
```

Advantages come from generalised advantage estimation:

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ A_{t+1}
R_t = A_t + V(s_t)
```

For a truncated episode, bootstrap `V(s_{t+1})`; for a terminated episode, do
not. This distinction is the most common PPO bug in continuous-control projects.

### Tanh-squashed Gaussian

The environment expects actions in `[-1,1]`. Use a Gaussian in an unconstrained
space `u` and squash:

```text
a = tanh(u),  u ~ N(μ_θ(s), σ_θ²)
log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u_i))
```

The correction term is not optional if you care about the policy gradient. A
common shortcut is to sample a Gaussian and clip the action; it works on easy
tasks but optimises a different distribution. Implement the tanh version once.

---

## 3. Implementation

### 3.1 Running observation normalisation

PPO is sensitive to observation scale. The state contains metres, radians and
velocities; normalise with running statistics and save them in the checkpoint.

```python
# rl/ppo.py (part 1)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = (self.var * self.count + batch_var * batch_count
              + delta ** 2 * self.count * batch_count / total)
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_state_dict(cls, d: dict) -> "RunningMeanStd":
        obj = cls(np.asarray(d["mean"]).shape)
        obj.mean = np.asarray(d["mean"], dtype=np.float64)
        obj.var = np.asarray(d["var"], dtype=np.float64)
        obj.count = float(d["count"])
        return obj
```

### 3.2 Tanh-normal distribution

```python
class TanhNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.dist = Normal(mean, std.clamp_min(1e-3))

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.dist.rsample()
        a = torch.tanh(u)
        log_prob = self.dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        return a, log_prob.sum(dim=-1)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        a = a.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a)
        log_prob = self.dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.dist.mean)

    def entropy(self) -> torch.Tensor:
        # Normal entropy; the tanh correction is omitted. PPO only needs a
        # relative entropy bonus, and this approximation is standard.
        return self.dist.entropy().sum(dim=-1)
```

### 3.3 Actor-critic

```python
from imitation_learning_reproduction.models.common import mlp


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 18, action_dim: int = 2,
                 hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.actor = mlp(obs_dim, action_dim, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self(obs)
        dist = TanhNormal(mean, std)
        if deterministic:
            action = dist.mode()
            log_prob = dist.log_prob(action)
        else:
            action, log_prob = dist.sample()
        return action, log_prob, value

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        _, _, value = self(obs)
        return value
```

The policy and value networks are separate. A shared trunk is cheaper but can
couple the two losses in ways that are annoying to debug. Start separate.

### 3.4 Evaluation wrapper

`PPOActorPolicy` is what `scripts/evaluate.py` and the checkpoint loader use. It
normalises with the saved running statistics and returns the deterministic mean.

```python
# eval/policies.py (addition)
import numpy as np
import torch

from imitation_learning_reproduction.rl.ppo import RunningMeanStd


class PPOActorPolicy:
    def __init__(self, actor, obs_rms: RunningMeanStd):
        self.actor = actor
        self.obs_rms = obs_rms
        self.device = next(actor.parameters()).device

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32)
        state = self.obs_rms.normalize(state[None, :]).squeeze(0)
        tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        mean, _, _ = self.actor(tensor)
        return torch.tanh(mean).squeeze(0).cpu().numpy().astype(np.float32)
```


### 3.4 Rollout buffer and GAE

```python
@dataclass
class RolloutBuffer:
    obs: np.ndarray          # (T, N, 18) raw
    next_obs: np.ndarray     # (T, N, 18) final/next observation
    actions: np.ndarray      # (T, N, 2)
    logprobs: np.ndarray     # (T, N)
    values: np.ndarray       # (T, N)
    next_values: np.ndarray  # (T, N)
    rewards: np.ndarray      # (T, N)
    terminated: np.ndarray   # (T, N) bool
    truncated: np.ndarray    # (T, N) bool

    @property
    def shape(self):
        return self.obs.shape[:2]

    def set_rewards(self, rewards: np.ndarray) -> None:
        if rewards.shape != self.rewards.shape:
            raise ValueError(f"reward shape {rewards.shape} != {self.rewards.shape}")
        self.rewards = rewards.astype(np.float32)

    def compute_gae(self, gamma: float, lam: float):
        T, N = self.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            done = self.terminated[t] | self.truncated[t]
            delta = (
                self.rewards[t]
                + gamma * self.next_values[t] * (~self.terminated[t])
                - self.values[t]
            )
            last = delta + gamma * lam * (~done) * last
            advantages[t] = last
        returns = advantages + self.values
        return advantages.reshape(-1), returns.reshape(-1)
```

`next_values` must be computed with the value function at the **true next
observation**, not the auto-reset observation. In the collection function:

- For a terminated or truncated step, `infos["final_observation"][i]` is the
  final observation; use it.
- For a terminated step, set `next_value = 0`.
- For a truncated step, use `V(final_obs)`.
- For a non-terminal step, use `V(next_obs)`.

### 3.5 Collection

```python
def collect_rollout(envs, actor: ActorCritic, obs_rms: RunningMeanStd,
                    obs: np.ndarray, n_steps: int, device: torch.device):
    T, N = n_steps, envs.num_envs
    obs_buf = np.zeros((T, N, *obs.shape[1:]), dtype=np.float32)
    next_obs_buf = np.zeros_like(obs_buf)
    act_buf = np.zeros((T, N, 2), dtype=np.float32)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)
    next_val_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    term_buf = np.zeros((T, N), dtype=bool)
    trunc_buf = np.zeros((T, N), dtype=bool)

    for t in range(T):
        obs_buf[t] = obs
        obs_norm = torch.as_tensor(obs_rms.normalize(obs), device=device)
        with torch.no_grad():
            action, logp, value = actor.act(obs_norm)
        action_np = action.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = envs.step(action_np)

        final_obs = infos.get("final_observation")
        next_obs_true = next_obs.copy()
        if final_obs is not None:
            for i in range(N):
                if final_obs[i] is not None:
                    next_obs_true[i] = final_obs[i]

        next_obs_norm = torch.as_tensor(obs_rms.normalize(next_obs_true), device=device)
        with torch.no_grad():
            next_value = actor.value(next_obs_norm).cpu().numpy()
        next_value = np.where(terminated, 0.0, next_value)

        obs_buf[t] = obs
        next_obs_buf[t] = next_obs_true
        act_buf[t] = action_np
        logp_buf[t] = logp.cpu().numpy()
        val_buf[t] = value.cpu().numpy()
        next_val_buf[t] = next_value
        rew_buf[t] = rewards
        term_buf[t] = terminated
        trunc_buf[t] = truncated

        obs = next_obs

    buffer = RolloutBuffer(obs_buf, next_obs_buf, act_buf, logp_buf, val_buf,
                           next_val_buf, rew_buf, term_buf, trunc_buf)
    return buffer, obs
```

The collection function uses raw observations in the buffer. Normalisation
happens in the trainer, after the running statistics have been updated with the
rollout. This is important: if you normalise once during collection and update
the statistics afterwards, the stored observations and the update use different
normalisers.

### 3.6 PPO update

```python
@dataclass
class PPOConfig:
    iterations: int = 60
    n_envs: int = 8
    n_steps: int = 1024
    epochs: int = 10
    minibatch_size: int = 256
    hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    seed: int = 0
    device: str = "cuda"
    out: str = "runs/ppo"
    eval_every: int = 10
    eval_episodes: int = 20


class PPOTrainer:
    def __init__(self, actor: ActorCritic, obs_rms: RunningMeanStd, cfg,
                 device: torch.device):
        self.actor = actor
        self.obs_rms = obs_rms
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> dict:
        adv, ret = buffer.compute_gae(self.cfg.gamma, self.cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = self.obs_rms.normalize(buffer.obs.reshape(-1, 18))
        actions = buffer.actions.reshape(-1, 2)
        old_logp = buffer.logprobs.reshape(-1)
        values_old = buffer.values.reshape(-1)

        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)

        n = len(obs_t)
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        updates = 0

        for _ in range(self.cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = indices[start:start + self.cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, device=self.device)
                mean, std, value = self.actor(obs_t[mb_t])
                dist = TanhNormal(mean, std)
                logp = dist.log_prob(act_t[mb_t])
                ratio = torch.exp(logp - old_logp_t[mb_t])

                surr1 = ratio * adv_t[mb_t]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv_t[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[mb_t])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip).float().mean().item()
                for k, v in [("policy_loss", policy_loss.item()), ("value_loss", value_loss.item()),
                             ("entropy", entropy.item()), ("approx_kl", approx_kl),
                             ("clip_frac", clip_frac)]:
                    stats[k] += v
                updates += 1

            if stats["approx_kl"] / max(updates, 1) > self.cfg.target_kl:
                break

        return {k: v / max(updates, 1) for k, v in stats.items()}
```

The early-stopping check uses the running mean KL and can trigger in the middle
of an epoch. That is intentional; PPO's clipping is a soft constraint and a large
KL means the update moved too far.

---

## 4. Training script

`scripts/train_ppo.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from imitation_learning_reproduction.envs.push_arm import PushArmConfig, PushArmEnv
from imitation_learning_reproduction.envs.wrappers import make_state_env
from imitation_learning_reproduction.eval.policies import PPOActorPolicy
from imitation_learning_reproduction.eval.rollout import evaluate_policy
from imitation_learning_reproduction.rl.ppo import (
    ActorCritic, PPOConfig, PPOTrainer, RolloutBuffer, RunningMeanStd, collect_rollout,
)
from imitation_learning_reproduction.utils.config import apply_overrides, load_config
from imitation_learning_reproduction.utils.seed import resolve_device, seed_everything


# TrainConfig is PPOConfig from rl.ppo; it is shown above once.

@dataclass
class Config:
    train: PPOConfig = field(default_factory=PPOConfig)


def make_env(seed: int):
    def thunk():
        return make_state_env(PushArmConfig(obs_mode="state"), seed=seed)
    return thunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config, Config), args.set).train
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    envs = SyncVectorEnv([make_env(cfg.seed + i) for i in range(cfg.n_envs)])
    obs, _ = envs.reset(seed=cfg.seed)
    obs_rms = RunningMeanStd(obs.shape[1:])
    obs_rms.update(obs)
    actor = ActorCritic(obs.shape[1], 2).to(device)
    trainer = PPOTrainer(actor, obs_rms, cfg, device)

    eval_env = PushArmEnv(PushArmConfig(obs_mode="state"))
    run_dir = Path(cfg.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for iteration in range(1, cfg.iterations + 1):
        buffer, obs = collect_rollout(envs, actor, obs_rms, obs, cfg.n_steps, device)
        stats = trainer.update(buffer)
        # Update the normaliser *after* the PPO update so the old log-probs and
        # the new policy see the same observation scaling.
        obs_rms.update(buffer.obs.reshape(-1, *buffer.obs.shape[2:]))

        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            policy = PPOActorPolicy(actor, obs_rms)
            results, _ = evaluate_policy(
                eval_env, policy,
                [20_000 + i for i in range(cfg.eval_episodes)],
            )
            success = np.mean([r.success for r in results])
            print(f"iter {iteration:03d} steps {iteration * cfg.n_envs * cfg.n_steps} "
                  f"success {success:.3f} kl {stats['approx_kl']:.4f}")
            if success > best:
                best = success
                torch.save({
                    "algorithm": "ppo",
                    "model": actor.state_dict(),
                    "model_config": {"obs_dim": obs.shape[1], "action_dim": 2,
                                     "hidden": cfg.hidden},
                    "obs_rms": obs_rms.state_dict(),
                    "config": cfg.__dict__,
                    "iteration": iteration,
                    "success": success,
                }, run_dir / "best.pt")

    envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
```

`configs/ppo.yaml`:

```yaml
train:
  iterations: 60
  n_envs: 8
  n_steps: 1024
  epochs: 10
  minibatch_size: 256
  lr: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.02
  seed: 0
  device: cuda
  out: runs/ppo
  eval_every: 10
  eval_episodes: 20
```

Run:

```bash
uv run python scripts/train_ppo.py --config configs/ppo.yaml
```

With 8 envs, 1024 steps and 60 iterations, this collects about 491k transitions.
On a CUDA GPU it should finish in a few minutes; the exact time depends on
MuJoCo stepping, the CPU/GPU split, and the number of environments. The CPU
fallback is much slower and is only for smoke tests.

Expected: success rises from near zero to **60–85%**. If PPO is stuck:

- Check that `next_values` bootstraps on truncation and not on termination.
- Check that observations are raw in the buffer and normalised only in `update`.
- Check the reward scale. The environment reward is roughly `-0.2` per step plus
  a `+5` success bonus; that is reasonable.
- Check the action distribution: if `log_std` collapses early, increase
  `ent_coef` or lower the learning rate.
- Evaluate with the deterministic mean. A stochastic policy can look worse.

Once PPO passes 60%, it is good enough to be an oracle. Do not spend days
tuning it before moving on; chapters 7–9 need a reasonable optimiser, not a
state-of-the-art one.

---

## 5. Debugging PPO

| Symptom | Likely cause |
|---|---|
| Success stays at 0 while entropy collapses | `log_std` initialised too small, or policy loss dominates |
| KL explodes after first update | learning rate too high, or observations not normalised |
| Value loss diverges | returns not detached from value, or reward scale huge |
| Policy learns to spin/hold | reward hacking; inspect rollouts |
| Train stochastic success good, deterministic bad | distribution mean is not a good action; check tanh correction |
| Eval success much lower than collection | evaluation seeds differ or action normalisation mismatch |
| GAE advantages all zero | `done` mask applied incorrectly, or rewards all equal |
| `nan` in loss | `atanh(±1)` or division by zero in std; clamp actions and std |

### A useful diagnostic

Log `mean |advantage|`, `value mean`, `log_std`, `clip_frac`, and the reward
distribution. A healthy run has a non-zero advantage spread, a value estimate
that tracks returns, a `log_std` that does not collapse to zero, and a clip
fraction below ~0.3.

### Bootstrap test

Write a unit test with a two-step episode and hand-computed GAE. Do not trust a
PPO implementation whose GAE has not been checked against a small example.

---

## 6. Tests

```python
# tests/test_ppo.py
import numpy as np
import pytest
import torch

from imitation_learning_reproduction.rl.ppo import RolloutBuffer, TanhNormal


def test_tanh_normal_log_prob_matches_numeric():
    torch.manual_seed(0)
    mean = torch.zeros(2)
    std = torch.ones(2)
    dist = TanhNormal(mean, std)
    a = torch.tensor([0.2, -0.4])
    logp = dist.log_prob(a)
    # Change of variables: p(a) = p(u) / |det J|, u=atanh(a)
    from torch.distributions import Normal
    base = Normal(mean, std)
    u = torch.atanh(a)
    expected = (base.log_prob(u) - torch.log(1 - a ** 2)).sum()
    torch.testing.assert_close(logp, expected, atol=1e-6)


def test_gae_hand_computed():
    T, N = 2, 1
    buffer = RolloutBuffer(
        obs=np.zeros((T, N, 18), np.float32),
        next_obs=np.zeros((T, N, 18), np.float32),
        actions=np.zeros((T, N, 2), np.float32),
        logprobs=np.zeros((T, N), np.float32),
        values=np.array([[0.5], [0.2]], np.float32),
        next_values=np.array([[0.2], [0.0]], np.float32),
        rewards=np.array([[1.0], [0.0]], np.float32),
        terminated=np.array([[False], [True]]),
        truncated=np.array([[False], [False]]),
    )
    adv, ret = buffer.compute_gae(gamma=0.9, lam=1.0)
    # delta_1 = 0 + 0 - 0.2 = -0.2; A_1 = -0.2
    # delta_0 = 1 + 0.9*0.2 - 0.5 = 0.68; A_0 = 0.68 + 0.9*(-0.2) = 0.5
    assert adv[0] == pytest.approx(0.5)
    assert adv[1] == pytest.approx(-0.2)
```

Add a smoke test that runs one PPO update on random data and asserts the loss is
finite. It will not tell you the policy is good, but it catches shape errors.

---

## 7. Checklist

- [ ] `TanhNormal.log_prob` matches the change-of-variables formula.
- [ ] GAE matches a hand-computed two-step example.
- [ ] Truncated episodes bootstrap; terminated episodes do not.
- [ ] Observation normaliser is updated on raw rollout observations.
- [ ] PPO reaches at least 60% success on the environment reward.
- [ ] Deterministic evaluation is used for the success number.
- [ ] Checkpoint contains actor, observation statistics, and config.
- [ ] The same trainer can accept an external reward (chapter 07+).

### Exercises

1. Replace the separate actor/critic with a shared trunk and compare.
2. Add a reward normaliser (running std of returns) and see whether it helps.
3. Implement a recurrent policy (GRU) and compare on the same seeds.
4. Run three seeds and report mean ± std. How large is the seed variance?
5. Deliberately break the truncation bootstrap and observe the failure. This is
   the best way to remember why it matters.

Next: [07 · Inverse Reinforcement Learning](07_inverse_rl.md).
