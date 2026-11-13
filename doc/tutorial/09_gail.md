# 09 · GAIL with PPO

> **Goal.** Implement Generative Adversarial Imitation Learning (GAIL) on the
> pushing task. A discriminator learns to distinguish expert transitions from
> policy transitions; its output becomes the PPO reward. Compare GAIL with AIRL
> and with BC, and learn why adversarial imitation is powerful but finicky.
>
> **Prerequisites.** Chapters 03, 04, 06. Chapter 07 helps with the adversarial
> view of IRL.
>
> **Time.** 1–2 days.
>
> **Artifacts.** `rl/gail.py`, `scripts/train_gail.py`, `configs/gail.yaml`,
> `tests/test_gail.py`.

---

## 1. Occupancy measures

A policy induces a discounted state-action occupancy measure:

```text
ρ_π(s,a) = π(a|s) Σ_t γ^t p_t(s)
```

where `p_t(s)` is the probability of being in state `s` at step `t` under `π`
starting from the initial-state distribution. Two policies behave similarly when
their occupancy measures are close.

GAIL (Ho & Ermon, 2016) minimises the Jensen-Shannon divergence between the
expert occupancy `ρ_E` and the policy occupancy `ρ_π`:

```text
min_π max_D  E_{ρ_π}[ log D(s,a) ] + E_{ρ_E}[ log(1 - D(s,a)) ]
```

The inner maximiser is a binary classifier. For a fixed policy, the optimal
discriminator is:

```text
D*(s,a) = ρ_E(s,a) / (ρ_E(s,a) + ρ_π(s,a))
```

and the outer objective becomes the JS divergence. At the optimum,
`D* = 0.5` everywhere: the discriminator cannot tell the two distributions
apart.

### The reward

GAIL does not need a separate reward network. Use the discriminator to define a
reward that is high when the policy looks like the expert:

```text
r(s,a) = -log(1 - D(s,a)) = softplus(f(s,a)),   f = discriminator logit
```

Some implementations use `log D - log(1-D) = f` directly; it differs by a
constant and a monotone transform. `softplus(f)` is the standard GAIL reward.
Clip it to avoid a huge value when the discriminator is very confident.

### GAIL vs AIRL

| | GAIL | AIRL |
|---|---|---|
| Discriminator input | `(s,a)` | `(s,a,s')` with shaping |
| Reward | `softplus(f)` | `f = g(s) + γh(s') - h(s)` |
| Learns an explicit reward? | No | Yes (`g`) |
| Recovers reward up to potential? | No | Yes, in principle |
| Stability | Sensitive to D strength | Similar, plus shaping |
| Typical use | Policy imitation | Reward transfer / interpretability |

Both are adversarial. AIRL can be seen as GAIL with a reward decomposition that
removes the state-only bias and makes reward recovery possible.

---

## 2. The discriminator

```python
# rl/gail.py (part 1)
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from imitation_learning_reproduction.models.common import mlp


class GAILDiscriminator(nn.Module):
    def __init__(self, state_dim: int = 18, action_dim: int = 2,
                 hidden: int = 256):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1, (hidden, hidden))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


def gail_reward(logits: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    """r = -log(1 - D) = softplus(logit)."""
    return torch.clamp(F.softplus(logits), max=clip)
```

The discriminator is a small MLP. It sees normalized states and raw normalized
actions. Do not feed it unnormalized states while PPO sees normalized states;
the two networks would be learning from different coordinate systems.

### Gradient penalty

GAIL is prone to a discriminator that becomes too confident. A gradient penalty
(WGAN-GP style) on the discriminator input stabilises training:

```python
def gradient_penalty(disc, state, action, expert_state, expert_action,
                     rng: torch.Generator | None = None) -> torch.Tensor:
    alpha = torch.rand(state.shape[0], 1, device=state.device, generator=rng)
    mixed_state = (alpha * state + (1 - alpha) * expert_state).requires_grad_(True)
    mixed_action = alpha * action + (1 - alpha) * expert_action
    logits = disc(mixed_state, mixed_action)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=mixed_state,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
```

Penalising only the state gradient is a common approximation and much cheaper
than penalising both state and action. Use `gp_coef=10.0` and apply it every
discriminator step, or every other step if it is too slow.

---

## 3. Training loop

The structure is the same as AIRL, with the discriminator and reward swapped:

```python
# rl/gail.py (part 2)
import numpy as np

from imitation_learning_reproduction.eval.policies import PPOActorPolicy
from imitation_learning_reproduction.eval.rollout import evaluate_policy
from imitation_learning_reproduction.rl.ppo import (
    ActorCritic, PPOTrainer, RunningMeanStd, collect_rollout,
)


class GAILTrainer:
    def __init__(self, actor, disc, obs_rms, ppo_cfg, gail_cfg, device):
        self.actor = actor
        self.disc = disc
        self.obs_rms = obs_rms
        self.ppo = PPOTrainer(actor, obs_rms, ppo_cfg, device)
        self.cfg = gail_cfg
        self.device = device
        self.opt = torch.optim.Adam(disc.parameters(), lr=gail_cfg.disc_lr,
                                    weight_decay=1e-4)

    def _norm(self, x):
        return torch.as_tensor(self.obs_rms.normalize(x), device=self.device)

    def update_discriminator(self, expert, buffer, rng):
        n = len(expert["state"])
        T, N = buffer.obs.shape[:2]
        s_p = self._norm(buffer.obs.reshape(-1, 18))
        a_p = torch.as_tensor(buffer.actions.reshape(-1, 2), device=self.device)
        acc_e = acc_p = 0.0

        for _ in range(self.cfg.disc_steps):
            idx = rng.integers(0, n, size=min(self.cfg.batch_size, n))
            idx_t = torch.as_tensor(idx, device=self.device)
            s_e = self._norm(expert["state"][idx])
            a_e = torch.as_tensor(expert["action"][idx], device=self.device)

            logits_e = self.disc(s_e, a_e)
            logits_p = self.disc(s_p, a_p)
            target_e = torch.full_like(logits_e, 1.0 - self.cfg.label_smooth)
            target_p = torch.full_like(logits_p, self.cfg.label_smooth)
            loss = (F.binary_cross_entropy_with_logits(logits_e, target_e)
                    + F.binary_cross_entropy_with_logits(logits_p, target_p))

            if self.cfg.gp_coef > 0:
                # The penalty needs matched batch sizes; sample a policy subset.
                idx_p = rng.integers(0, len(s_p), size=len(s_e))
                idx_p_t = torch.as_tensor(idx_p, device=self.device)
                loss = loss + self.cfg.gp_coef * gradient_penalty(
                    self.disc, s_p[idx_p_t], a_p[idx_p_t], s_e, a_e
                )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
            self.opt.step()

            acc_e = (logits_e > 0).float().mean().item()
            acc_p = (logits_p < 0).float().mean().item()

        with torch.no_grad():
            logits_p = self.disc(s_p, a_p)
            rewards = gail_reward(logits_p, self.cfg.reward_clip)
        buffer.set_rewards(rewards.reshape(T, N).cpu().numpy())
        return {"disc_acc_expert": acc_e, "disc_acc_policy": acc_p,
                "reward_mean": float(rewards.mean())}
```

### Outer loop

```python
def train_gail(envs, eval_env, expert, cfg, device, out_dir):
    import torch
    from pathlib import Path

    obs, _ = envs.reset(seed=cfg.seed)
    obs_rms = RunningMeanStd(obs.shape[1:])
    obs_rms.update(expert["state"])
    actor = ActorCritic(obs.shape[1], 2, cfg.ppo.hidden).to(device)

    if cfg.bc_init:
        ckpt = torch.load(cfg.bc_init, map_location="cpu", weights_only=False)
        if ckpt.get("kind") == "state":
            state = {k.removeprefix("net."): v for k, v in ckpt["model"].items()
                     if k.startswith("net.")}
            actor.actor.load_state_dict(state, strict=True)

    disc = GAILDiscriminator(obs.shape[1], 2, cfg.disc_hidden).to(device)
    trainer = GAILTrainer(actor, disc, obs_rms, cfg.ppo, cfg, device)
    rng = np.random.default_rng(cfg.seed)
    best = -1.0
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for outer in range(cfg.outer_iters):
        for inner in range(cfg.inner_updates):
            buffer, obs = collect_rollout(envs, actor, obs_rms, obs,
                                          cfg.ppo.n_steps, device)
            d_stats = trainer.update_discriminator(expert, buffer, rng)
            ppo_stats = trainer.ppo.update(buffer)
            obs_rms.update(buffer.obs.reshape(-1, *buffer.obs.shape[2:]))

        results, _ = evaluate_policy(
            eval_env, PPOActorPolicy(actor, obs_rms),
            [60_000 + i for i in range(cfg.eval_episodes)],
        )
        success = float(np.mean([r.success for r in results]))
        print(f"outer {outer:02d} success {success:.3f} "
              f"disc_e {d_stats['disc_acc_expert']:.2f} "
              f"disc_p {d_stats['disc_acc_policy']:.2f} "
              f"reward {d_stats['reward_mean']:.3f} "
              f"kl {ppo_stats['approx_kl']:.4f}")
        if success > best:
            best = success
            torch.save({
                "algorithm": "gail",
                "model": actor.state_dict(),
                "model_config": {"obs_dim": obs.shape[1], "action_dim": 2,
                                 "hidden": cfg.ppo.hidden},
                "discriminator": disc.state_dict(),
                "obs_rms": obs_rms.state_dict(),
                "config": cfg.__dict__,
                "outer": outer,
                "success": success,
            }, run_dir / "best.pt")
    return best
```

`update_discriminator` recomputes the reward for **all** policy transitions after
the discriminator update. This is the correct ordering: the reward used by PPO is
the one from the current discriminator, not the one from the previous outer
iteration.

### Configuration

```python
from dataclasses import dataclass, field
from imitation_learning_reproduction.rl.ppo import PPOConfig


@dataclass
class GAILConfig:
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(out="runs/gail"))
    outer_iters: int = 5
    inner_updates: int = 8
    disc_steps: int = 3
    disc_lr: float = 3e-4
    disc_hidden: int = 256
    label_smooth: float = 0.1
    gp_coef: float = 10.0
    reward_clip: float = 10.0
    batch_size: int = 1024
    bc_init: str | None = None
    seed: int = 0
    out: str = "runs/gail"
    eval_episodes: int = 20
```

`configs/gail.yaml`:

```yaml
ppo:
  n_envs: 8
  n_steps: 1024
  epochs: 10
  minibatch_size: 256
  hidden: [256, 256]
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
  out: runs/gail
outer_iters: 5
inner_updates: 8
disc_steps: 3
disc_lr: 0.0003
disc_hidden: 256
label_smooth: 0.1
gp_coef: 10.0
reward_clip: 10.0
batch_size: 1024
bc_init: null
seed: 0
out: runs/gail
eval_episodes: 20
```

---

## 4. What to expect

A healthy GAIL run:

- Discriminator accuracy around **0.5–0.8** on both classes. This is the main
  indicator that the adversarial game is balanced.
- Reward mean decreases as the policy becomes more expert-like, but not to zero;
  the discriminator always has some signal.
- Success around **55–80%**, often a little below ACT/DP and comparable to AIRL.
- BC warm start helps sample efficiency; pure GAIL from random can work but is
  slower on this task.

### Failure modes

| Symptom | Likely cause |
|---|---|
| D accuracy 1.0 immediately | D too strong: reduce `disc_steps`/`disc_lr`, add label smoothing/gradient penalty |
| D accuracy 0.5 from the start and stays | D too weak: increase capacity/lr, or check the reward sign |
| Reward saturates at the clip | D overconfident; reduce clip or D strength |
| Success collapses after an outer iteration | PPO overfits a transient D; reduce inner updates, add BC/KL regulariser |
| Policy just stands still | D gives high reward for a static state; inspect expert/policy occupancy |
| GAIL beats AIRL but reward is uninterpretable | Expected: GAIL has no explicit reward |

### Diagnose the adversarial balance

Plot discriminator accuracy on expert and policy batches over time. If expert
accuracy rises while policy accuracy falls, the discriminator is winning; the
policy reward becomes a constant and PPO stops learning. If both hover around
0.5, the game is balanced but the reward may be too weak to drive improvement.
GAIL lives in this narrow band; that is why it needs careful tuning.

---

## 5. Tests

```python
# tests/test_gail.py
import torch

from imitation_learning_reproduction.rl.gail import (
    GAILDiscriminator, gail_reward, gradient_penalty,
)


def test_discriminator_separable():
    torch.manual_seed(0)
    disc = GAILDiscriminator(state_dim=4, action_dim=2, hidden=32)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-2)
    expert_s = torch.randn(128, 4) + 2.0
    expert_a = torch.randn(128, 2)
    policy_s = torch.randn(128, 4) - 2.0
    policy_a = torch.randn(128, 2)
    for _ in range(100):
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    disc(expert_s, expert_a), torch.ones(128))
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    disc(policy_s, policy_a), torch.zeros(128)))
        opt.zero_grad(); loss.backward(); opt.step()
    assert (disc(expert_s, expert_a) > 0).float().mean() > 0.9
    assert (disc(policy_s, policy_a) < 0).float().mean() > 0.9


def test_gail_reward_matches_definition():
    logits = torch.tensor([0.0, 1.0, -1.0])
    d = torch.sigmoid(logits)
    expected = -torch.log(1.0 - d)
    torch.testing.assert_close(gail_reward(logits), expected, atol=1e-6)


def test_gradient_penalty_is_finite():
    torch.manual_seed(0)
    disc = GAILDiscriminator(state_dim=4, action_dim=2, hidden=16)
    s = torch.randn(8, 4, requires_grad=True)
    a = torch.randn(8, 2)
    se = torch.randn(8, 4)
    ae = torch.randn(8, 2)
    gp = gradient_penalty(disc, s, a, se, ae)
    assert torch.isfinite(gp)
    assert gp.item() >= 0.0
```

The separable test checks that the discriminator can learn at all. The
gradient-penalty test catches a missing `create_graph=True` or a detached input.

---

## 6. Checklist

- [ ] Discriminator takes normalized state and raw action.
- [ ] GAIL reward is `softplus(logit)` and is clipped.
- [ ] D is updated before the PPO reward is computed.
- [ ] Gradient penalty (or another stabiliser) is enabled.
- [ ] D accuracy is logged for both classes.
- [ ] GAIL success is above random and ideally near BC.
- [ ] BC warm start is optional and documented.
- [ ] You can explain the difference between GAIL and AIRL.

### Exercises

1. Remove the gradient penalty and chart D accuracy and success. What breaks?
2. Replace `softplus(logit)` with `logit` and compare stability.
3. Use spectral normalisation instead of a gradient penalty.
4. Compare GAIL with a BC warm start vs from scratch on the same seeds.
5. Add a small environment-reward term to the GAIL reward and measure whether it
   helps or merely hides adversarial instability.

Next: [10 · ACT](10_act.md).
