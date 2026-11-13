# 07 · Inverse Reinforcement Learning: MaxEnt, an Exact Lab, and AIRL

> **Goal.** Understand inverse reinforcement learning (IRL) from the maximum-entropy
> objective, verify the math on a tiny tabular gridworld where the true reward is
> known, then scale the idea to the arm with **AIRL** (Adversarial Inverse
> Reinforcement Learning), a practical MaxEnt IRL method.
>
> **Prerequisites.** Chapter 06 (PPO). Chapter 03 (expert transitions).
>
> **Time.** 2–3 days.
>
> **Artifacts.** `envs/gridworld.py`, `rl/irl.py` (tabular MaxEnt),
> `rl/airl.py`, `scripts/train_airl.py`, `configs/airl.yaml`,
> `tests/test_irl.py`.

---

## 1. What IRL adds

Behaviour cloning learns a policy. IRL learns a **reward function** that explains
the demonstrations, then learns a policy for that reward. The reward is the
transferable object: it can be inspected, composed with new costs, and reused
when the dynamics or the task distribution change.

The catch is **reward ambiguity**. Many reward functions induce the same optimal
policy. A constant shift does not change anything; a potential-based shaping term
`γΦ(s') - Φ(s)` changes the reward but not the optimal policy (Ng et al., 1999).
“Recovering the true reward” is therefore only meaningful up to these
equivalences, and only if the demonstrations are near-optimal and the dynamics are
known or learned.

We will do two things:

1. **Exact MaxEnt IRL on a gridworld.** Small enough to solve the partition
   function exactly. This is the reference implementation that lets you check the
   signs, the soft Bellman equation and the gradient.
2. **AIRL on the arm.** A neural reward and an adversarial discriminator make
   MaxEnt IRL tractable in continuous, high-dimensional state spaces.

The gridworld is not a detour. It is the unit test for the theory. It is a
tabular NumPy lab and therefore runs on the CPU; the arm-based AIRL training in
this chapter uses the same CUDA default as every other training recipe.

---

## 2. Maximum-entropy IRL

### 2.1 The model

MaxEnt IRL (Ziebart et al., 2008) assumes demonstrations are drawn from a
Boltzmann distribution over trajectories:

```text
p_θ(τ) = (1 / Z_θ) exp( r_θ(τ) ),
Z_θ = ∫ exp( r_θ(τ) ) dτ,
r_θ(τ) = Σ_t r_θ(s_t, a_t).
```

The expert is assumed to be stochastically optimal: among trajectories with equal
reward, all are equally likely. Maximising the likelihood of the demonstrations
gives:

```text
∇_θ L = E_{τ~D}[ ∇_θ r_θ(τ) ] - E_{τ~p_θ}[ ∇_θ r_θ(τ) ].
```

The second expectation is the policy induced by the reward. If the policy is
soft-optimal, its state-action distribution matches the Boltzmann distribution,
and the gradient is a **feature-matching** difference:

```text
∇_θ L = E_D[φ(s,a)] - E_{π_θ}[φ(s,a)]   (linear reward r = θ·φ).
```

This is the same feature-matching idea that reappears in Apprenticeship Learning
(chapter 08), but here it comes from a probabilistic model rather than a
max-margin objective.

### 2.2 Soft Bellman equations

For a reward `r(s,a)` and discount `γ`, the soft Q-value and soft value are:

```text
Q(s,a) = r(s,a) + γ E_{s'}[ V(s') ]
V(s)   = log Σ_a exp Q(s,a)          (discrete actions)
```

and the soft-optimal policy is `π(a|s) = exp(Q(s,a) - V(s))`. For continuous
actions, the sum becomes a log-integral and is generally intractable. That is the
reason for the adversarial approximation below.

For the gridworld, everything is finite: run soft value iteration to convergence,
compute the policy, compute the expected state visitation, and take a gradient
step on `θ`.

### 2.3 Why the partition function is hard

`Z_θ` integrates over all trajectories. For a 60-step episode with a 2-D
continuous action, that is an intractable integral. Guided cost learning and
adversarial IRL avoid it by learning a discriminator that estimates the
density ratio between expert and policy trajectories. AIRL is the version we
implement.

---

## 3. Exact lab: a 5×5 gridworld

### 3.1 The MDP

`src/imitation_learning_reproduction/envs/gridworld.py`:

```python
from __future__ import annotations

import numpy as np


class GridWorld:
    """A deterministic 5x5 grid with one obstacle and one goal."""

    def __init__(self, size: int = 5, goal: int = 24, obstacles: tuple[int, ...] = (12,)):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.goal = goal
        self.obstacles = set(obstacles)
        self.start = 0
        self.transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.features = np.eye(self.n_states)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.transitions[s, a, self._next(s, a)] = 1.0

        self.gt_reward = np.full(self.n_states, -0.02)
        self.gt_reward[self.goal] = 1.0
        for s in self.obstacles:
            self.gt_reward[s] = -1.0
        self.gt_reward[self.goal] = 1.0

    def _next(self, state: int, action: int) -> int:
        r, c = divmod(state, self.size)
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1
        r = int(np.clip(r, 0, self.size - 1))
        c = int(np.clip(c, 0, self.size - 1))
        nxt = r * self.size + c
        return state if nxt in self.obstacles else nxt
```

The obstacle is absorbing in the sense that entering it is impossible; the agent
stays in the previous cell. The exact choice does not matter for the lab.

### 3.2 Soft value iteration and expected visitation

```python
def soft_value_iteration(reward: np.ndarray, transitions: np.ndarray,
                         gamma: float = 0.9, iters: int = 1000, tol: float = 1e-8):
    """Return (Q, V, policy) for a discrete MaxEnt MDP."""
    n_states, n_actions = reward.shape[0], transitions.shape[1]
    V = np.zeros(n_states, dtype=np.float64)
    for _ in range(iters):
        Q = reward[:, None] + gamma * np.einsum("san,n->sa", transitions, V)
        V_new = np.logaddexp.reduce(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = reward[:, None] + gamma * np.einsum("san,n->sa", transitions, V)
    logits = Q - Q.max(axis=1, keepdims=True)
    policy = np.exp(logits)
    policy /= policy.sum(axis=1, keepdims=True)
    return Q, V, policy


def expected_visitation(policy: np.ndarray, transitions: np.ndarray,
                        gamma: float = 0.9, start: int = 0) -> np.ndarray:
    """Discounted state visitation μ(s) for a stochastic policy."""
    n_states = policy.shape[0]
    P_pi = np.einsum("sa,san->sn", policy, transitions)
    start_dist = np.zeros(n_states)
    start_dist[start] = 1.0
    return np.linalg.solve(np.eye(n_states) - gamma * P_pi.T, start_dist)


def feature_expectation(policy: np.ndarray, transitions: np.ndarray,
                        features: np.ndarray, gamma: float = 0.9,
                        start: int = 0) -> np.ndarray:
    return expected_visitation(policy, transitions, gamma, start) @ features
```

`logaddexp.reduce` is the numerically stable log-sum-exp over actions. The
policy is the softmax of Q. For a linear reward `r = features @ θ`, the feature
expectation is the state visitation times the feature matrix.

### 3.3 Expert demonstrations

Generate demonstrations from the soft-optimal policy under the true reward, with
a small amount of sampling:

```python
def sample_expert_trajectories(env: GridWorld, n_traj: int = 100,
                               horizon: int = 15, seed: int = 0):
    _, _, policy = soft_value_iteration(env.gt_reward, env.transitions, gamma=0.9)
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(n_traj):
        s = env.start
        traj = []
        for _ in range(horizon):
            a = rng.choice(env.n_actions, p=policy[s])
            traj.append((s, a))
            s = int(np.argmax(env.transitions[s, a]))
        trajectories.append(traj)
    return trajectories


def trajectory_feature_expectation(trajectories, features, gamma=0.9):
    mu = np.zeros(features.shape[1])
    for traj in trajectories:
        for t, (s, _) in enumerate(traj):
            mu += (gamma ** t) * features[s]
    return mu / len(trajectories)
```

### 3.4 MaxEnt IRL

```python
def maxent_irl(env: GridWorld, expert_trajectories, gamma: float = 0.9,
               lr: float = 0.05, iters: int = 500):
    """Linear-reward MaxEnt IRL. Returns θ and the reward history."""
    mu_expert = trajectory_feature_expectation(expert_trajectories, env.features, gamma)
    theta = np.zeros(env.n_states)
    history = []

    for it in range(iters):
        reward = env.features @ theta
        _, _, policy = soft_value_iteration(reward, env.transitions, gamma)
        mu_policy = feature_expectation(policy, env.transitions, env.features, gamma)
        grad = mu_expert - mu_policy
        theta += lr * grad
        history.append({
            "iter": it,
            "grad_norm": float(np.linalg.norm(grad)),
            "theta": theta.copy(),
        })

    return theta, history
```

Why does this work? The expert feature expectation is fixed. The policy's
feature expectation is the model's prediction. Gradient ascent increases the
reward on features the expert visits more than the policy — exactly the
likelihood gradient. The reward is identifiable only up to a constant, so do not
compare `theta` elementwise to the ground truth; compare **rankings** and
**differences**.

### 3.5 What to test

```python
# tests/test_irl.py
import numpy as np
import pytest

from imitation_learning_reproduction.envs.gridworld import (
    GridWorld, expected_visitation, maxent_irl, sample_expert_trajectories,
    soft_value_iteration,
)


def test_soft_policy_is_distribution():
    env = GridWorld()
    _, V, policy = soft_value_iteration(env.gt_reward, env.transitions)
    assert np.isfinite(V).all()
    np.testing.assert_allclose(policy.sum(axis=1), 1.0, atol=1e-8)
    assert (policy >= 0).all()


def test_visitation_matches_simulation():
    env = GridWorld()
    _, _, policy = soft_value_iteration(env.gt_reward, env.transitions)
    mu = expected_visitation(policy, env.transitions)
    # Sum of discounted occupancy equals 1/(1-gamma) for a continuing chain.
    assert mu.sum() == pytest.approx(1.0 / (1.0 - 0.9), rel=1e-5)


def test_maxent_recovers_goal_ordering():
    env = GridWorld()
    trajectories = sample_expert_trajectories(env, n_traj=200, horizon=20, seed=0)
    theta, _ = maxent_irl(env, trajectories, gamma=0.9, lr=0.05, iters=800)
    reward = env.features @ theta
    visited = sorted({s for traj in trajectories for s, _ in traj})
    assert reward[env.goal] > reward[env.start] + 0.1
    assert max(visited, key=lambda s: reward[s]) == env.goal
    corr = np.corrcoef([reward[s] for s in visited],
                       [env.gt_reward[s] for s in visited])[0, 1]
    assert corr > 0.5
```

The obstacle is in the grid but never visited by the expert, so its reward is
**not identifiable** from these demonstrations. MaxEnt IRL can assign it any
value without changing the likelihood. The test therefore checks the visited
states and the goal/start ordering, not the obstacle. This is a concrete lesson
about reward ambiguity: if the expert never visits a state, no IRL method can
tell you what the reward there should be.

If the test fails:

- Check the sign of the gradient.
- Check that `transitions[s,a]` is a distribution over next states.
- Check the discount in `expected_visitation`.
- Check that the expert uses the **true** reward, not the learned one.

Plot `theta` as a 5×5 heatmap and compare it with `gt_reward`. The shape should
be similar up to a constant shift.

---

## 4. Scaling up: adversarial IRL and AIRL

Exact MaxEnt IRL needs the partition function. AIRL (Fu et al., 2018) replaces it
with a discriminator. The derivation is:

1. The optimal MaxEnt policy satisfies `π(a|s) ∝ exp(Q(s,a))`.
2. The density ratio between expert and policy is
   `ρ_E(s,a) / ρ_π(s,a) = exp(r(s,a) - V(s))` up to constants.
3. A discriminator `D(s,a) = σ(f(s,a))` trained to distinguish expert from
   policy estimates that log-ratio: `f(s,a) ≈ r(s,a) - V(s)`.
4. Train the policy to maximise `f`; train `f` to distinguish expert from policy.
   At equilibrium, `f` is the learned reward (up to a state-only constant).

AIRL adds a shaping term so the recovered reward is not entangled with the
discriminator's state-only bias:

```text
f(s,a,s') = g(s) + γ h(s') - h(s)
D(s,a,s') = σ(f(s,a,s'))
```

`g` is the reward network; `h` is a potential. The shaping term
`γh(s') - h(s)` is policy-invariant, so `g` is the part that matters for reward
recovery. In practice, train the policy on `f` and log both `f` and `g`.

### 4.1 The reward network

```python
# rl/airl.py (part 1)
from __future__ import annotations

import torch
from torch import nn

from imitation_learning_reproduction.models.common import mlp


class RewardNet(nn.Module):
    def __init__(self, state_dim: int = 18, hidden: int = 256, gamma: float = 0.99):
        super().__init__()
        self.g = mlp(state_dim, 1, (hidden, hidden))
        self.h = mlp(state_dim, 1, (hidden, hidden))
        self.gamma = gamma

    def forward(self, state, action, next_state, done):
        # action is accepted for API symmetry; this reward is state-only.
        g = self.g(state).squeeze(-1)
        h = self.h(state).squeeze(-1)
        h_next = self.h(next_state).squeeze(-1)
        f = g + self.gamma * h_next * (1.0 - done) - h
        return f, g
```

A state-only reward is a deliberate choice. The puck/goal/arm state contains
enough information for this task, and a state-only reward is easier to interpret.
If you want an action-dependent reward, concatenate the action into `g` and
remove the action argument from the comment.

### 4.2 Configuration

```python
from dataclasses import dataclass, field

from imitation_learning_reproduction.rl.ppo import PPOConfig


@dataclass
class AIRLConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    outer_iters: int = 5
    inner_updates: int = 8
    disc_steps: int = 3
    disc_lr: float = 3e-4
    disc_hidden: int = 256
    label_smooth: float = 0.1
    reward_clip: float = 10.0
    batch_size: int = 1024
    bc_init: str | None = None
    seed: int = 0
    out: str = "runs/airl"
    eval_episodes: int = 20
```

### 4.3 Training loop

```python
# rl/airl.py (part 2)
import numpy as np
import torch.nn.functional as F

from imitation_learning_reproduction.eval.policies import PPOActorPolicy
from imitation_learning_reproduction.rl.ppo import (
    ActorCritic, PPOTrainer, RunningMeanStd, collect_rollout,
)


class AIRLTrainer:
    def __init__(self, actor, reward_net, obs_rms, ppo_cfg, airl_cfg, device):
        self.actor = actor
        self.reward_net = reward_net
        self.obs_rms = obs_rms
        self.ppo = PPOTrainer(actor, obs_rms, ppo_cfg, device)
        self.cfg = airl_cfg
        self.device = device
        self.disc_opt = torch.optim.Adam(
            reward_net.parameters(), lr=airl_cfg.disc_lr, weight_decay=1e-4
        )

    def _norm(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(x), device=self.device)

    def update_discriminator(self, expert, buffer, rng: np.random.Generator) -> dict:
        n = len(expert["state"])
        T, N = buffer.obs.shape[:2]
        s_p = self._norm(buffer.obs.reshape(-1, 18))
        a_p = torch.as_tensor(buffer.actions.reshape(-1, 2), device=self.device)
        sn_p = self._norm(buffer.next_obs.reshape(-1, 18))
        d_p = torch.as_tensor(
            (buffer.terminated | buffer.truncated).reshape(-1).astype(np.float32),
            device=self.device,
        )
        acc_e = acc_p = 0.0
        for _ in range(self.cfg.disc_steps):
            idx = rng.integers(0, n, size=min(self.cfg.batch_size, n))
            idx_t = torch.as_tensor(idx, device=self.device)
            s_e = self._norm(expert["state"][idx])
            a_e = torch.as_tensor(expert["action"][idx], device=self.device)
            sn_e = self._norm(expert["next_state"][idx])
            d_e = torch.as_tensor(expert["done"][idx], device=self.device)

            f_e, _ = self.reward_net(s_e, a_e, sn_e, d_e)
            f_p, _ = self.reward_net(s_p, a_p, sn_p, d_p)
            target_e = torch.full_like(f_e, 1.0 - self.cfg.label_smooth)
            target_p = torch.full_like(f_p, self.cfg.label_smooth)
            loss = (F.binary_cross_entropy_with_logits(f_e, target_e)
                    + F.binary_cross_entropy_with_logits(f_p, target_p))

            self.disc_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
            self.disc_opt.step()

            acc_e = (f_e > 0).float().mean().item()
            acc_p = (f_p < 0).float().mean().item()

        with torch.no_grad():
            f_p, g_p = self.reward_net(s_p, a_p, sn_p, d_p)
            rewards = torch.clamp(f_p, -self.cfg.reward_clip, self.cfg.reward_clip)
        buffer.set_rewards(rewards.reshape(T, N).cpu().numpy())
        return {"disc_acc_expert": acc_e, "disc_acc_policy": acc_p,
                "f_policy_mean": float(f_p.mean()), "g_policy_mean": float(g_p.mean())}
```

The discriminator is trained on **all** policy transitions from the last rollout
and a random expert minibatch. This is standard but note that the policy samples
are from the previous policy, not the updated one; the outer loop corrects that.

### 4.4 The outer loop

```python
def train_airl(envs, eval_env, expert, cfg, device, out_dir):
    import torch
    from pathlib import Path

    from imitation_learning_reproduction.eval.policies import PPOActorPolicy
    from imitation_learning_reproduction.eval.rollout import evaluate_policy
    from imitation_learning_reproduction.rl.ppo import (
        ActorCritic, RunningMeanStd, collect_rollout,
    )

    obs, _ = envs.reset(seed=cfg.seed)
    obs_rms = RunningMeanStd(obs.shape[1:])
    obs_rms.update(expert["state"])
    actor = ActorCritic(obs.shape[1], 2, cfg.ppo.hidden).to(device)

    # Optional BC warm start: copy the BC MLP weights into the PPO actor.
    if cfg.bc_init:
        ckpt = torch.load(cfg.bc_init, map_location="cpu", weights_only=False)
        if ckpt.get("kind") == "state":
            state = {k.removeprefix("net."): v for k, v in ckpt["model"].items()
                     if k.startswith("net.")}
            actor.actor.load_state_dict(state, strict=True)

    reward_net = RewardNet(obs.shape[1], cfg.disc_hidden, cfg.ppo.gamma).to(device)
    trainer = AIRLTrainer(actor, reward_net, obs_rms, cfg.ppo, cfg, device)
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

        policy = PPOActorPolicy(actor, obs_rms)
        results, _ = evaluate_policy(eval_env, policy,
                                     [30_000 + i for i in range(cfg.eval_episodes)])
        success = float(np.mean([r.success for r in results]))
        print(f"outer {outer:02d} success {success:.3f} "
              f"disc_e {d_stats['disc_acc_expert']:.2f} "
              f"disc_p {d_stats['disc_acc_policy']:.2f} "
              f"f {d_stats['f_policy_mean']:.3f} "
              f"kl {ppo_stats['approx_kl']:.4f}")
        if success > best:
            best = success
            torch.save({
                "algorithm": "airl",
                "model": actor.state_dict(),
                "model_config": {"obs_dim": obs.shape[1], "action_dim": 2,
                                 "hidden": cfg.ppo.hidden},
                "reward_net": reward_net.state_dict(),
                "obs_rms": obs_rms.state_dict(),
                "config": cfg.__dict__,
                "outer": outer,
                "success": success,
            }, run_dir / "best.pt")
    return best
```

You need to assemble the script around this function:

1. Load `TransitionDataset` from `data/push_arm`.
2. Build `expert = {state, action, next_state, done}` NumPy arrays.
3. Build vectorised envs (`SyncVectorEnv`) and a raw eval env.
4. Construct the `AIRLConfig`.
5. Call `train_airl`.
6. Close envs.

The full script is deliberately omitted because it is the same boilerplate as
`train_ppo.py` with a different trainer. If you copy-paste `train_ppo.py`, delete
the environment-reward assumptions and use `train_airl`.

### 4.5 `configs/airl.yaml`

```yaml
ppo:
  iterations: 0          # unused by AIRL; the outer loop controls updates
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
  out: runs/airl
  eval_every: 0
  eval_episodes: 20
outer_iters: 5
inner_updates: 8
disc_steps: 3
disc_lr: 0.0003
disc_hidden: 256
label_smooth: 0.1
reward_clip: 10.0
batch_size: 1024
bc_init: null
seed: 0
out: runs/airl
eval_episodes: 20
```

`iterations` is unused because the outer loop calls `PPOTrainer.update` directly.
Leaving it in the nested config keeps the config schema uniform; document it
instead of silently ignoring it.

Run:

```bash
uv run python scripts/train_airl.py --config configs/airl.yaml
```

---

## 5. What to expect and how to debug

### Healthy AIRL

- Discriminator accuracy on expert and policy hovers around **0.5–0.8**. If it
  is 1.0 for both, the discriminator has memorised the data and the reward is
  uninformative.
- `f_policy_mean` does not explode; clamp is active only occasionally.
- Success rises from the BC warm start (if used) or from random.
- The learned `g(s)` should correlate with `-||puck - goal||` and with pushing
  progress. Plot `g` against distance-to-goal on expert states.

### Common failures

| Symptom | Likely cause |
|---|---|
| Success stays at 0 | Discriminator reward is constant or wrong sign |
| Reward explodes | No clamp/normalisation; discriminator overfits |
| Discriminator accuracy 1.0 | Too many D steps, too high D lr, no label smoothing |
| Policy ignores puck | Reward uses only absolute state and finds a shortcut |
| Policy oscillates | Reward changes too fast relative to PPO's value function |
| Learned reward is all potential | `g` collapsed; shaping term dominates; regularise `h` |
| AIRL worse than BC | Expected on easy tasks; BC is a strong baseline |

### Diagnostics that actually help

1. **Freeze the reward and train PPO longer.** If success is still zero, the
   reward is bad or PPO is broken.
2. **Evaluate the discriminator on held-out expert states.** If accuracy is 1.0
   there, it is overfitting.
3. **Visualise `g` along an expert trajectory.** It should increase as the puck
   approaches the goal, not just as time passes.
4. **Compare to GAIL.** If GAIL works and AIRL does not, the shaping term or
   reward recovery is the difference.

Reward recovery is hard to evaluate quantitatively. The honest metric is the
**policy success rate**, plus a qualitative check that the learned reward prefers
goal-reaching states. Do not claim “we recovered the true reward” from a single
heatmap.

---

## 6. Tests

In addition to the gridworld tests:

```python
def test_reward_net_shaping_is_potential_based():
    import torch
    from imitation_learning_reproduction.rl.airl import RewardNet

    net = RewardNet(state_dim=4, hidden=16, gamma=0.9)
    s = torch.randn(8, 4)
    a = torch.randn(8, 2)
    sn = torch.randn(8, 4)
    done = torch.zeros(8)
    f, g = net(s, a, sn, done)
    assert f.shape == (8,)
    assert g.shape == (8,)
    # With h=0 the shaping term vanishes; not generally true after training.
```

A stronger test initialises `g` and `h` to zero and checks `f=0`; then sets `h`
to a constant and checks `f=g`. This verifies the `(1-done)` mask and the sign of
the shaping term.

---

## 7. Checklist

- [ ] Gridworld soft value iteration is stable and produces a valid policy.
- [ ] Expected visitation sums to `1/(1-γ)` on a continuing chain.
- [ ] MaxEnt IRL recovers goal > start and goal > obstacle in reward ordering.
- [ ] AIRL reward net implements `g + γh(s') - h(s)` with a terminal mask.
- [ ] Discriminator accuracy stays below 0.9; reward is clamped.
- [ ] AIRL success is above random and ideally near BC.
- [ ] You can explain why exact MaxEnt IRL is intractable on the arm.
- [ ] You can explain the difference between AIRL and GAIL (chapter 09).

### Exercises

1. Implement **guided cost learning** with importance sampling and compare its
   variance to AIRL on the gridworld.
2. Add a gradient penalty to the AIRL discriminator and measure stability.
3. Use a state-action reward `g(s,a)` instead of a state-only reward. Does the
   policy improve? Is the reward interpretable?
4. Train AIRL with and without a BC warm start. How much does the warm start
   matter for sample efficiency?
5. Evaluate the learned reward by training a fresh PPO policy on it from scratch.
   Does it reproduce the AIRL policy?

Next: [08 · Apprenticeship Learning](08_apprenticeship.md).
