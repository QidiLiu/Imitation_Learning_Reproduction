# 08 · Apprenticeship Learning

> **Goal.** Reproduce the feature-expectation matching algorithm of Abbeel & Ng
> (2004) on the pushing task: define features, compute the expert's feature
> expectation, solve the max-margin projection, and use PPO as the inner RL
> oracle. Compare it with AIRL and GAIL.
>
> **Prerequisites.** Chapters 03, 04, 06. Chapter 07 is helpful but not required.
>
> **Time.** 1–2 days.
>
> **Artifacts.** `rl/apprenticeship.py`, `scripts/train_apprenticeship.py`,
> `configs/apprenticeship.yaml`, `tests/test_apprenticeship.py`.

---

## 1. The idea

Apprenticeship Learning avoids learning a reward function pointwise. It asks for
a policy whose **feature expectations** match the expert's:

```text
μ(π) = E_{τ~π}[ Σ_t γ^t φ(s_t) ] ∈ R^d
μ_E  = E_{τ~D}[ Σ_t γ^t φ(s_t) ]
```

If `μ(π) ≈ μ_E` for a rich enough feature map, the policy behaves like the
expert under the features we care about. The algorithm alternates:

1. Find a reward direction `w` that makes the expert look better than every
   policy found so far.
2. Solve the MDP for that reward (here: PPO), producing a new policy.
3. Add its feature expectation to the set and repeat.

This is the **max-margin** view of IRL. It is closely related to MaxEnt IRL, but
the objective is a margin rather than a likelihood, and the output is a mixture
of policies rather than a single reward.

### Why features?

A raw 18-D state is not a feature map. We choose features that encode the task:

| Feature | Intuition |
|---|---|
| `1` | bias / constant shift, unidentifiable |
| `-||puck - goal||` | goal progress |
| `exp(-||puck - goal|| / 0.1)` | near-goal shaping |
| `-||ee - puck||` | contact approach |
| `exp(-||ee - puck|| / 0.1)` | near-contact shaping |
| `-||puck_velocity||` | discourage uncontrolled sliding |
| `-||last_action||` | smoothness |
| `1[dist < 0.05]` | success indicator |

These are not the true reward; they are the measurable quantities whose
expectations we want to match. Feature design is the main modelling choice in
Apprenticeship Learning.

---

## 2. The max-margin objective

Given expert expectation `μ_E` and a set of policies with expectations
`{μ_1, …, μ_i}`, find a direction `w` and margin `t` such that:

```text
max_{w, t}  t
s.t.        w·μ_E ≥ w·μ_j + t    for all j
            ||w||₂ ≤ 1
```

For a fixed `w`, the best policy is `argmax_π w·μ(π)`. The projection algorithm
of Abbeel & Ng avoids solving the full QP at every iteration:

```text
1. μ_E = expert feature expectation
2. π_0 = initial policy; μ_0 = μ(π_0)
3. for i = 1, 2, …:
     solve  min_λ || μ_E - Σ_j λ_j μ_j ||²
            s.t.  λ_j ≥ 0, Σ λ_j = 1
     w_i = μ_E - Σ_j λ_j μ_j
     if ||w_i|| is small: stop
     π_i = RL(reward = w_i · φ)
     μ_i = μ(π_i)
```

The QP projects `μ_E` onto the convex hull of the policies found so far. The
residual `w_i` is the direction in which the expert is still better than the
mixture. Training on `w_i · φ` pushes the next policy toward the expert.

The algorithm returns a **mixture** of the policies with weights `λ`. In
practice, people evaluate the latest policy or a uniform mixture; we will keep
the latest and report the feature-matching error as a diagnostic. A proper
mixture evaluation is an exercise.

### Finite horizon and discounting

Our episodes terminate on success and have variable length. Use a fixed discount
`γ = 0.99` and sum until termination. For the expert data, the episodes are
successful and terminate at success; for policy rollouts, the same rule applies.
Do not pad failed rollouts with the last state for feature matching; stop at the
terminal step.

---

## 3. Features and standardisation

### `rl/apprenticeship.py` (part 1)

```python
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "bias",
    "-puck_goal_dist",
    "near_goal",
    "-ee_puck_dist",
    "near_contact",
    "-puck_speed",
    "-action_norm",
    "success",
]


def push_features(state: np.ndarray) -> np.ndarray:
    """Map one 18-D state to an 8-D hand-designed feature vector."""
    puck = state[6:8]
    goal = state[10:12]
    ee = state[4:6]
    pv = state[8:10]
    action = state[12:14]
    d_goal = float(np.linalg.norm(puck - goal))
    d_ee = float(np.linalg.norm(ee - puck))
    return np.array([
        1.0,
        -d_goal,
        np.exp(-d_goal / 0.1),
        -d_ee,
        np.exp(-d_ee / 0.1),
        -float(np.linalg.norm(pv)),
        -float(np.linalg.norm(action)),
        float(d_goal < 0.05),
    ], dtype=np.float64)


class FeatureStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.maximum(std, 1e-6)

    @classmethod
    def fit(cls, states: np.ndarray) -> "FeatureStandardizer":
        feats = np.stack([push_features(s) for s in states])
        return cls(feats.mean(axis=0), feats.std(axis=0))

    def transform(self, state: np.ndarray) -> np.ndarray:
        return (push_features(state) - self.mean) / self.std

    def transform_batch(self, states: np.ndarray) -> np.ndarray:
        return np.stack([self.transform(s) for s in states])

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureStandardizer":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))
```

Standardising features is not part of the original algorithm, but it makes the
QP well conditioned and puts the reward on a comparable scale across dimensions.
The bias feature becomes zero after standardisation; that is fine, the constant
is not identifiable anyway.

### Feature expectations

```python
def episode_feature_expectation(states: np.ndarray, standardizer: FeatureStandardizer,
                                gamma: float = 0.99) -> np.ndarray:
    feats = standardizer.transform_batch(states)
    discounts = gamma ** np.arange(len(feats))
    return (discounts[:, None] * feats).sum(axis=0)


def expert_feature_expectation(episodes: list[dict],
                               standardizer: FeatureStandardizer,
                               gamma: float = 0.99) -> np.ndarray:
    mus = [episode_feature_expectation(ep["state"], standardizer, gamma)
           for ep in episodes if ep["success_episode"]]
    return np.mean(mus, axis=0)
```

`episode_feature_expectation` is a discounted sum, not an average. Keep the
discount convention identical for expert and policy rollouts, or the comparison
is meaningless.

---

## 4. The QP projection

We need:

```text
min_λ  0.5 λ^T G λ - b^T λ
s.t.   λ ≥ 0, 1^T λ = 1
G_ij = μ_i · μ_j
b_i  = μ_E · μ_i
```

The simplex projection is standard:

```python
def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_qp_simplex(mu_e: np.ndarray, mus: list[np.ndarray],
                     iters: int = 3000) -> np.ndarray:
    """Projected gradient descent for the convex-hull projection QP."""
    M = np.stack(mus, axis=0)                    # (k, d)
    G = M @ M.T                                  # (k, k)
    b = M @ mu_e                                 # (k,)
    L = np.linalg.norm(G, ord=2) + 1e-9
    lam = np.ones(len(mus), dtype=np.float64) / len(mus)
    for _ in range(iters):
        grad = G @ lam - b
        lam = project_simplex(lam - grad / L)
    return lam
```

Projected gradient descent converges for this convex QP. `3000` iterations on a
`k ≤ 8` problem is instantaneous. If you see oscillation, reduce the step to
`1 / (2L)`; the Lipschitz constant is an upper bound, not a tuning parameter.

### Tests for the QP

```python
def test_project_simplex_known():
    v = np.array([0.4, 0.1, -0.2])
    p = project_simplex(v)
    np.testing.assert_allclose(p.sum(), 1.0)
    assert (p >= 0).all()
    np.testing.assert_allclose(p, [0.63333333, 0.33333333, 0.03333333], atol=1e-8)


def test_qp_finds_expert_in_hull():
    mu_e = np.array([1.0, 0.0])
    mus = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    lam = solve_qp_simplex(mu_e, mus)
    np.testing.assert_allclose(lam, [1.0, 0.0], atol=1e-5)
```

The second test is the important one: if the expert is already in the convex
hull, the projection residual is zero and no further policy is needed.

---

## 5. The outer loop with PPO

### Reward injection

PPO's rollout buffer can accept any reward. For a direction `w`:

```python
buffer, obs = collect_rollout(envs, actor, obs_rms, obs, n_steps, device)
features = standardizer.transform_batch(buffer.obs.reshape(-1, 18))
rewards = (features @ w).reshape(buffer.rewards.shape)
buffer.set_rewards(rewards)
ppo.update(buffer)
```

Scale `w` to unit norm before training. The norm is the margin; the direction is
the reward. Use:

```python
w_unit = w / (np.linalg.norm(w) + 1e-8)
reward = features @ w_unit
reward = np.clip(reward, -5.0, 5.0)
```

Clipping keeps PPO stable when a feature has a large outlier. The bias feature is
zero after standardisation, so it does not affect the reward.

### Policy feature expectation

```python
def rollout_feature_expectation(env, policy, seeds, standardizer, gamma=0.99):
    mus = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        policy.reset()
        feats = []
        for _ in range(env.cfg.episode_steps):
            feats.append(standardizer.transform(obs["state"]))
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        discounts = gamma ** np.arange(len(feats))
        mus.append((discounts[:, None] * np.stack(feats)).sum(axis=0))
    return np.mean(mus, axis=0)
```

This is the same discount convention as the expert function. Evaluate it on the
same reset seeds you will use for the final result, or on a separate fixed set;
do not mix and match.

### The algorithm

```python
def train_apprenticeship(envs, eval_env, expert_episodes, cfg, device, out_dir):
    import torch
    from pathlib import Path
    from imitation_learning_reproduction.rl.ppo import (
        ActorCritic, PPOTrainer, RunningMeanStd, collect_rollout,
    )
    from imitation_learning_reproduction.eval.policies import PPOActorPolicy
    from imitation_learning_reproduction.eval.rollout import evaluate_policy

    states = np.concatenate([ep["state"] for ep in expert_episodes], axis=0)
    standardizer = FeatureStandardizer.fit(states)
    mu_e = expert_feature_expectation(expert_episodes, standardizer, cfg.gamma)

    obs, _ = envs.reset(seed=cfg.seed)
    obs_rms = RunningMeanStd(obs.shape[1:])
    obs_rms.update(states)
    actor = ActorCritic(obs.shape[1], 2, cfg.ppo.hidden).to(device)
    ppo = PPOTrainer(actor, obs_rms, cfg.ppo, device)

    mus = []
    best = -1.0
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for outer in range(cfg.outer_iters):
        # 1. Current policy's feature expectation (deterministic rollouts).
        policy = PPOActorPolicy(actor, obs_rms)
        mu = rollout_feature_expectation(
            eval_env, policy, [40_000 + i for i in range(cfg.feature_episodes)],
            standardizer, cfg.gamma,
        )
        mus.append(mu)

        # 2. Project the expert onto the convex hull and get the residual.
        lam = solve_qp_simplex(mu_e, mus, cfg.qp_iters)
        w = mu_e - lam @ np.stack(mus)
        margin = float(np.linalg.norm(w))
        print(f"outer {outer:02d} margin {margin:.4f}")

        if margin < cfg.margin_tol:
            break

        w_unit = w / (margin + 1e-8)

        # 3. Improve the policy under reward w·φ.
        for _ in range(cfg.inner_updates):
            buffer, obs = collect_rollout(envs, actor, obs_rms, obs,
                                          cfg.ppo.n_steps, device)
            feats = standardizer.transform_batch(buffer.obs.reshape(-1, 18))
            rewards = np.clip(feats @ w_unit, -5.0, 5.0).reshape(buffer.rewards.shape)
            buffer.set_rewards(rewards)
            ppo.update(buffer)
            obs_rms.update(buffer.obs.reshape(-1, *buffer.obs.shape[2:]))

        # 4. Evaluate and checkpoint.
        results, _ = evaluate_policy(
            eval_env, PPOActorPolicy(actor, obs_rms),
            [50_000 + i for i in range(cfg.eval_episodes)],
        )
        success = float(np.mean([r.success for r in results]))
        print(f"  success {success:.3f}")
        if success > best:
            best = success
            torch.save({
                "algorithm": "apprenticeship",
                "model": actor.state_dict(),
                "model_config": {"obs_dim": obs.shape[1], "action_dim": 2,
                                 "hidden": cfg.ppo.hidden},
                "obs_rms": obs_rms.state_dict(),
                "feature_standardizer": standardizer.state_dict(),
                "config": cfg.__dict__,
                "outer": outer,
                "margin": margin,
                "success": success,
            }, run_dir / "best.pt")
    return best
```

This is a faithful projection algorithm with two approximations:

1. The inner RL oracle is PPO with a finite budget, not an exact MDP solver.
2. The returned policy is the latest one, not the convex combination `λ`.

Both are standard in practical Apprenticeship Learning. Document them; do not
pretend the theoretical guarantee survives.

### Configuration

```python
from dataclasses import dataclass, field
from imitation_learning_reproduction.rl.ppo import PPOConfig


@dataclass
class ApprenticeshipConfig:
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(out="runs/apprenticeship"))
    gamma: float = 0.99
    outer_iters: int = 6
    inner_updates: int = 5
    qp_iters: int = 3000
    margin_tol: float = 0.02
    feature_episodes: int = 20
    eval_episodes: int = 20
    bc_init: str | None = None
    seed: int = 0
    out: str = "runs/apprenticeship"
```

`configs/apprenticeship.yaml` follows the same shape as `airl.yaml`:

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
  device: auto
  out: runs/apprenticeship
gamma: 0.99
outer_iters: 6
inner_updates: 5
qp_iters: 3000
margin_tol: 0.02
feature_episodes: 20
eval_episodes: 20
bc_init: null
seed: 0
out: runs/apprenticeship
```

---

## 6. What to expect

- The margin `||w||` should decrease over outer iterations. If it oscillates,
  the inner PPO is not solving the reward well, or the features are noisy.
- Success should improve over the first few outer iterations and then plateau.
  Expected range: **45–70%**, usually below ACT/DP but comparable to BC/AIRL.
- The feature-matching error `||μ_E - μ(π)||` is a better diagnostic than the
  success rate for debugging the algorithm. Log both.
- The QP λ is often sparse: the projection uses a few policies. Inspect it.

### Why Apprenticeship Learning can fail here

- The features are hand-designed and may not capture the true task.
- PPO is not an exact `argmax_π w·μ(π)` solver; the max-margin guarantee assumes
  an exact MDP solver.
- The expert is a scripted controller, not a human; its feature expectations may
  be narrow.
- The finite horizon and early termination make the discounted feature sums
  sensitive to episode length.

These are not bugs. They are the practical gap between the 2004 algorithm and a
continuous-control benchmark.

---

## 7. Debugging

| Symptom | Likely cause |
|---|---|
| Margin never decreases | PPO not learning the feature reward; scale/clip issue |
| QP λ is one-hot on the first policy | Later policies are worse; inner RL is failing |
| Feature expectations huge | Forgot to standardize or to discount |
| Reward range huge | `w` not normalized; clip is saturating |
| Policy succeeds but features don't match | Features are insufficient or the expert is inconsistent |
| Success drops after an outer iteration | PPO overwrote a good policy; keep the best checkpoint and consider a KL-to-previous penalty |

A KL penalty to the previous policy is a reasonable extension: it prevents the
oracle from forgetting a good behaviour when the reward direction changes.

---

## 8. Tests

```python
# tests/test_apprenticeship.py
import numpy as np

from imitation_learning_reproduction.rl.apprenticeship import (
    FeatureStandardizer, project_simplex, solve_qp_simplex,
)


def test_project_simplex_known():
    p = project_simplex(np.array([0.4, 0.1, -0.2]))
    np.testing.assert_allclose(p, [0.63333333, 0.33333333, 0.03333333], atol=1e-8)


def test_feature_standardizer_roundtrip():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(100, 18)).astype(np.float32)
    s = FeatureStandardizer.fit(states)
    feats = s.transform_batch(states)
    assert feats.shape == (100, 8)
    assert np.isfinite(feats).all()
    assert abs(feats.mean(axis=0)).max() < 0.2


def test_qp_projects_expert_in_hull():
    mu_e = np.array([1.0, 0.0])
    mus = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    lam = solve_qp_simplex(mu_e, mus)
    np.testing.assert_allclose(lam, [1.0, 0.0], atol=1e-5)
```

Add an integration test that runs two outer iterations with a tiny PPO budget
(`n_steps=64`, `n_envs=2`) and asserts that the margin is finite and the
checkpoint is written. Mark it `@pytest.mark.slow`.

---

## 9. Checklist

- [ ] Features are documented and standardized.
- [ ] Expert and policy feature expectations use the same discount.
- [ ] `project_simplex` passes the known-case test.
- [ ] QP projects an expert already in the hull to a zero residual.
- [ ] PPO accepts the feature reward through `buffer.set_rewards`.
- [ ] Margin decreases over outer iterations (or you can explain why not).
- [ ] Success is above random and reported with a confidence interval.
- [ ] The approximation (finite PPO oracle, latest policy) is documented.

### Exercises

1. Evaluate the actual mixture policy: sample a policy index from `λ`, run the
   episode with that actor, and compare to the latest policy.
2. Replace the linear features with random Fourier features and see whether the
   margin decreases faster.
3. Add a KL penalty to the previous policy in the PPO reward.
4. Solve the max-margin QP directly with a small `scipy.optimize` call and
   compare with projected gradient.
5. Compare Apprenticeship Learning and AIRL on the same seeds. Which has lower
   variance? Which is easier to debug?

Next: [09 · GAIL with PPO](09_gail.md).
