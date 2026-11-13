# 04 · Evaluation Protocol and Metrics

> **Goal.** Fix the evaluation protocol before training anything: the same reset
> seeds, the same success criterion, the same number of episodes, deterministic
> actions, confidence intervals, and a small set of robustness variants. Every
> algorithm in chapters 5–12 is measured with this harness.
>
> **Prerequisites.** Chapters 02–03.
>
> **Time.** Half a day.
>
> **Artifacts.** `eval/rollout.py`, `eval/metrics.py`, `eval/policies.py`,
> `scripts/evaluate.py`, `tests/test_eval.py`.

---

## 1. Why a protocol chapter?

Imitation-learning results are easy to accidentally inflate:

- evaluating on the training demonstrations;
- choosing the best of five seeds and reporting only that one;
- using a different reset distribution for each method;
- letting a policy act stochastically at evaluation while another acts
  deterministically;
- reporting “success” when the puck passes near the goal for one frame;
- comparing success rates on 10 episodes without uncertainty.

None of these is malicious. They happen because evaluation is usually written
last, under time pressure. Write it now, while there is nothing to tune.

The protocol is deliberately boring:

| Item | Value |
|---|---|
| Evaluation seeds | `10_000 + i`, `i = 0 … n-1` |
| Episodes | 50 for the final table; 20 for iteration |
| Policy mode | deterministic (mean action or one fixed sample) |
| Success | puck centre within 5 cm of goal at any step |
| Episode end | success or 60 steps |
| Environment | default colours, `top` camera, no distractors |
| Test split | never used for tuning |
| Report | success rate + 95% Wilson interval, mean final/min distance |

A method may be evaluated with more episodes for a final report, but never with
a different seed set. “Same seeds” means the exact same list, not a new random
sample of the same size.

### Which split?

- **Train:** fit the model.
- **Val:** choose hyperparameters and early stopping.
- **Test:** run once per method after all choices are frozen.

The evaluation harness runs on reset seeds, not on episodes from the dataset.
The test split is a set of *demonstrations* held out from training. The reset
seeds are a separate held-out set of *initial conditions*. Use both: the test
split measures imitation of unseen demonstrations, and the reset seeds measure
closed-loop task performance.

---

## 2. The policy interface

Every policy, classical or neural, implements:

```python
class Policy(Protocol):
    def reset(self) -> None:
        """Called at the start of every episode."""

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return an action in [-1, 1]^2. No batch dimension."""
```

The observation is the dictionary returned by `PushArmEnv`, not a normalized
tensor. The policy owns its normalization and its action queue. This is the key
design decision that keeps the evaluation loop algorithm-agnostic.

Examples:

- `RandomPolicy`: samples from the action space.
- `ScriptedExpert`: wraps `ScriptedPusher`.
- `BCPolicy`: normalizes `obs["state"]`, runs the MLP, unnormalizes the action.
- `ACTPolicy`: buffers the last `T_o` images, runs the transformer, and pops one
  action from its chunk queue.
- `DiffusionPolicy`: same queue, with a diffusion sample.
- `PPOActor`: normalizes state with the running statistics saved in the
  checkpoint and returns the deterministic mean.

`reset()` is not optional. A chunking policy that forgets to clear its queue will
execute actions from the previous episode.

### `eval/policies.py` (start with the trivial policies)

```python
from __future__ import annotations

from typing import Protocol

import numpy as np

from imitation_learning_reproduction.envs.expert import ScriptedPusher


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


class RandomPolicy:
    def __init__(self, action_dim: int = 2, seed: int = 0):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)


class ExpertPolicy:
    def __init__(self):
        self.expert = ScriptedPusher()

    def reset(self) -> None:
        self.expert.reset()

    def act(self, obs) -> np.ndarray:
        return self.expert.act(obs["state"])
```

The expert baseline is a sanity check on the environment and protocol. If the
expert scores 60% in the collection loop but 20% here, the evaluation harness is
wrong, not the expert.

---

## 3. The rollout loop

### `eval/rollout.py`

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .policies import Policy


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    final_dist: float
    min_dist: float
    smoothness: float
    latency_ms: float
    rewards: list[float] = field(default_factory=list)


def run_episode(env, policy: Policy, seed: int, record: bool = False) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    policy.reset()
    actions = []
    rewards = []
    success = False
    min_dist = float("inf")
    frames = []
    latencies = []

    for step in range(env.cfg.episode_steps):
        # CUDA kernels are asynchronous; synchronize before timing inference.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        action = np.asarray(policy.act(obs), dtype=np.float32).reshape(-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        success = success or bool(info["success"])
        min_dist = min(min_dist, float(info["dist"]))
        if record:
            frames.append(obs["image"])

        if terminated or truncated:
            break

    actions = np.asarray(actions)
    smoothness = float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1))) \
        if len(actions) > 1 else 0.0

    return EpisodeResult(
        seed=seed,
        success=success,
        steps=len(actions),
        final_dist=float(info["dist"]),
        min_dist=float(min_dist),
        smoothness=smoothness,
        latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        rewards=rewards,
    )


def evaluate_policy(
    env,
    policy: Policy,
    seeds: list[int],
    record_first: int = 0,
) -> tuple[list[EpisodeResult], list[np.ndarray]]:
    results = []
    recordings = []
    for i, seed in enumerate(seeds):
        record = i < record_first
        result = run_episode(env, policy, seed, record=record)
        results.append(result)
        if record:
            # The loop currently records the observation after each step. For a
            # video, prepend the reset frame; see the exercise.
            pass
    return results, recordings
```

The `record` path is intentionally incomplete in the listing: recording the
reset frame requires capturing the observation before the loop. Add it when you
implement video:

```python
    obs, info = env.reset(seed=seed)
    frames = [obs["image"]] if record else []
```

Do **not** include environment stepping time in `latency_ms` if you want to
compare policy inference costs. If you want end-to-end control latency, report it
separately as `latency + env.step`.

### Determinism at evaluation

- Call `policy.act` with the same observation sequence; do not sample actions
  from a distribution.
- On CUDA, synchronize before and after `policy.act` when measuring latency;
  otherwise you time kernel launches, not execution.
- For diffusion policies, use a fixed initial noise seed per episode if you want
  exact reproducibility; otherwise the sampler is stochastic. Report that.
- For ACT, set the CVAE latent to zero (the prior mean) at inference.
- Close the environment and recreate it for a different visual variant so model
  state cannot leak.

---

## 4. Metrics and confidence intervals

### `eval/metrics.py`

```python
from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(results) -> dict:
    n = len(results)
    successes = sum(int(r.success) for r in results)
    lo, hi = wilson_interval(successes, n)
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "success_ci95": [lo, hi],
        "mean_steps": float(np.mean([r.steps for r in results])) if n else 0.0,
        "mean_final_dist": float(np.mean([r.final_dist for r in results])) if n else 0.0,
        "mean_min_dist": float(np.mean([r.min_dist for r in results])) if n else 0.0,
        "mean_smoothness": float(np.mean([r.smoothness for r in results])) if n else 0.0,
        "mean_latency_ms": float(np.mean([r.latency_ms for r in results])) if n else 0.0,
    }
```

Why Wilson instead of the normal approximation? With 20 episodes and a 0.15
success rate, the normal interval produces negative lower bounds and is badly
calibrated. Wilson is the right default for small binomial samples.

**Do not compare overlapping confidence intervals as a hypothesis test.** If the
intervals overlap, the difference may still be meaningful with paired seeds; if
they do not, that is suggestive but not a formal test. For the final report, a
paired bootstrap over the same reset seeds is a better comparison:

```python
def paired_bootstrap(success_a, success_b, n_boot=2000, seed=0):
    """Difference in success rate, paired by episode seed."""
    a = np.asarray(success_a, dtype=np.float64)
    b = np.asarray(success_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    return float(a.mean() - b.mean()), np.percentile(diffs, [2.5, 97.5]).tolist()
```

This is optional; the main table can report Wilson intervals.

### What to report

At minimum:

| Column | Why |
|---|---|
| success rate + CI | primary metric |
| mean final distance | distinguishes “almost” from “nowhere near” |
| mean min distance | catches transient success |
| mean episode length | a successful short episode is not automatically better, but a very long one suggests hesitation |
| action smoothness | jitter is a common failure of chunking/diffusion |
| inference latency | deployment relevance and chunk-size trade-off |
| parameter count | fairness and reproducibility |

Do not report only the best seed. If you run multiple seeds, report every seed
in a table or appendix, and the mean ± std across seeds as a secondary summary.

---

## 5. Evaluation script

`scripts/evaluate.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from imitation_learning_reproduction.envs.push_arm import PushArmConfig, PushArmEnv
from imitation_learning_reproduction.eval.metrics import summarize
from imitation_learning_reproduction.eval.policies import ExpertPolicy, RandomPolicy
from imitation_learning_reproduction.eval.rollout import evaluate_policy
from imitation_learning_reproduction.utils.seed import resolve_device


def build_env(obs_mode: str, visual_variant: str, image_size: int, n_points: int):
    return PushArmEnv(PushArmConfig(
        obs_mode=obs_mode,
        visual_variant=visual_variant,
        image_size=image_size,
        n_points=n_points,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "expert", "checkpoint"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--visual-variant", default="default")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    device = resolve_device(args.device)

    if args.policy == "random":
        policy = RandomPolicy(seed=args.seed)
    elif args.policy == "expert":
        policy = ExpertPolicy()
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --policy checkpoint")
        from imitation_learning_reproduction.eval.policies import load_policy
        policy = load_policy(args.checkpoint, device=device)

    env = build_env(args.obs_mode, args.visual_variant, args.image_size, args.n_points)
    seeds = [args.seed + i for i in range(args.episodes)]
    results, _ = evaluate_policy(env, policy, seeds)
    env.close()

    summary = summarize(results)
    summary["policy"] = args.policy
    summary["visual_variant"] = args.visual_variant
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

`load_policy` is the one function that grows in every chapter. Start with a
registry and add branches as policies appear:

```python
def load_policy(checkpoint_path: str, device: str = "cuda"):
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    algo = ckpt["algorithm"]
    if algo == "bc":
        from imitation_learning_reproduction.eval.policies import BCPolicy
        return BCPolicy.from_checkpoint(ckpt).to(device)
    if algo == "act":
        ...
    raise ValueError(f"unknown algorithm {algo!r}")
```

`map_location="cpu"` is only a portable checkpoint-loading choice; the policy is
then moved to `device`. The default `device` for evaluation is CUDA, matching
training. Pass `--device cpu` only for a smoke test.

Do not put the model construction in `evaluate.py`; it will become an
unmaintainable switch. Each policy class knows how to load its own checkpoint.

### Baseline commands

```bash
uv run python scripts/evaluate.py --policy random --obs-mode state --episodes 50
uv run python scripts/evaluate.py --policy expert --obs-mode state --episodes 50
```

Expected: random < 5% success; expert around its collection success rate (with
small sampling variation). If random is above 10%, inspect the success criterion
and the reset distribution; a policy that succeeds by accident suggests the puck
starts too close to the goal.

---

## 6. Robustness protocol

The default evaluation measures in-distribution performance. Chapters 11–13 also
measure robustness with the visual variants implemented in chapter 02:

| Variant | What changes | What it tests |
|---|---|---|
| `default` | nothing | in-distribution |
| `color` | table/arm/puck colours | colour overfitting |
| `light` | light intensity/direction | lighting overfitting |
| `shift` | camera translated by ~12 cm | camera/view overfitting |
| `distractors` | three visual-only objects | clutter robustness |

Use the **same seeds** and the same number of episodes for every variant. Do not
retrain on the variants. The point is to measure the gap.

```bash
for variant in default color light shift distractors; do
  uv run python scripts/evaluate.py --policy checkpoint \
    --checkpoint runs/dp3/best.pt --obs-mode point_cloud \
    --visual-variant "$variant" --episodes 50 \
    --out "results/dp3_${variant}.json"
done
```

A large drop on `shift` for an image policy and a small drop for a point-cloud
policy is the expected qualitative result. A large drop for DP3 on `shift`
usually means the camera pose used to backproject points is wrong or the policy
has memorised point ordering, not that 3-D representations are useless.

---

## 7. Tests

```python
# tests/test_eval.py
import numpy as np
import pytest

from imitation_learning_reproduction.eval.metrics import paired_bootstrap, wilson_interval


def test_wilson_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0)
    assert 0.25 < hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert lo > 0.65
    assert hi == pytest.approx(1.0)


def test_wilson_symmetry():
    lo1, hi1 = wilson_interval(3, 20)
    lo2, hi2 = wilson_interval(17, 20)
    assert lo1 == pytest.approx(1 - hi2, abs=1e-6)
    assert hi1 == pytest.approx(1 - lo2, abs=1e-6)


def test_paired_bootstrap_identical():
    a = [True, False, True, True]
    diff, ci = paired_bootstrap(a, a)
    assert diff == 0.0
    assert ci[0] == 0.0 and ci[1] == 0.0
```

Also test the rollout loop with a dummy environment and a policy that returns a
fixed action. The point is to catch queue/reset bugs before a real policy exists.

---

## 8. Checklist

- [ ] Every policy implements `reset()` and `act(obs) -> (2,)`.
- [ ] Evaluation seeds are `10_000 … 10_049` for the final table.
- [ ] Success is any-time within 5 cm, computed from `info["success"]`.
- [ ] Success rate is reported with a Wilson 95% interval.
- [ ] Random and expert baselines are measured with the same harness.
- [ ] Test split is untouched until the final benchmark.
- [ ] Robustness variants use the same seeds.
- [ ] `results/*.json` includes policy, variant, seed range, and git commit (optional but recommended).

### Exercises

1. Add episode videos. Prepend the reset frame, save a GIF every tenth episode,
   and verify that the video shows the success or failure.
2. Add `success_at_k` (success within the first 20 steps) to distinguish fast
   from slow policies.
3. Implement paired bootstrap over the same seeds and report the paired
   difference between BC and ACT.
4. Add an `--episodes 200` mode for the final report and measure how much the
   Wilson interval shrinks.

Next: [05 · Behavioural Cloning](05_behavior_cloning.md).
