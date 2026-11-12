# 13 · Benchmark, Ablations, and Analysis

> **Goal.** Turn the individual experiments into one defensible benchmark:
> a results table with confidence intervals, a robustness matrix, a small ablation
> study, and a failure analysis that explains the numbers rather than just
> reporting them.
>
> **Prerequisites.** Chapters 5–12.
>
> **Time.** 1 day, plus training time.
>
> **Artifacts.** `scripts/benchmark.py`, `results/benchmark.md`,
> `results/benchmark.png`, a short report in `results/report.md`.

---

## 1. Freeze the protocol

Before running the final benchmark, write down and do not change:

- the dataset path, split seed, and number of demonstrations;
- the evaluation seeds (`10_000 … 10_049`);
- the success criterion (5 cm, any-time);
- the visual variants and their implementation;
- the checkpoint selection rule (best validation loss for supervised methods,
  best evaluation success for RL/adversarial methods — note the asymmetry);
- the number of evaluation episodes (50; 200 for the final table if time allows);
- the hardware and software versions.

The checkpoint-selection rule is a subtle source of unfairness. Supervised
methods select by validation loss; RL methods select by evaluation success on the
same reset seeds used for reporting. That is optimistic for RL. Either use a
separate validation seed set for model selection or report both “best” and
“last”. Pick one rule and state it.

For a cleaner comparison:

- use `val` demonstrations for supervised early stopping;
- use a **validation seed set** `20_000 … 20_049` for RL checkpoint selection;
- report on the disjoint test seed set `10_000 … 10_049`.

If you already selected RL checkpoints on `10_000`, do not hide it. Re-run the
final evaluation on a fresh set or disclose the selection bias.

---

## 2. Results table

Store one JSON per method and variant. `scripts/evaluate.py` already writes:

```json
{
  "policy": "checkpoint",
  "visual_variant": "default",
  "episodes": 50,
  "successes": 39,
  "success_rate": 0.78,
  "success_ci95": [0.65, 0.87],
  "mean_steps": 31.4,
  "mean_final_dist": 0.071,
  "mean_min_dist": 0.032,
  "mean_smoothness": 0.14,
  "mean_latency_ms": 8.2
}
```

Add metadata by hand or in the evaluation script:

```json
{
  "method": "act",
  "checkpoint": "runs/act/20261112-214000/best.pt",
  "params": 812345,
  "train_minutes": 38,
  "git_commit": "abc1234",
  "selection": "best val loss"
}
```

The final table:

| Method | Obs | Params | Train time | Success (95% CI) | Final dist | Min dist | Latency |
|---|---|---:|---:|---|---:|---:|---:|
| Random | state | 0 | — | 0.02 (0.00–0.07) | — | — | — |
| Expert | state | 0 | — | 0.62 (0.48–0.74) | — | — | — |
| BC (state) | state | 0.1M | 2 min | TODO | TODO | TODO | TODO |
| BC (image) | image | 0.3M | 8 min | TODO | TODO | TODO | TODO |
| PPO (env reward) | state | 0.05M | 6 min | TODO | TODO | TODO | TODO |
| AIRL | state | 0.2M | 25 min | TODO | TODO | TODO | TODO |
| Apprenticeship | state | 0.2M | 30 min | TODO | TODO | TODO | TODO |
| GAIL | state | 0.2M | 25 min | TODO | TODO | TODO | TODO |
| ACT | image | 1.0M | 35 min | TODO | TODO | TODO | TODO |
| DP | image | 2.0M | 45 min | TODO | TODO | TODO | TODO |
| DP3 | point cloud | 1.6M | 35 min | TODO | TODO | TODO | TODO |

Numbers are illustrative; measure your own. Use the same hardware for latency
comparisons, and warm up the model before timing.

### Aggregating the JSONs

`scripts/benchmark.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        data["file"] = path.name
        rows.append(data)
    return rows


def fmt_ci(lo, hi):
    return f"[{lo:.2f}, {hi:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/benchmark.md")
    args = parser.parse_args()

    rows = load_rows(Path(args.results))
    lines = [
        "| Method | Variant | Episodes | Success | 95% CI | Final dist | Min dist | Latency (ms) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        name = r.get("method", r.get("policy", r["file"].replace(".json", "")))
        lines.append(
            f"| {name} | {r.get('visual_variant', 'default')} | {r['episodes']} | "
            f"{r['success_rate']:.3f} | {fmt_ci(*r['success_ci95'])} | "
            f"{r['mean_final_dist']:.3f} | {r['mean_min_dist']:.3f} | "
            f"{r['mean_latency_ms']:.1f} |"
        )
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
```

The script is intentionally dumb. A benchmark harness that silently renames
methods or drops failing runs is worse than a spreadsheet.

### Plotting

A grouped bar chart of success rate with Wilson error bars is enough:

```python
import matplotlib.pyplot as plt
import numpy as np

methods = ["BC", "ACT", "DP", "DP3"]
rates = np.array([0.58, 0.82, 0.79, 0.86])
lo = rates - np.array([0.44, 0.69, 0.66, 0.74])
hi = np.array([0.71, 0.90, 0.88, 0.93]) - rates

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(methods, rates, yerr=[lo, hi], capsize=4, color="#4c72b0")
ax.set_ylim(0, 1)
ax.set_ylabel("success rate")
ax.set_title("PlanarPush, 50 episodes, 95% Wilson CI")
fig.tight_layout()
fig.savefig("results/benchmark.png", dpi=160)
```

Do not use error bars that assume a normal distribution with 10 episodes. Use
Wilson intervals or bootstrap.

---

### 2.5 Checkpoint contract and policy registry

`scripts/evaluate.py` calls `load_policy(checkpoint)`. For that to work, every
training script must save the same fields:

```text
algorithm        "bc" | "act" | "dp" | "dp3" | "ppo" | "airl" | "gail" | "apprenticeship"
model            state_dict of the policy network
model_config     kwargs needed to rebuild the network
stats            dataset statistics (state mean/std, action low/high, pc mean/std)
obs_rms          running observation statistics (RL methods only)
kind             "state" or "image" (BC only)
obs_horizon      observation history (ACT/DP/DP3)
execute_horizon  chunk execution length (ACT/DP/DP3)
config           the full training config (for provenance)
```

Save `model_config` explicitly; do not try to infer architecture from the
state-dict shapes. A registry with lazy imports keeps the evaluation script
small:

```python
# eval/policies.py
from __future__ import annotations

import torch


def load_policy(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    algo = ckpt["algorithm"]
    stats = ckpt.get("stats")
    model_config = ckpt.get("model_config", {})
    from imitation_learning_reproduction.training.normalizer import (
        ActionNormalizer, StateNormalizer,
    )

    def state_norm():
        return StateNormalizer(torch.tensor(stats["state_mean"]),
                               torch.tensor(stats["state_std"]))

    def action_norm():
        return ActionNormalizer(torch.tensor(stats["action_low"]),
                                torch.tensor(stats["action_high"]))

    if algo == "bc":
        from imitation_learning_reproduction.models.bc import BCPolicy, ImageBCPolicy
        if ckpt.get("kind") == "image":
            model = ImageBCPolicy(**model_config)
            model.set_normalizers(state_norm(), action_norm())
            model.load_state_dict(ckpt["model"])
            model.eval()
            return model
        model = BCPolicy.from_checkpoint(ckpt)
        return model

    if algo == "act":
        from imitation_learning_reproduction.models.act import ACT
        model = ACT(**model_config)
        model.set_normalizers(state_norm(), action_norm())
        model.load_state_dict(ckpt["model"])
        return ACTPolicy(
            model, stats,
            obs_horizon=ckpt.get("obs_horizon", 2),
            execute_horizon=ckpt.get("execute_horizon", 8),
            temporal_ensemble=ckpt.get("temporal_ensemble", True),
        )

    if algo == "dp":
        from imitation_learning_reproduction.models.dp import DiffusionPolicy
        model = DiffusionPolicy(**model_config)
        model.set_normalizers(state_norm(), action_norm())
        model.load_state_dict(ckpt["model"])
        return DPPolicy(
            model, stats,
            obs_horizon=ckpt.get("obs_horizon", 2),
            execute_horizon=ckpt.get("execute_horizon", 8),
            num_inference_steps=ckpt.get("num_inference_steps", 10),
        )

    if algo == "dp3":
        from imitation_learning_reproduction.models.dp3 import DP3Policy
        model = DP3Policy(**model_config)
        model.set_normalizers(state_norm(), action_norm())
        model.load_state_dict(ckpt["model"])
        return DP3EvalPolicy(
            model, stats,
            obs_horizon=ckpt.get("obs_horizon", 2),
            execute_horizon=ckpt.get("execute_horizon", 8),
            num_inference_steps=ckpt.get("num_inference_steps", 10),
        )

    if algo in {"ppo", "airl", "gail", "apprenticeship"}:
        from imitation_learning_reproduction.rl.ppo import (
            ActorCritic, RunningMeanStd,
        )
        actor = ActorCritic(**model_config)
        actor.load_state_dict(ckpt["model"])
        obs_rms = RunningMeanStd.from_state_dict(ckpt["obs_rms"])
        return PPOActorPolicy(actor, obs_rms)

    raise ValueError(f"unknown algorithm {algo!r}")
```

`PPOActorPolicy`, `ACTPolicy`, `DPPolicy` and `DP3EvalPolicy` all live in
`eval/policies.py`; the registry refers to them directly. Keep that file focused
on adapters and checkpoint loading, not on training.

The image-BC branch loads `ImageBCPolicy` directly; its `act` method is defined in
chapter 05. If you add a new policy type, add exactly one branch here and one
checkpoint contract entry. Do not grow `scripts/evaluate.py`.

---

## 3. Robustness matrix

Run every learned policy under the five variants. The table should be method ×
variant:

| Method | default | colour | light | shift | distractors |
|---|---:|---:|---:|---:|---:|
| BC (image) | TODO | TODO | TODO | TODO | TODO |
| ACT | TODO | TODO | TODO | TODO | TODO |
| DP | TODO | TODO | TODO | TODO | TODO |
| DP3 | TODO | TODO | TODO | TODO | TODO |

State-based methods use the same physics and are unaffected by the visual
variants (they do not see colours), so they are a useful control. If a state
method changes across visual variants, the variant is changing physics or the
task distribution, not just appearance. That is a bug.

### Interpreting robustness

- A drop on `colour` for an image method and none for DP3 is expected.
- A drop on `shift` for DP and none for DP3 is the most interesting result.
- A drop for DP3 on `shift` suggests a camera-pose/backprojection bug.
- `distractors` adds geometry; both methods can degrade. Do not claim 3-D
  invariance to new objects.

Report absolute success rates, not only relative drops. “DP drops 10 points from
80%” is different from “DP drops 10 points from 30%”.

---

## 4. Ablations

Choose a small, hypothesis-driven set. Do not run a grid search and report the
best cell without correcting for selection.

### Data efficiency

Train BC, ACT, DP and DP3 on `10, 25, 50, 100` demonstrations. Plot success vs
number of demonstrations. This is the most informative ablation for imitation
learning and the easiest to explain.

### Observation horizon

`T_o ∈ {1, 2, 4}` for ACT/DP. A single frame may be enough for this task; if so,
say so. More history is not automatically better.

### Action horizon

`T_p ∈ {1, 8, 16, 32}` and execution horizon `k ∈ {1, 4, 8, 16}`. Separate the
prediction horizon from the execution horizon; they have different effects on
smoothness and reactivity.

### Diffusion steps

DDIM `{5, 10, 25, 50}` at inference, with the same trained model. Plot success
and latency. A flat success curve with a large latency increase is a useful
result.

### Point count

`N ∈ {128, 256, 512, 1024}` for DP3. Plot success and training time. More points
can hurt if the object points are a small fraction and the encoder overfits.

### Reward/adversarial choices

- AIRL: with/without BC warm start; with/without gradient penalty.
- GAIL: D steps `{1, 3, 5}`; gradient penalty on/off.
- Apprenticeship: feature set variants; QP vs latest policy.

### Report the compute

Every ablation costs time. Include a table of GPU/CPU hours. A result that is 2
points better at 10× compute is a different claim from a free improvement.

---

## 5. Failure analysis

A benchmark without failure analysis is a leaderboard. For each method, inspect
at least 10 failing episodes and classify them:

| Failure type | Description | Likely cause |
|---|---|---|
| No approach | Arm never moves toward the puck | Policy ignores state/image; wrong normalization |
| Contact but no push | Paddle reaches the puck and stops | Action scale/saturation; reward/policy collapse |
| Push wrong direction | Puck moves away from goal | Goal encoding/view confusion; feature insufficiency |
| Overshoot | Puck passes the goal | No deceleration; chunk too long; no closed-loop correction |
| Oscillation | Arm alternates around the puck | Chunk boundary jitter; deterministic mean of a multimodal policy |
| Distractor capture | Arm follows a distractor | Image/point-cloud overfitting; clutter |
| Camera-shift failure | Behaviour collapses on `shift` | Fixed-view overfitting; calibration bug |
| Early stop | Episode terminates before success | Success flag bug or env termination |

For each failure, save a GIF and the puck-goal distance curve. Put 2–3 examples
in the report. A concrete failure image is more persuasive than another decimal
in the table.

### Distance curves

Plot `||puck - goal||` vs step for 5 successes and 5 failures per method. This
shows whether a method is “almost working” (distance decreases then stalls) or
“not working” (distance flat or increasing). Aggregate success rate hides this.

---

## 6. Report template

Write `results/report.md` with:

1. **Setup.** Environment, dataset size, success criterion, seeds, hardware.
2. **Protocol.** Checkpoint selection, number of episodes, confidence intervals.
3. **Results.** Main table and robustness matrix.
4. **Ablations.** 2–4 plots, each with a hypothesis.
5. **Failure analysis.** 2–3 concrete examples.
6. **Discussion.** Which method wins, which is most sample-efficient, which is
   most robust, and what you would do with more compute.
7. **Limitations.** Scripted expert, planar task, single camera, small dataset,
   no human demos, no sim-to-real.
8. **Reproduction instructions.** Exact commands and commit hash.

A good discussion answers questions like:

- Does action chunking help more than a bigger MLP?
- Does 3-D help under colour/lighting shift, or only under camera shift?
- Is GAIL’s instability visible in the discriminator accuracy curve?
- Does AIRL recover a reward that correlates with distance-to-goal?
- Is Apprenticeship Learning’s margin a useful predictor of success?

---

## 7. Reproducibility checklist

- [ ] `uv.lock` committed; `uv sync` reproduces the environment.
- [ ] Dataset stats and splits committed or regenerated with a fixed seed.
- [ ] Every command in the report is copy-pasteable.
- [ ] Evaluation seeds and episode count recorded.
- [ ] Checkpoint selection rule stated.
- [ ] Hardware and wall-clock times recorded.
- [ ] Results JSONs and plots committed under `results/`.
- [ ] Code version (git commit) recorded in the result metadata.
- [ ] At least one seed-variance estimate for the main methods.
- [ ] Limitations stated honestly.

---

## 8. Checklist

- [ ] Main table has 50 episodes per method and Wilson intervals.
- [ ] Robustness matrix uses the same seeds.
- [ ] At least three ablations with a clear hypothesis.
- [ ] Failure taxonomy with concrete examples.
- [ ] Report includes limitations and reproduction commands.
- [ ] No method was tuned on the test seeds without disclosure.

### Exercises

1. Run a second seed for BC, ACT, DP and DP3 and report mean ± std across seeds.
2. Add a paired bootstrap comparison between ACT and DP.
3. Evaluate all methods on 200 episodes and see whether the ranking changes.
4. Add a “distance AUC” metric and compare it with success rate.
5. Write a one-page abstract of your benchmark as if it were a workshop paper.

Next: [14 · LeRobot, real robots, and next steps](14_lerobot_and_next_steps.md).
