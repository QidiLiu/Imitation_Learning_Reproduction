# Imitation Learning Reproduction — A Hands-On Tutorial

This tutorial is a complete, build-it-yourself path from the equations of imitation
learning to seven working algorithms on **one shared robotic-arm task**. It is written
for someone who knows PyTorch and supervised learning, has read the basic idea of
imitation learning, and now wants to make the ideas concrete.

You will build the project yourself. Each chapter gives the theory, the design
decisions, reference implementations for the non-obvious parts, commands to run,
and acceptance tests that tell you whether your implementation is actually correct.

---

## 1. What you will build

A deliberately small but physically meaningful benchmark called **PlanarPush**:

| Piece | Choice |
|---|---|
| Robot | 2-DoF planar arm with a paddle end-effector |
| Task | Push an orange puck to a green goal marker on a table |
| Simulator | MuJoCo 4.x |
| Control | 20 Hz delta joint-position actions (2-D) |
| Observations | 18-D proprioceptive state, 96×96 top RGB, 512-point cloud |
| Episodes | Up to 60 control steps (3 s), terminate on success |
| Demonstrations | Scripted expert, successful episodes filtered, stored in HDF5 |
| Compute | CUDA-first; CPU is an explicit smoke-test fallback |

The 2-DoF arm is intentional. It is large enough to contain the hard parts of
imitation learning — continuous control, contact, partial observability from a
camera, multimodal human-like behaviour — but small enough that every experiment
finishes quickly on a single CUDA GPU. The algorithms do not care that the arm
has two joints; the same code scales to a 6-DoF arm by changing the environment
and dimensions.

**Branch note.** This repository has two branches: `main` holds the tutorial and
environment scaffolding only; `reproduction` is where you hand-write the
implementation. The tutorial text is shared by both. Use
`git switch reproduction` before creating code, and `git switch main` when you
want a clean reading branch.

### The algorithms

| # | Algorithm | What it learns | Core idea |
|---|---|---|---|
| 5 | Behavioural Cloning (BC) | Policy | Supervised regression `s → a` |
| 7 | Inverse RL (MaxEnt + AIRL) | Reward + policy | Explain demos with a reward, then optimize it |
| 8 | Apprenticeship Learning | Reward weights + policy | Match feature expectations |
| 9 | GAIL + PPO | Policy (implicit reward) | Match occupancy measures adversarially |
| 10 | ACT | Action-chunk policy | Transformer + CVAE over action sequences |
| 11 | Diffusion Policy | Action-chunk policy | Conditional denoising over action sequences |
| 12 | DP3 | 3-D action-chunk policy | Point-cloud encoder + diffusion |

The PPO chapter (6) is shared infrastructure for chapters 7–9.

### Why this order?

BC establishes the data pipeline and the evaluation protocol. PPO gives you a
reliable RL oracle. IRL, Apprenticeship Learning and GAIL all need that oracle.
ACT, Diffusion Policy and DP3 then move from state-only observations to image and
point-cloud observations, reusing the same dataset, splits and evaluation harness.

The tutorial order differs slightly from the list on the project landing page:
PPO comes before IRL/GAIL because it is a prerequisite, not because it is a
separate imitation-learning method.

---

## 2. Roadmap

| Chapter | Topic | Main artifact | Time (part-time) |
|---|---|---|---|
| [01](01_setup.md) | Environment, layout, config, smoke tests | `pyproject.toml`, `utils/config.py` | 0.5 day |
| [02](02_simulation.md) | MuJoCo task, env API, camera, point cloud | `envs/push_arm.py` | 1–2 days |
| [03](03_data.md) | Scripted expert, HDF5 dataset, windows | `data/`, `expert.py` | 1–2 days |
| [04](04_evaluation.md) | Fixed evaluation protocol, metrics, CIs | `eval/` | 0.5 day |
| [05](05_behavior_cloning.md) | BC, image-BC baseline, compounding error | `models/bc.py` | 0.5–1 day |
| [06](06_ppo.md) | PPO, GAE, vectorized envs, observation norm | `rl/ppo.py` | 1–2 days |
| [07](07_inverse_rl.md) | MaxEnt IRL, exact gridworld lab, AIRL | `rl/airl.py`, `envs/gridworld.py` | 2–3 days |
| [08](08_apprenticeship.md) | Feature expectations, max-margin, QP | `rl/apprenticeship.py` | 1–2 days |
| [09](09_gail.md) | GAIL objective, discriminator, stability | `rl/gail.py` | 1–2 days |
| [10](10_act.md) | ACT, CVAE, transformer, temporal ensemble | `models/act.py` | 2 days |
| [11](11_diffusion_policy.md) | DDPM/DDIM, 1-D U-Net, receding horizon | `models/diffusion.py` | 2 days |
| [12](12_dp3.md) | Point clouds, PointNet, 3-D policy, robustness | `models/dp3.py` | 1–2 days |
| [13](13_benchmark.md) | Unified benchmark, ablations, report | `scripts/benchmark.py` | 1 day |
| [14](14_lerobot_and_next_steps.md) | LeRobot datasets, real robots, extensions | adapter notes | 1–2 days |
| [A](appendix_a_math.md) | Derivations and notation | reference | — |
| [B](appendix_b_debugging.md) | Debugging, performance, reproducibility | reference | — |

Compute for the full core path is a few CUDA GPU hours on a modern GPU
(roughly 1–4 hours with the reduced configs, depending on the GPU and how many
ablations you run). Visual policy training dominates. The CPU fallback is much
slower and is intended only for smoke tests; benchmark numbers should be produced
with CUDA.

---

## 3. The rules of the project

Three rules keep the comparison honest:

1. **One environment and one dataset.** Every algorithm sees the same
   `PlanarPush` task, the same successful demonstrations, the same train/val/test
   episode split, and the same observation/action normalization. Do not tune the
   environment per algorithm.
2. **One evaluation protocol.** Evaluate on a fixed set of reset seeds with a
   deterministic policy, no test-time cherry-picking, and report a confidence
   interval. The protocol is specified once in chapter 04 and never changed.
3. **Always keep a dumb baseline.** A method is only interesting relative to
   random actions, the scripted expert, and BC. A method that beats random but
   loses to BC has not demonstrated much.

Two engineering rules make the code readable:

- Small modules with one responsibility; no monolithic `train.py` that contains
  five algorithms.
- Every tensor shape, array axis order and normalization convention is written
  down next to the code. Most imitation-learning bugs are shape or normalization
  bugs, not algorithm bugs.

---

## 4. Repository layout you will create

```text
Imitation_Learning_Reproduction/
├── pyproject.toml
├── README.md
├── configs/
│   ├── collect.yaml
│   ├── bc.yaml
│   ├── ppo.yaml
│   ├── airl.yaml
│   ├── apprenticeship.yaml
│   ├── gail.yaml
│   ├── act.yaml
│   ├── dp.yaml
│   └── dp3.yaml
├── data/
│   └── push_arm/
│       ├── demos.hdf5
│       ├── meta.json
│       └── splits.json
├── doc/tutorial/                 # this tutorial
├── scripts/
│   ├── collect_demos.py
│   ├── inspect_dataset.py
│   ├── train_bc.py
│   ├── train_ppo.py
│   ├── train_airl.py
│   ├── train_apprenticeship.py
│   ├── train_gail.py
│   ├── train_act.py
│   ├── train_dp.py
│   ├── train_dp3.py
│   ├── evaluate.py
│   └── benchmark.py
├── src/imitation_learning_reproduction/
│   ├── __init__.py
│   ├── envs/
│   │   ├── assets/push_arm.xml
│   │   ├── push_arm.py
│   │   ├── cameras.py
│   │   ├── expert.py
│   │   ├── gridworld.py
│   │   └── wrappers.py
│   ├── data/
│   │   ├── store.py
│   │   ├── dataset.py
│   │   ├── stats.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── common.py
│   │   ├── bc.py
│   │   ├── act.py
│   │   ├── diffusion.py
│   │   ├── dp.py
│   │   └── dp3.py
│   ├── rl/
│   │   ├── ppo.py
│   │   ├── airl.py
│   │   ├── apprenticeship.py
│   │   └── gail.py
│   ├── training/
│   │   ├── normalizer.py
│   │   ├── checkpoint.py
│   │   └── logger.py
│   ├── eval/
│   │   ├── rollout.py
│   │   ├── metrics.py
│   │   └── policies.py
│   └── utils/
│       ├── config.py
│       └── seed.py
└── tests/
    ├── test_smoke.py
    ├── test_env.py
    ├── test_dataset.py
    ├── test_cameras.py
    ├── test_ppo.py
    ├── test_irl.py
    ├── test_apprenticeship.py
    ├── test_act.py
    ├── test_diffusion.py
    └── test_dp3.py
```

The package name is long because the repository started with it. In the tutorial
we import explicitly:

```python
from imitation_learning_reproduction.envs.push_arm import PushArmEnv
```

If you prefer, rename the package to `ilr` and update `pyproject.toml`, but do it
in chapter 01 before writing code, not in the middle of the project.

---

## 5. What “done” looks like

At the end you should have:

- A deterministic MuJoCo environment with a documented observation and action
  contract.
- At least 100 successful scripted demonstrations, a validated HDF5 dataset, and
  a fixed train/val/test split.
- A scripted-expert, random, and BC baseline evaluated with confidence intervals.
- Working implementations of BC, PPO, AIRL (plus an exact tabular MaxEnt IRL
  lab), Apprenticeship Learning, GAIL, ACT, Diffusion Policy and DP3.
- A benchmark table with success rate, final distance, inference latency and
  robustness under colour/lighting/camera shift.
- A short written analysis of which method wins, which fails, and why.

A reasonable sanity band for the task, on 50 evaluation episodes, is:

| Method | Success rate (sanity band, not a target) |
|---|---|
| Random | < 5% |
| Scripted expert | 55–80% |
| BC (state) | 50–75% |
| BC (image) | 35–65% |
| PPO (environment reward) | 60–85% |
| AIRL | 50–75% |
| Apprenticeship Learning | 45–70% |
| GAIL + PPO | 55–80% |
| ACT | 70–90% |
| Diffusion Policy | 70–90% |
| DP3 | 75–95% |

These are broad bands, not leaderboard numbers. A result outside a band is a
signal to investigate, not proof that something is wrong. Record your actual
numbers and hardware.

---

## 6. Prerequisites

You should be comfortable with:

- Python, type hints, dataclasses, `pytest`;
- PyTorch tensors, `nn.Module`, autograd, a basic training loop;
- probability (expectations, Gaussians, KL divergence);
- the MDP notation `(S, A, P, r, γ)` and policy gradients at the level of
  REINFORCE/PPO.

You do **not** need prior MuJoCo, robotics, or Transformer experience. The
tutorial introduces the pieces it uses. If you have never implemented PPO, do
chapter 06 carefully; chapters 7–9 depend on it.

The tutorial assumes the `uv` environment already created in the repository. Do
not create a second virtual environment. Every command in the tutorial is run
with `uv run`, for example:

```bash
uv run python scripts/train_bc.py --config configs/bc.yaml
```

Chapter 01 adds the missing dependencies to the existing project.

---

## 7. How to use a chapter

Each chapter follows the same structure:

1. **Goal and prerequisites** — what you will have at the end.
2. **Theory** — the minimum math needed to implement the algorithm correctly.
3. **Design** — shapes, interfaces, and trade-offs.
4. **Implementation steps** — reference code for the non-obvious parts and a
   precise description of the boilerplate.
5. **Run and evaluate** — commands and what to look for.
6. **Debugging** — the failure modes we have seen in this kind of project.
7. **Exercises** — optional extensions.
8. **Checklist** — concrete acceptance criteria.

Do not skip the tests. In particular, the overfit-one-batch tests catch nearly
every shape/loss/normalization bug before you spend an hour on training.

---

## 8. Notation

| Symbol | Meaning |
|---|---|
| `s_t`, `a_t` | state and action at control step `t` |
| `τ` | trajectory `(s_0, a_0, …, s_T)` |
| `π_θ(a|s)` | policy |
| `r_φ(s,a)` | learned reward |
| `γ` | discount factor |
| `D` | demonstration dataset |
| `φ(s)` | hand-designed feature vector |
| `μ(π)` | feature expectation of policy `π` |
| `ρ_π(s,a)` | state-action occupancy measure |
| `T_o`, `T_p` | observation horizon, action-prediction horizon |
| `H, W` | image height and width |
| `N` | number of points in a point cloud |

Full derivations are in [Appendix A](appendix_a_math.md).

---

## 9. References

Start with these; each chapter adds specific references.

- Argall et al., *A Survey of Robot Learning from Demonstration* (2009).
- Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network* (1989).
- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning* (DAgger, 2011).
- Abbeel & Ng, *Apprenticeship Learning via Inverse Reinforcement Learning*
  (2004).
- Ziebart et al., *Maximum Entropy Inverse Reinforcement Learning* (2008).
- Ho & Ermon, *Generative Adversarial Imitation Learning* (2016).
- Fu, Luo, Levine, *Learning Robust Rewards with Adversarial Inverse
  Reinforcement Learning* (AIRL, 2018).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware* (ACT, 2023).
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion* (2023).
- Ze et al., *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via
  Simple 3D Representations* (RSS 2024).
- The LeRobot project: <https://github.com/huggingface/lerobot>.

---

## 10. Validation notes

While preparing the tutorial I validated the core simulation assumptions with a
CPU-only renderer/geometry prototype (Linux, Python 3.14, MuJoCo 4.0.1). This
does not change the training default: every training recipe in the tutorial uses
CUDA, and CPU is only an explicit fallback.

- EGL headless rendering at 96×96 works through Mesa when `MUJOCO_GL=egl`;
  `glfw` does not work without a display and OSMesa is not installed.
- A known box backprojects from depth to the correct world coordinates within
  ~1 cm at 64×64.
- The exact XML in chapter 02 loads and produces `nq=9, nv=8, nu=2`.
- The scripted expert in chapter 03 succeeds on about 66% of 100 seeds, with
  successful episodes averaging ~37 control steps.

These are sanity checks for the design, not a substitute for the tests you will
write. Your success rate will differ with small changes to the controller and the
task distribution; the tutorial's acceptance bands account for that.

---

## 11. Start here

1. Read [chapter 01](01_setup.md).
2. Create the repository layout and the configuration utility.
3. Run the smoke test.
4. Move to [chapter 02](02_simulation.md) and build the robot.

If you get stuck, [Appendix B](appendix_b_debugging.md) has a symptom-to-cause
table for the errors that occur most often in this project.
