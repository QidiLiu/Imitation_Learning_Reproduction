# Imitation Learning Reproduction

A tutorial-driven, from-scratch reproduction of seven imitation-learning
algorithms on one small MuJoCo robotic-arm task:

- Behavioural Cloning
- Inverse Reinforcement Learning (MaxEnt lab + AIRL)
- Apprenticeship Learning
- GAIL with PPO
- ACT (transformer policy)
- Diffusion Policy
- DP3 (3D diffusion policy)

The task is a 2-DoF planar pusher with a 96×96 top camera, a 512-point cloud,
18-D proprioception and 2-D continuous actions. It is deliberately small enough
to train on a single CUDA GPU, while keeping the parts of imitation learning that
matter: continuous control, contact, partial observability, multimodal
demonstrations and a fixed evaluation protocol. All training recipes default to
CUDA; CPU is an explicit fallback for smoke tests.

## Start here

Read the tutorial index:

**[`doc/tutorial/README.md`](doc/tutorial/README.md)**

It contains the roadmap, the project specification, the expected results and the
links to all chapters. The tutorial is written for you to implement the project
chapter by chapter; it provides reference implementations for the non-obvious
math and model code, plus acceptance tests.

## Repository branches

- **`main`** contains only the tutorial (this README plus `doc/tutorial/`) and
  the environment scaffolding (`pyproject.toml`, `uv.lock`, `.python-version`,
  the placeholder source package). It is the reading/reference branch.
- **`reproduction`** is the working branch for the hand-written implementation
  of every algorithm. The reproduction code belongs there, not on `main`.

```bash
git switch reproduction   # write code here
git switch main           # read the tutorial
```

The tutorial itself is shared by both branches; use `git switch` rather than
copying documentation between branches.

## Repository status

- The tutorial is complete (chapters 01–14, appendices A–B).
- The implementation lives on `reproduction`; `main` stays tutorial-only.
- The `uv` environment in this repository is the one to use:

```bash
uv sync
uv run python -V
```

## Suggested first session

```bash
cat doc/tutorial/README.md
cat doc/tutorial/01_setup.md
cat doc/tutorial/02_simulation.md
```

Do not skip the point-cloud calibration test in chapter 02; DP3 depends on it.

## License

Add a license before publishing. The tutorial text and the code you write are
yours; the cited papers and LeRobot remain under their respective licenses.
