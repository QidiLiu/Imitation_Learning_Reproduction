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
to train on a CPU, while keeping the parts of imitation learning that matter:
continuous control, contact, partial observability, multimodal demonstrations and
a fixed evaluation protocol.

## Start here

Read the tutorial index:

**[`doc/tutorial/README.md`](doc/tutorial/README.md)**

It contains the roadmap, the project specification, the expected results and the
links to all chapters. The tutorial is written for you to implement the project
chapter by chapter; it provides reference implementations for the non-obvious
math and model code, plus acceptance tests.

## Repository status

- The tutorial is complete (chapters 01–14, appendices A–B).
- The implementation is intentionally left to the reader; the tutorial
  specifies the modules, tensor shapes, commands and tests.
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
