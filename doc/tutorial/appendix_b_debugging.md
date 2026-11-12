# Appendix B · Debugging, Performance, and Reproducibility

This appendix is the “what do I do when it does not work” reference. Most bugs in
this project are not algorithmic; they are shape, normalisation, sign, or
lifecycle bugs. Use the tables first, then the deeper checklists.

---

## B.1 Shape cheat sheet

Write these on a sticky note. Most errors are a violation of one of these.

| Object | Shape |
|---|---|
| Single state | `(18,)` |
| Batch of states | `(B, 18)` |
| Observation window | `(T_o, 18)` |
| Image frame | `(3, H, W)` or `(H, W, 3)` depending on layer |
| Dataset image item | `(T_o, H, W, 3)` uint8 |
| Model image input | `(B, T_o, 3, H, W)` float in `[-1,1]` |
| Point cloud frame | `(N, 3)` |
| Dataset point cloud item | `(T_o, N, 3)` float32 |
| Action | `(2,)` |
| Action chunk | `(T_p, 2)` |
| Batch action chunk | `(B, T_p, 2)` |
| PPO rollout | `(T, N_envs, 18)` |
| Diffusion U-Net input | `(B, 2, T_p)` |
| Transformer memory | `(B, T_o * tokens + T_o, d_model)` |

If a model complains about a 4-D vs 5-D tensor, do not add `unsqueeze` until you
know which dimension is missing. Print shapes in the dataset and at the model
boundary; the mismatch is usually a missing `T_o` or a missing batch dimension.

---

## B.2 Normalisation checklist

Normalisation bugs produce policies that train but fail. Check each item:

- [ ] State stats computed on the **train** split only.
- [ ] State stats saved in the checkpoint and restored at evaluation.
- [ ] Action min/max computed on the train split, with `low < high`.
- [ ] Normalised actions in `[-1,1]`; inverse before `env.step`.
- [ ] Images converted to float and scaled to `[-1,1]` exactly once.
- [ ] Point clouds normalised with train mean/std, not per frame.
- [ ] PPO observations normalised with the running stats used during training.
- [ ] Expert transitions for AIRL/GAIL use the same state normaliser as the policy.
- [ ] No normalisation applied twice (dataset + model).
- [ ] No normalisation applied zero times (raw metres into a neural net).

A quick test: take one real observation, run it through the policy's preprocessing
and inverse transforms, and check that the resulting action is in `[-1,1]` and
that `inverse(normalize(x)) ≈ x`.

---

## B.3 MuJoCo problems

| Symptom | Likely cause / fix |
|---|---|
| `gladLoadGL error` | No GL context; try `MUJOCO_GL=egl` |
| `OSMesa` import error | Install `libosmesa6` or use EGL |
| GLFW display error | Headless; use EGL |
| Render is all one colour | Camera points at empty space or lights saturate |
| Depth is all `zfar` | Camera sees no geometry; check table/camera |
| Point cloud mirrored | Camera rotation transpose or y sign |
| Joint does not move | Collision with base/adjacent link; set `contype=0` |
| Actuator force large, joint stuck | Constraint at joint; inspect contacts |
| Goal at old position | Forgot `mj_forward` after changing `body_pos` |
| Puck tunnels through table | Timestep too large or geometry too thin |
| Expert success low | IK branch, paddle contact, action scale, goal reachability |

### Inspect contacts

```python
mujoco.mj_forward(model, data)
for i in range(data.ncon):
    c = data.contact[i]
    g1 = model.geom(c.geom1).name
    g2 = model.geom(c.geom2).name
    print(g1, g2, c.dist)
```

If the arm link is in contact with the base, that is your stuck joint.

### Inspect camera pose

```python
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
print(data.cam_xpos[cam_id], data.cam_xmat[cam_id].reshape(3, 3))
```

The camera looks along its local `-z`; the columns of `cam_xmat` are the camera
axes in world coordinates.

---

## B.4 HDF5 problems

| Symptom | Fix |
|---|---|
| `unable to open file` | Another process has it open; close viewers |
| `file already open` in workers | Do not pickle `h5py.File`; preload or open per worker |
| Dataset loads but images are black | Check `image` dtype and HWC/CHW order |
| `KeyError: 'ep_000123'` | Split names and HDF5 keys disagree; regenerate splits |
| File corrupt after a crash | Write to a temporary file and rename on success |
| Huge file | Use gzip and chunked image datasets |

Write datasets to `demos.hdf5.tmp` and `os.replace` to `demos.hdf5` after a
successful finalize. A half-written HDF5 is worse than no dataset.

---

## B.5 PyTorch problems

| Symptom | Likely cause |
|---|---|
| Loss NaN | `log(0)`, `atanh(±1)`, division by zero, unnormalised inputs |
| Loss decreases then explodes | LR too high; missing grad clip; bad normalisation |
| Train/eval mismatch | BatchNorm with batch size 1; dropout; normaliser not in eval |
| All actions constant | Action normaliser degenerate; policy collapsed to mean |
| Gradients zero | Detached tensor in the loss; `torch.no_grad` around training |
| CUDA OOM | Batch too large; image resolution; activation checkpointing |
| Slow CPU | Too many threads contending; large images; no `torch.no_grad` at eval |
| Different results each run | Seeds not set; DataLoader workers; nondeterministic kernels |

### Print one batch, always

```python
batch = next(iter(loader))
for k, v in batch["obs"].items():
    print(k, v.shape, v.dtype, float(v.float().min()), float(v.float().max()))
print("action", batch["action"].shape, batch["action"].min(), batch["action"].max())
print("pad", batch["action_pad"].sum(dim=1))
```

Do this before the first training run. It catches most data bugs in 30 seconds.

### Overfit one batch

For any supervised model, train on one batch until the loss is near zero. If it
cannot overfit, the model/loss is wrong. This is faster than a full training run
and much more diagnostic.

---

## B.6 PPO / adversarial problems

| Symptom | Fix |
|---|---|
| Value loss diverges | Check returns are detached; reduce LR; reward scale |
| KL explodes | Lower LR; check observation normalisation; reduce epochs |
| Entropy collapses | Increase entropy coefficient; check log_std init |
| Success zero but reward rises | Reward hacking; inspect rollouts |
| Truncation bootstrap wrong | Use final observation for truncated steps |
| GAIL D accuracy 1.0 | Reduce D steps/LR; add label smoothing/gradient penalty |
| GAIL reward saturates | Clip reward; strengthen D regularisation |
| AIRL reward all potential | Regularise `h`; check `g` gradient |
| Apprenticeship margin oscillates | Inner PPO not solving the reward; scale features |

Log the following every iteration:

```text
policy_loss, value_loss, entropy, approx_kl, clip_frac,
mean_reward, success, disc_acc_expert, disc_acc_policy,
margin, feature_matching_error
```

If you cannot explain a curve, you cannot debug it.

---

## B.7 Diffusion / ACT problems

| Symptom | Fix |
|---|---|
| Loss NaN | Clamp logvar; check `alpha_bar` indexing; action range |
| Samples are noise | Reverse step uses wrong timestep or wrong `alpha_bar` |
| Condition ignored | Check condition shape; FiLM wiring; condition normalisation |
| Good loss, bad rollouts | Unnormalisation; action horizon; execution horizon |
| Jitter at chunk boundaries | Temporal ensembling; shorter execution horizon |
| KL collapse in ACT | Anneal KL; reduce max KL; check decoder capacity |
| Model overfits | More augmentation; smaller model; more demos |

### Diffusion sanity test

For a fixed `x_0` and `ε`:

```python
t = torch.tensor([0])
x_t = scheduler.add_noise(x0, eps, t)
assert torch.allclose(x_t, x0, atol=1e-5)  # at t=0, alpha_bar ~ 1
```

Then check that a 10-step DDIM sample from a model trained on a tiny dataset
reproduces the training actions approximately. If not, the sampler is wrong.

---

## B.8 Performance

### CPU

- Set `torch.set_num_threads(4–8)`; more threads can contend on small models.
- Use `num_workers=0` for small datasets; worker startup dominates.
- Render at 96×96, not 480×640, unless the task needs it.
- Reuse the MuJoCo `Renderer`; creating one per step is expensive.
- Render RGB and depth in one function, but only for data collection.
- Use `torch.no_grad()` for all evaluation and feature extraction.

### GPU

- `pin_memory=True`, `num_workers=4`, `persistent_workers=True`.
- Mixed precision for vision models; check that GroupNorm and diffusion
  schedulers remain stable.
- Move the batch to the device once, not per key.
- Use `channels_last` for CNN-heavy models if supported.

### Profiling

```python
import time

start = time.perf_counter()
for _ in range(20):
    action = policy.act(obs)
print("policy ms", (time.perf_counter() - start) / 20 * 1000)

start = time.perf_counter()
for _ in range(20):
    env.step(action)
print("env ms", (time.perf_counter() - start) / 20 * 1000)
```

Measure policy inference separately from environment stepping. A “slow policy”
may actually be slow rendering.

---

## B.9 Determinism

Layers of determinism:

1. `seed_everything(seed)` for Python/NumPy/Torch.
2. Environment reset seeds for task sampling.
3. DataLoader worker seeds.
4. CuDNN deterministic flags (optional).
5. MuJoCo deterministic stepping (same model/data/controls).

Check:

```python
obs1, _ = env.reset(seed=123)
obs2, _ = env.reset(seed=123)
np.testing.assert_allclose(obs1["state"], obs2["state"])
```

If this fails, the reset is using an unseeded RNG or leftover state. The visual
variant must not change physics; if it does, the evaluation is not paired.

---

## B.10 Test pyramid

| Level | Examples | Runtime |
|---|---|---|
| Unit | normalizer roundtrip, GAE, QP, scheduler, PointNet invariance | ms |
| Contract | env shapes, dataset shapes, padding, checkpoint load | < 5 s |
| Overfit | one batch, four demos | < 1 min |
| Smoke | one PPO update, one diffusion sample, one ACT forward | < 1 min |
| Integration | 20-episode evaluation with expert/random | < 1 min |
| Slow | full training, 50-episode benchmark | hours |

Run the first four on every commit. Run slow tests nightly or before a release.

```bash
uv run pytest -q -m "not slow"
uv run pytest -q -m slow
```

---

## B.11 Reproducibility checklist

- [ ] `uv.lock` committed.
- [ ] Dataset generation command and seed documented.
- [ ] Splits and stats saved with the dataset.
- [ ] Training config saved in the run directory.
- [ ] Checkpoint contains model, normalizer, config and step.
- [ ] Evaluation seeds and episode count fixed.
- [ ] Hardware recorded for latency numbers.
- [ ] Git commit recorded in result metadata.
- [ ] Random seeds reported for multi-seed results.
- [ ] Figures generated from saved JSON/CSV, not hand-edited.

---

## B.12 Git hygiene

- Ignore `data/`, `runs/`, `results/` by default.
- Commit small reference datasets or use Git LFS.
- Do not commit checkpoints (hundreds of MB) to a normal Git repo.
- Tag the final benchmark commit.
- Put long commands in `scripts/` or a `Makefile`, not only in your shell history.
- A `results/` directory with JSON and a report is more useful than a screenshot.

---

## B.13 A final diagnostic order

When something fails, check in this order:

1. **Shapes.** Print them at the dataset and model boundary.
2. **Normalisation.** Compare train/eval transforms; check the checkpoint.
3. **Signs and indices.** Camera y/z, GAE done mask, diffusion timesteps.
4. **Data.** Visualise a batch; are the images/pucks/actions sensible?
5. **Overfit.** Can the model fit one batch?
6. **Baseline.** Does the expert/random baseline still behave?
7. **Algorithm.** Only now question the objective and hyperparameters.

The algorithm is usually the last thing that is wrong. This is good news: the
first six are cheap to test.
