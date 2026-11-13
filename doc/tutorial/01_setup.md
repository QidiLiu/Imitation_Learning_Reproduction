# 01 · Environment, Repository Layout, and Configuration

> **Goal.** Turn the empty repository into a reproducible Python project: dependencies
> installed with `uv`, a package layout that the rest of the tutorial will use,
> a small configuration system, deterministic seeding, and a smoke test that
> proves MuJoCo can render headlessly.
>
> **Prerequisites.** The repository's existing `.venv` and `uv`. No MuJoCo code yet.
>
> **Time.** 2–4 hours.

---

## 1. Check what you already have

From the repository root:

```bash
uv run python -V
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

At the time of writing this project uses Python 3.14 and PyTorch 2.15. **All
training configs in this tutorial default to CUDA.** A CUDA-capable GPU is the
expected environment. CPU is supported only as an explicit fallback for smoke
tests and CI; the scripts do not silently fall back.

If the command above prints `False`, install a CUDA-enabled PyTorch build that
matches your driver, or set `device: cpu` explicitly and treat that run as a
smoke test rather than a benchmark. See the PyTorch installation selector for the
correct CUDA wheel.

Do **not** create a new virtual environment. `uv run` uses `.venv` and keeps the
lockfile in sync.

### Add the project dependencies

```bash
uv add "mujoco>=4.0,<5" "gymnasium>=1.4,<2" "h5py>=3.16" \
       "PyYAML>=6.0" "tqdm>=4.70" "matplotlib>=3.10" \
       "tensorboard>=2.20" "imageio>=2.38" "numpy>=2.2"
uv add --dev "pytest>=9.0" "ruff>=0.14"
```

Versions tested while preparing this tutorial (2026-11): `mujoco 4.0.1`,
`gymnasium 1.4.0`, `h5py 3.16.0`, `numpy 2.2.6`, `pytest 9.2.0`. Newer patch
versions should work; if MuJoCo changes the Renderer API, chapter 02 has a
compatibility note.

LeRobot is **not** a core dependency. It is heavy, and only chapter 14 needs it.
Install it separately when you get there so the core environment stays small.

### Add project tooling

Edit `pyproject.toml` so it contains something close to:

```toml
[project]
name = "imitation-learning-reproduction"
version = "0.1.0"
description = "From-scratch imitation learning on a small MuJoCo arm"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "torch>=2.15.0",
    "torchvision>=0.29.2",
    "mujoco>=4.0,<5",
    "gymnasium>=1.4,<2",
    "h5py>=3.16",
    "PyYAML>=6.0",
    "tqdm>=4.70",
    "matplotlib>=3.10",
    "tensorboard>=2.20",
    "imageio>=2.38",
    "numpy>=2.2",
]

[project.scripts]
ilr-check = "imitation_learning_reproduction.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = [
    "slow: tests that train or render for more than a few seconds",
]

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

You do not need the `ilr-check` entry point; it is a convenient place for a
version-printing command later. If you add it, create `src/imitation_learning_reproduction/cli.py`
with a `main()` function.

> **Package name.** The existing source package is
> `src/imitation_learning_reproduction/`. Keep it. If you rename it to something
> short such as `ilr`, update the `uv_build` module configuration and every
> import at the same time. Doing the rename halfway through the tutorial creates
> a large, uninteresting diff.

After editing, run:

```bash
uv sync
uv run python -c "import mujoco, gymnasium, h5py, torch; print('ok')"
```

If `uv sync` changes the lockfile, commit the lockfile with the code. The lockfile
is part of reproducibility.

---

## 2. Create the directory skeleton

Create the following directories. Empty `__init__.py` files are fine for now.

```text
configs/
data/
doc/tutorial/
runs/
results/
scripts/
src/imitation_learning_reproduction/
    envs/
    envs/assets/
    data/
    models/
    rl/
    training/
    eval/
    utils/
tests/
```

```bash
mkdir -p configs data doc/tutorial runs results scripts \
         src/imitation_learning_reproduction/{envs/assets,data,models,rl,training,eval,utils} \
         tests
touch src/imitation_learning_reproduction/{__init__.py,envs/__init__.py,data/__init__.py} \
      src/imitation_learning_reproduction/{models/__init__.py,rl/__init__.py} \
      src/imitation_learning_reproduction/{training/__init__.py,eval/__init__.py,utils/__init__.py}
```

Update `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.venv/
.pytest_cache/
.ruff_cache/

# Project outputs
data/
runs/
*.gif
*.mp4

# MuJoCo / Mesa
.mujoco/
.mesa_shader_cache/
```

Do **not** ignore `configs/`, `doc/`, `scripts/`, `src/` or `tests/`. The dataset
is generated, so it is ignored by default; if you want to share a small reference
dataset, un-ignore a specific file or use Git LFS.

---

## 3. A tiny configuration system

Every script needs the same three things:

- load nested YAML into typed dataclasses;
- override a field from the command line without editing YAML;
- serialize the effective config into the run directory.

Hydra is excellent but brings a large conceptual surface for a project whose
interesting part is the algorithms. We will use a ~70-line utility.

### `src/imitation_learning_reproduction/utils/config.py`

```python
from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _build(tp: Any, value: Any) -> Any:
    """Recursively build a (possibly nested) dataclass from a YAML value."""
    if value is None:
        return None

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError(f"expected a mapping for {tp.__name__}, got {type(value).__name__}")
        hints = typing.get_type_hints(tp)
        kwargs = {}
        for field in dataclasses.fields(tp):
            if field.name in value:
                kwargs[field.name] = _build(hints[field.name], value[field.name])
        return tp(**kwargs)

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin in (typing.Union, types.UnionType):
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _build(candidate, value)
            except (TypeError, ValueError):
                continue
        return value

    if origin is list:
        item_tp = args[0] if args else Any
        return [_build(item_tp, item) for item in value]

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], item) for item in value)
        return tuple(_build(t, item) for t, item in zip(args, value, strict=False))

    if origin is dict:
        key_tp, val_tp = args if len(args) == 2 else (Any, Any)
        return {_build(key_tp, k): _build(val_tp, v) for k, v in value.items()}

    if tp is Path:
        return Path(value)
    if tp in (int, float, str, bool):
        return tp(value)
    return value


def load_config(path: str | Path, cls: type[T]) -> T:
    """Load YAML from *path* into a dataclass of type *cls*."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return _build(cls, raw)


def apply_overrides(cfg: T, overrides: list[str] | None) -> T:
    """Apply ``--set a.b=value`` overrides in place and return *cfg*."""
    for item in overrides or []:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key=value")
        value = yaml.safe_load(raw_value)
        target: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise AttributeError(f"unknown config field {key!r}")
        setattr(target, parts[-1], value)
    return cfg


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """JSON/YAML-friendly view of a dataclass config."""
    def convert(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {f.name: convert(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(cfg)
```

Usage in a script:

```python
import argparse
from dataclasses import dataclass, field
from imitation_learning_reproduction.utils.config import apply_overrides, load_config

@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"
    lr: float = 1e-3

@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    out: str = "runs/example"

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--set", nargs="*", default=[])
args = parser.parse_args()
cfg = apply_overrides(load_config(args.config, Config), args.set)
```

`--set train.lr=1e-4` now works without touching YAML, and the result is still a
typed dataclass. The conversion is intentionally conservative: unknown fields
are ignored, so a typo in YAML is silent. The `--set` path checks field
existence, but YAML loading does not. For a project of this size that is an
acceptable trade-off; if you grow it, add a strict mode.

Write two unit tests for this utility now; you will rely on it everywhere.

```python
# tests/test_config.py
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from imitation_learning_reproduction.utils.config import apply_overrides, load_config


@dataclass
class Train:
    lr: float = 1e-3
    epochs: int = 10


@dataclass
class Config:
    seed: int = 0
    out: Path = Path("runs/x")
    train: Train = field(default_factory=Train)


def test_load_nested(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 3\nout: runs/y\ntrain:\n  lr: 0.1\n")
    cfg = load_config(path, Config)
    assert cfg.seed == 3
    assert cfg.out == Path("runs/y")
    assert cfg.train.lr == pytest.approx(0.1)
    assert cfg.train.epochs == 10


def test_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 0.1\n")
    cfg = load_config(path, Config)
    apply_overrides(cfg, ["train.epochs=3", "seed=7"])
    assert cfg.train.epochs == 3
    assert cfg.seed == 7


def test_unknown_override_raises(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 1\n")
    cfg = load_config(path, Config)
    with pytest.raises(AttributeError):
        apply_overrides(cfg, ["train.nope=1"])
```

---

## 4. Seeding and device selection

Determinism in this project has three layers:

1. Python/NumPy/PyTorch RNGs for model initialization and data order.
2. The environment's reset seed for task sampling.
3. MuJoCo's deterministic physics given the same state and controls.

The first two need explicit seeding. The third is deterministic in practice when
you do not use random physics options.

### `src/imitation_learning_reproduction/utils/seed.py`

```python
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and PyTorch. Call once per process, before model creation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str = "cuda") -> torch.device:
    """Resolve the training device. CUDA is the default; CPU is explicit only."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Install a CUDA-enabled "
            "PyTorch build, or pass device='cpu' only for smoke tests."
        )
    return device
```

Use `deterministic_torch=False` by default. Deterministic cuDNN kernels can be
much slower and occasionally unavailable for an operation. Reproducibility of
the *experiment* is usually sufficient; bitwise reproducibility of a GPU kernel
is a different, stricter goal.

DataLoader workers are another source of nondeterminism. The CUDA default uses
`num_workers=4` and `pin_memory=True`. If you fall back to CPU for a smoke test,
use `num_workers=0`. If you use workers on either device, seed them with a
`worker_init_fn` or accept run-to-run variation.

---

## 5. MuJoCo headless rendering

MuJoCo itself is deterministic and does not need a display. Rendering does.
On Linux servers and CI machines there are three common backends:

| `MUJOCO_GL` | Requirement | Notes |
|---|---|---|
| `egl` | EGL/OpenGL driver (NVIDIA, or Mesa software) | Best default on headless Linux |
| `osmesa` | `libOSMesa` | Pure software, slower, sometimes unavailable |
| `glfw` | X11/Wayland display | Use on a desktop with `DISPLAY` set |

On a headless CUDA machine, EGL normally uses the NVIDIA driver; on a CPU-only
machine it can fall back to Mesa software EGL. `glfw` requires an X11/Wayland
display, and `osmesa` may not be installed. Test the backend on the machine that
will run data collection before committing to it:

```bash
MUJOCO_GL=egl uv run python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d)
print(r.render().shape)
"
```

If `egl` prints a shader-cache warning about a read-only home directory, set:

```bash
export MESA_SHADER_CACHE_DISABLE=true
```

For everyday use, put this in `tests/conftest.py` **before** any `mujoco` import:

```python
# tests/conftest.py
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
```

and at the top of the environment module:

```python
import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402  (must come after setting MUJOCO_GL)
```

This import-order trick is ugly but standard. Document it once and keep it in one
place. Do not scatter `os.environ` changes across scripts.

**Renderer API note.** In MuJoCo 3.x you may see `Renderer(model, height, width,
depth=True)`. MuJoCo 4.0 removed the `depth` keyword. The portable pattern used
in this tutorial is:

```python
renderer = mujoco.Renderer(model, height, width)

renderer.update_scene(data, camera=camera)
rgb = renderer.render()                    # (H, W, 3) uint8

renderer.enable_depth_rendering()
renderer.update_scene(data, camera=camera)
depth = renderer.render()                  # (H, W) float32, metres
renderer.disable_depth_rendering()
```

Always call `update_scene` again after toggling the mode. Relying on the previous
scene is a common source of stale images.

---

## 6. Smoke test

Create `tests/test_smoke.py`. It should prove four things:

1. The dependencies import.
2. Torch works and CUDA is available for training (or is explicitly skipped for a CPU smoke test).
3. MuJoCo can render a non-empty image headlessly.
4. The config utility round-trips a nested dataclass.

```python
# tests/test_smoke.py
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pytest
import torch

from imitation_learning_reproduction.utils.seed import resolve_device, seed_everything


def test_torch_works():
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, device=device)
    y = torch.randn(3, 2, device=device)
    assert torch.allclose(x @ y, (y.T @ x.T).T, atol=1e-6)
    assert x.device.type == device.type


def test_mujoco_render():
    xml = """
    <mujoco>
      <visual><global offwidth="64" offheight="64"/></visual>
      <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="1 1 .1" rgba=".5 .5 .5 1"/>
        <body pos="0 0 .1"><geom type="sphere" size=".05" rgba="1 0 0 1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert image.std() > 1.0, "the image is flat; rendering probably failed"


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is the training default; run this on a CUDA machine")
def test_cuda_device():
    assert resolve_device("cuda").type == "cuda"
```

Run:

```bash
uv run pytest tests/test_smoke.py tests/test_config.py -q
```

Expected output: all tests pass, with MuJoCo possibly printing a GL warning on
the first run.

If the render test fails:

| Symptom | Fix |
|---|---|
| `gladLoadGL error` | Backend cannot create a context; try `MUJOCO_GL=egl` |
| `OSMesa` import error | Install `libosmesa6` or use EGL |
| `GLFW` display error | No X/Wayland display; use EGL on a server |
| Image is all one colour | Geometry outside the camera; add a camera or use the default free camera |
| `AttributeError: Renderer ... depth` | You are on MuJoCo 4.x; use `enable_depth_rendering()` |

---

## 7. Logging and run directories

Adopt one convention now, before there are results to lose:

```text
runs/
└── bc/
    └── 20261112-213000/
        ├── config.yaml
        ├── metrics.csv
        ├── checkpoints/
        │   ├── last.pt
        │   └── best.pt
        └── eval/
            └── episode_0000.gif
```

- `config.yaml` is the *effective* config, including CLI overrides.
- `metrics.csv` has one row per log event; it is append-only.
- `checkpoints/` contains model, optimizer, normalizer and step.
- `eval/` contains only evaluation artifacts, never training data.

A small `training/logger.py` will be added in chapter 05. For now, create the
directories and write the config:

```python
from datetime import datetime
from pathlib import Path

from imitation_learning_reproduction.utils.config import config_to_dict
import yaml

def make_run_dir(root: str, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def dump_config(cfg, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
```

Do not put the timestamp in the YAML; it belongs in the directory name. Do not
overwrite an existing run directory; a failed experiment is still evidence.

---

## 8. Coding conventions for the rest of the tutorial

- **Array layout.** State is `(state_dim,)` per step; a batch of states is
  `(B, state_dim)`; a window of observations is `(T_o, state_dim)`; a batch of
  windows is `(B, T_o, state_dim)`. Images are `(3, H, W)` per frame and
  `(B, T_o, 3, H, W)` for a batch. Point clouds are `(N, 3)` per frame and
  `(B, T_o, N, 3)` for a batch. Write this in a docstring in `data/schema.py`.
- **Units.** Metres, radians, seconds. Never mix degrees into an observation.
- **Actions.** The environment action is a normalized delta joint target in
  `[-1, 1]`. The physical mapping is defined once in chapter 02. Supervised
  policies may normalize actions further for training, but the evaluation
  wrapper always converts back to `[-1, 1]` before calling `env.step`.
- **No global mutable state.** The normalizer lives in the policy checkpoint, not
  in a module global.
- **Type hints on public functions.** They are documentation that the type
  checker and your future self can read.
- **Tests next to the math.** If a function implements an equation with a subtle
  sign or index, write a hand-computed test. The gridworld IRL chapter and the
  diffusion scheduler chapter do this.

---

## 9. Checklist

- [ ] `uv sync` succeeds and does not create a second environment.
- [ ] `uv run python -c "import mujoco, gymnasium, h5py, torch"` succeeds.
- [ ] `MUJOCO_GL=egl` renders a 64×64 image; you know your fallback backend.
- [ ] `tests/test_smoke.py` and `tests/test_config.py` pass.
- [ ] The directory skeleton exists.
- [ ] `.gitignore` excludes `data/`, `runs/` and caches.
- [ ] You can explain why every experiment needs its own run directory.

### Exercises

1. Add a `--set` override for a list, e.g. `model.hidden=[128,128]`, and a test
   that it parses as a list of ints.
2. Add a `--dry-run` flag to the config example that prints the effective config
   and exits.
3. Write a `scripts/check_env.py` that prints Python, Torch, MuJoCo, Gymnasium and
   GPU availability, plus the result of the 64×64 render test. This is useful on
   a new machine.

Next: [02 · The MuJoCo task](02_simulation.md).
