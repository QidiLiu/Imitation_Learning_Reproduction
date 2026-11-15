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
        apply_overrides(cfg, ["train.node=1"])
