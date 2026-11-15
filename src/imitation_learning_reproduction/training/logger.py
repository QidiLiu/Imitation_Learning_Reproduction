from datetime import datetime
from pathlib import Path

from imitation_learning_reproduction.utils.config import config_to_dict
import yaml


def make_run_dir(in_root: str, in_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(in_root) / in_name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dump_config(in_cfg, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(in_cfg), sort_keys=False))
