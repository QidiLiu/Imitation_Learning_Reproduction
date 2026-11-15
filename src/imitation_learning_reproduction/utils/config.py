from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _build(in_tp: Any, in_value: Any) -> Any:
    """Recursively build a (possibly nested) dataclass from a YAML value."""
    if in_value is None:
        return None

    if dataclasses.is_dataclass(in_tp):
        if not isinstance(in_value, dict):
            raise TypeError(f"expected a mapping for {in_tp.__name__}, got {type(in_value).__name__}")
        hints = typing.get_type_hints(in_tp)
        kwargs = {}

        for field in dataclasses.fields(in_tp):
            if field.name in in_value:
                kwargs[field.name] = _build(hints[field.name], in_value[field.name])

        return in_tp(**kwargs)

    origin = typing.get_origin(in_tp)
    args = typing.get_args(in_tp)

    if origin in (typing.Union, types.UnionType):
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _build(candidate, in_value)
            except (TypeError, ValueError):
                continue

        return in_value

    if origin is list:
        item_tp = args[0] if args else Any
        return [_build(item_tp, item) for item in in_value]

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], item) for item in in_value)
        return tuple(_build(t, item) for t, item in zip(args, in_value, strict=False))

    if origin is dict:
        key_tp, val_tp = args if len(args) == 2 else (Any, Any)
        return {_build(key_tp, k): _build(val_tp, v) for k, v in in_value.items()}

    if in_tp is Path:
        return Path(in_value)

    if in_tp in (int, float, str, bool):
        return in_tp(in_value)

    return in_value


def load_config(in_path: str | Path, in_cls: type[T]) -> T:
    """Load YAML from path into a dataclass of type cls."""
    raw = yaml.safe_load(Path(in_path).read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{in_path} must contain a YAML mapping")
    return _build(in_cls, raw)


def apply_overrides(in_cfg: T, in_overrides: list[str] | None) -> T:
    """Apply `--set a.b=value` overrides in place and return cfg."""
    for item in in_overrides or []:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form `key=value`")
        value = yaml.safe_load(raw_value)
        target: Any = in_cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise AttributeError(f"unknown config field {key!r}")
        setattr(target, parts[-1], value)

    return in_cfg


def config_to_dict(in_cfg: Any) -> dict[str, Any]:
    """JSON/YAML - friendly view of a dataclass config."""
    def convert(in_value: Any) -> Any:
        if dataclasses.is_dataclass(in_value):
            return {f.name: convert(getattr(in_value, f.name)) for f in dataclasses.fields(in_value)}
        if isinstance(in_value, Path):
            return str(in_value)
        if isinstance(in_value, (list, tuple)):
            return [convert(v) for v in in_value]
        if isinstance(in_value, dict):
            return {k: convert(v) for k, v in in_value.items()}
        return in_value

    return convert(in_cfg)

