from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_env_config(base_path: str | Path, difficulty: str | None = None) -> dict[str, Any]:
    cfg = load_yaml(base_path)
    if difficulty is not None:
        cfg["difficulty"] = difficulty
    return cfg


def set_by_dotted_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if not parts:
        raise ValueError("Override path must not be empty.")

    current = payload
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected dotted.path=value.")
    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override '{override}'. Override key must not be empty.")
    return key, yaml.safe_load(raw_value)


def apply_overrides(base: dict[str, Any], overrides: list[str] | None = None) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for override in overrides or []:
        key, value = parse_override(override)
        set_by_dotted_path(merged, key, value)
    return merged
