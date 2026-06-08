from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_path = config.pop("base", None)
    if base_path:
        base_config_path = Path(base_path)
        if not base_config_path.is_absolute():
            relative_base_path = config_path.parent / base_config_path
            base_config_path = (
                relative_base_path
                if relative_base_path.exists()
                else base_config_path
            )
        base_config = load_config(base_config_path)
        return deep_update(base_config, config)
    return config


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
