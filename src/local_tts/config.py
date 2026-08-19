"""TOML configuration loading for reproducible reading preferences."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load a small, strictly sectioned TOML config when one is requested."""
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML config {path}: {exc}") from exc
    if not isinstance(raw, dict):  # pragma: no cover - tomllib always returns a dict
        raise ValueError("config must contain TOML tables")
    sections: dict[str, dict[str, Any]] = {}
    for name, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"config section [{name}] must be a table")
        sections[name] = values
    return sections


def config_value(config: dict[str, dict[str, Any]], section: str, key: str, fallback: Any) -> Any:
    """Return a config value, leaving CLI values authoritative when present."""
    return config.get(section, {}).get(key, fallback)
