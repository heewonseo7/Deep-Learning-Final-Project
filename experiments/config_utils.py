"""Load YAML configs for experiment scripts."""

from pathlib import Path

import yaml


def load_yaml_config(path: Path) -> dict:
    """Load a YAML experiment config; raise if missing or empty."""
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Empty or invalid YAML: {path}")
    return cfg
