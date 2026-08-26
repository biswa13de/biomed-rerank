"""Configuration loader — the single source of truth for experimental constants.

Every module reads its settings from here rather than hard-coding them, so that the
brief's Experimental Constraint (identical model / dataset / hardware / inference
settings across configurations) is enforced structurally rather than by discipline.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.yaml"


class Config:
    """Dotted-path access over the YAML config, with paths resolved to absolutes."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        self._data = data
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. ``cfg.get("rerank.batch_size")``."""
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, path: str) -> Any:
        """Like :meth:`get`, but fails loudly rather than silently defaulting."""
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {path!r} in {CONFIG_PATH}")
        return value

    def path(self, path: str) -> Path:
        """Resolve a config value that names a directory, relative to the project root."""
        resolved = self.root / str(self.require(path))
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self) -> dict[str, Any]:
        return self._data


@functools.lru_cache(maxsize=1)
def load_config(config_path: Path | str = CONFIG_PATH) -> Config:
    """Load and cache the experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve_device(requested: str = "auto") -> str:
    """Pick the torch device.

    ``auto`` prefers Apple MPS when present, else CPU. The resolved value is written
    into the run manifest so the report can state the hardware that produced every
    number — the Experimental Constraint clause requires hardware to be held constant,
    which we can only evidence if we record it.
    """
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
