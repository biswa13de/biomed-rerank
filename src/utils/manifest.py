"""Run manifest — records the exact conditions that produced a set of results.

The brief's Experimental Constraint clause requires model, dataset, hardware, and
inference settings to remain consistent across configurations. A claim that they were
held constant is only credible if the conditions were recorded at run time, so every
run writes library versions, hardware, and resolved config to
``results/logs/run_manifest.json``.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRACKED_LIBRARIES = (
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "sentence_transformers",
    "rank_bm25",
    "pytrec_eval",
    "scipy",
)


def _library_versions() -> dict[str, str]:
    import importlib

    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        info["torch_device_available"] = {
            "mps": bool(torch.backends.mps.is_available()),
            "cuda": bool(torch.cuda.is_available()),
        }
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def build_manifest(
    run_label: str,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest for one run."""
    return {
        "run_label": run_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": _hardware(),
        "libraries": _library_versions(),
        "config": config or {},
        "extra": extra or {},
    }


def write_manifest(
    run_label: str,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append this run's manifest to ``run_manifest.json`` and return the file path.

    Appending rather than overwriting keeps the provenance of earlier results after a
    re-run — useful when the final execution happens in the lab environment and we
    need to show the same code produced the same numbers on different hardware.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"

    history: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            history = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            history = []

    history.append(build_manifest(run_label, config, extra))
    manifest_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return manifest_path
