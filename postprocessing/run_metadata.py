"""Small JSON sidecars for replaceable postprocessing outputs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def git_revision(repo: str | Path) -> str:
    """Return the current Git revision, including a dirty marker."""
    root = Path(repo)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    return revision + ("-dirty" if dirty else "")


def write_sidecar(
    output: str | Path,
    *,
    repo: str | Path,
    inputs,
    configuration,
    seed: int | None = None,
    extra: dict | None = None,
) -> Path:
    """Write ``<output>.json`` with the declared inputs and run settings."""
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + ".json")
    payload = {
        "output": str(output),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(repo),
        "inputs": [str(Path(item)) for item in inputs],
        "configuration": _json_value(configuration),
        "seed": seed,
    }
    if extra:
        payload.update(_json_value(extra))
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sidecar
