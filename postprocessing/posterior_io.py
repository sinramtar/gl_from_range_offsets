"""Read retained Bayesian-inversion result files."""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd


PAIR_DIAGNOSTICS = {
    "diff_tides": lambda result: result["max_kdes"][2],
    "stddev": lambda result: result["stds"][3],
    "nESS": lambda result: result["nESS"],
    "ks_dist": lambda result: result["ks_distance"][3],
    "ks_pval": lambda result: result["ks_pvalue"][3],
    "evidence": lambda result: np.float32(result["evidence"]),
}


def _numpy_pickle_compatibility() -> None:
    """Provide the NumPy 2 pickle module names when reading under NumPy 1."""
    if "numpy._core" in sys.modules:
        return
    sys.modules["numpy._core"] = np.core
    for name in ("multiarray", "umath", "numeric", "numerictypes", "_multiarray_umath"):
        try:
            sys.modules[f"numpy._core.{name}"] = importlib.import_module(f"numpy.core.{name}")
        except ImportError:
            pass


def load_result(path: str | Path) -> dict:
    """Load one zstd-compressed BIP result."""
    _numpy_pickle_compatibility()
    with Path(path).open("rb") as stream:
        payload = zstd.ZstdDecompressor(max_window_size=2**31).decompress(stream.read())
    return pickle.loads(payload)


def pair_diagnostics(result: dict) -> dict[str, float]:
    """Return the diagnostics carried by a BIP result."""
    if "Null" in result:
        return {name: np.nan for name in PAIR_DIAGNOSTICS}
    return {name: float(extract(result)) for name, extract in PAIR_DIAGNOSTICS.items()}


def posterior_samples(result: dict) -> np.ndarray | None:
    """Return equally weighted ``x_GL`` draws, or ``None`` for a null result."""
    if "Null" in result:
        return None
    return np.asarray(result["samples"], dtype=float)[:, 3]


def result_times(path: str | Path) -> tuple[str, str]:
    """Return the two acquisition dates encoded by a result filename."""
    stem = Path(path).name.removesuffix(".zstd")
    fields = stem.split("_")
    if len(fields) != 2:
        raise ValueError(f"BIP filename must be PRIMARY_SECONDARY.zstd, got {stem!r}")
    return fields[0], fields[1]
