"""Small shared worker-budget helpers for HDF5 fast paths."""

from __future__ import annotations

import os


def effective_workers(requested: int, *, allow_zero: bool = False) -> int:
    """Clamp a requested budget to the CPUs available to this process."""
    if requested < 0 or (requested == 0 and not allow_zero):
        raise ValueError("HDF5 worker count must be positive")
    if requested == 0:
        return 0
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return min(requested, max(1, available))


def env_workers(name: str, default: int, *, allow_zero: bool = False) -> int:
    """Read and validate an HDF5 worker environment variable."""
    try:
        requested = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return effective_workers(requested, allow_zero=allow_zero)
