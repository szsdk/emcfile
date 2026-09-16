"""Optional Numba kernels for pattern-local deltas."""

from __future__ import annotations

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads


@njit(nogil=True, parallel=True)
def _encode_parallel(values: np.ndarray, offsets: np.ndarray, encoded: np.ndarray) -> None:
    if values.size == 0:
        return
    encoded[0] = values[0]
    for index in prange(1, values.size):  # type: ignore[attr-defined]
        encoded[index] = values[index] - values[index - 1]
    for pattern in range(offsets.size - 1):
        start, stop = offsets[pattern], offsets[pattern + 1]
        if start < stop:
            encoded[start] = values[start]


def encode_parallel(values: np.ndarray, offsets: np.ndarray, encoded: np.ndarray, workers: int) -> None:
    """Run with a temporary Numba thread mask and always restore it."""
    old = get_num_threads()
    try:
        set_num_threads(min(workers, config.NUMBA_NUM_THREADS))  # type: ignore[attr-defined]
        _encode_parallel(values, offsets, encoded)
    finally:
        set_num_threads(old)
