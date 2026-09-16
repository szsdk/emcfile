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


@njit(nogil=True)
def _decode(encoded: np.ndarray, offsets: np.ndarray, decoded: np.ndarray) -> None:
    for pattern in range(offsets.size - 1):
        start, stop = offsets[pattern], offsets[pattern + 1]
        if start < stop:
            value = encoded[start]
            decoded[start] = value
            for index in range(start + 1, stop):
                value += encoded[index]
                decoded[index] = value


def encode_parallel(values: np.ndarray, offsets: np.ndarray, encoded: np.ndarray, workers: int) -> None:
    """Run with a temporary Numba thread mask and always restore it."""
    old = get_num_threads()
    try:
        set_num_threads(min(workers, config.NUMBA_NUM_THREADS))  # type: ignore[attr-defined]
        _encode_parallel(values, offsets, encoded)
    finally:
        set_num_threads(old)


def decode_inplace(encoded: np.ndarray, offsets: np.ndarray, decoded: np.ndarray) -> None:
    _decode(encoded, offsets, decoded)
