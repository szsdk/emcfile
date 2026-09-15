"""Pattern-local integer transforms used by HDF5 format v2."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange, set_num_threads


@njit(nogil=True)
def _encode_segmented_delta_njit(values: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64], encoded: npt.NDArray[np.uint32]) -> None:
    if values.size == 0:
        return
    encoded[0] = values[0]
    for index in range(1, values.size):
        encoded[index] = values[index] - values[index - 1]
    for pattern in range(offsets.size - 1):
        start, stop = offsets[pattern], offsets[pattern + 1]
        if start < stop:
            encoded[start] = values[start]


@njit(nogil=True, parallel=True)
def _encode_segmented_delta_parallel_njit(values: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64], encoded: npt.NDArray[np.uint32]) -> None:
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
def _decode_segmented_delta_njit(encoded: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64], decoded: npt.NDArray[np.uint32]) -> None:
    for pattern in range(offsets.size - 1):
        start, stop = offsets[pattern], offsets[pattern + 1]
        if start < stop:
            value = encoded[start]
            decoded[start] = value
            for index in range(start + 1, stop):
                value += encoded[index]
                decoded[index] = value


def _offsets(counts: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    counts = np.asarray(counts)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("pattern counts must be a one-dimensional non-negative array")
    offsets = np.empty(counts.size + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.uint64, out=offsets[1:])
    return offsets


def _values(values: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("position values must be a one-dimensional integer array")
    if values.dtype == np.uint32:
        return values
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
        raise ValueError("position values must fit in uint32")
    return values.astype(np.uint32, copy=False)


def _check(values: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64]) -> None:
    decreases = values[1:] < values[:-1]
    starts = offsets[np.flatnonzero(np.diff(offsets))]
    decreases[starts[starts > 0].astype(np.intp) - 1] = False
    if np.any(decreases):
        raise ValueError("position values must be sorted within every pattern")


def encode_pattern_local_delta(values: npt.ArrayLike, counts: npt.ArrayLike, *, check_sorted: bool = False) -> npt.NDArray[np.uint32]:
    """Delta encode values, resetting at each non-empty pattern."""
    values, offsets = _values(values), _offsets(counts)
    if int(offsets[-1]) != values.size:
        raise ValueError("pattern counts do not match the number of values")
    if check_sorted:
        _check(values, offsets)
    encoded = np.empty_like(values)
    _encode_segmented_delta_njit(values, offsets, encoded)
    return encoded


def encode_pattern_local_delta_parallel(values: npt.ArrayLike, counts: npt.ArrayLike, workers: int, *, check_sorted: bool = False) -> npt.NDArray[np.uint32]:
    """Native-threaded encoder; sortedness checking is explicitly opt-in."""
    if workers <= 1:
        return encode_pattern_local_delta(values, counts, check_sorted=check_sorted)
    values, offsets = _values(values), _offsets(counts)
    if int(offsets[-1]) != values.size:
        raise ValueError("pattern counts do not match the number of values")
    if check_sorted:
        _check(values, offsets)
    encoded = np.empty_like(values)
    set_num_threads(min(workers, config.NUMBA_NUM_THREADS))  # type: ignore[attr-defined]
    _encode_segmented_delta_parallel_njit(values, offsets, encoded)
    return encoded


def decode_pattern_local_delta(encoded: npt.ArrayLike, counts: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Invert pattern-local uint32 delta encoding."""
    encoded, offsets = _values(encoded), _offsets(counts)
    if int(offsets[-1]) != encoded.size:
        raise ValueError("pattern counts do not match the number of encoded values")
    decoded = np.empty_like(encoded)
    _decode_segmented_delta_njit(encoded, offsets, decoded)
    return decoded
