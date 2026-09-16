"""Pattern-local integer transforms used by HDF5 format v2.

The reference implementation deliberately has no Numba import. The optional
native kernels are selected only for the parallel fast path.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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


def _encode(values: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64], encoded: npt.NDArray[np.uint32]) -> None:
    if values.size:
        encoded[0] = values[0]
        encoded[1:] = values[1:] - values[:-1]
        starts = offsets[:-1]
        starts = starts[starts < offsets[1:]].astype(np.intp)
        encoded[starts] = values[starts]


def decode_segmented_delta_inplace(encoded: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64], decoded: npt.NDArray[np.uint32]) -> None:
    """Decode in place without requiring the optional Numba runtime."""
    for start, stop in zip(offsets[:-1], offsets[1:]):
        if start < stop:
            begin, end = int(start), int(stop)
            decoded[begin:end] = np.cumsum(encoded[begin:end], dtype=np.uint32)


def encode_pattern_local_delta(values: npt.ArrayLike, counts: npt.ArrayLike, *, check_sorted: bool = False) -> npt.NDArray[np.uint32]:
    values, offsets = _values(values), _offsets(counts)
    if int(offsets[-1]) != values.size:
        raise ValueError("pattern counts do not match the number of values")
    if check_sorted:
        _check(values, offsets)
    encoded = np.empty_like(values)
    _encode(values, offsets, encoded)
    return encoded


def encode_pattern_local_delta_parallel(values: npt.ArrayLike, counts: npt.ArrayLike, workers: int, *, check_sorted: bool = False) -> npt.NDArray[np.uint32]:
    """Use optional Numba kernels, falling back to the reference transform."""
    if workers <= 1:
        return encode_pattern_local_delta(values, counts, check_sorted=check_sorted)
    values, offsets = _values(values), _offsets(counts)
    if int(offsets[-1]) != values.size:
        raise ValueError("pattern counts do not match the number of values")
    if check_sorted:
        _check(values, offsets)
    encoded = np.empty_like(values)
    try:
        from ._delta_numba import encode_parallel
    except ModuleNotFoundError:
        _encode(values, offsets, encoded)
    else:
        encode_parallel(values, offsets, encoded, workers)
    return encoded


def decode_pattern_local_delta(encoded: npt.ArrayLike, counts: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    encoded, offsets = _values(encoded), _offsets(counts)
    if int(offsets[-1]) != encoded.size:
        raise ValueError("pattern counts do not match the number of encoded values")
    decoded = np.empty_like(encoded)
    decode_segmented_delta_inplace(encoded, offsets, decoded)
    return decoded
