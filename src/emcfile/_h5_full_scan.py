"""Direct-chunk full scans for the standard HDF5 shuffle+Zstd layout."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import zstandard
from numba import njit

from ._delta import _decode_segmented_delta_njit

MIN_FULL_SCAN_BYTES = 8 * 1024**2


@njit(nogil=True)
def _unshuffle_u32(src: Any, dst: Any, n: int) -> None:
    for i in range(dst.size):
        dst[i] = (np.uint32(src[i]) | (np.uint32(src[n + i]) << 8)
                  | (np.uint32(src[2 * n + i]) << 16) | (np.uint32(src[3 * n + i]) << 24))


def _eligible_dataset(dataset: h5py.Dataset) -> bool:
    if sys.byteorder != "little" or dataset.is_virtual or dataset.ndim != 1 or dataset.chunks is None:
        return False
    if dataset.dtype.str not in ("<u4", "<i4"):
        return False
    plist = dataset.id.get_create_plist()
    if plist.get_nfilters() != 2:
        return False
    shuffle, zstd = plist.get_filter(0), plist.get_filter(1)
    n = dataset.chunks[0]
    return shuffle[0] == 2 and shuffle[2] == (4,) and zstd[0] == 32015 and dataset.id.get_num_chunks() == (dataset.size + n - 1) // n


def eligible(group: Any) -> bool:
    """Whether *group* uses the exact portable physical layout we own."""
    names = ("place_ones", "place_multi", "count_multi")
    if str(group.attrs.get("version", "")) != "2" or str(group.attrs.get("position_encoding", "absolute")) != "delta" or any(name not in group for name in names):
        return False
    datasets = [group[name] for name in names]
    return [dataset.dtype.str for dataset in datasets] == ["<u4", "<u4", "<i4"] and all(_eligible_dataset(dataset) for dataset in datasets)


def _read(dataset: h5py.Dataset, pool: ThreadPoolExecutor, workers: int) -> npt.NDArray[Any]:
    n = dataset.chunks[0]
    out = np.empty(dataset.size, dtype=np.uint32)
    def work(batch: list[tuple[int, int, bytes]]) -> None:
        decoder = zstandard.ZstdDecompressor()
        for start, mask, payload in batch:
            if mask & ~3:
                raise ValueError("Unexpected HDF5 v2 chunk filter mask")
            raw = payload if mask & 2 else decoder.decompress(payload, max_output_size=n * 4)
            if len(raw) != n * 4:
                raise ValueError("Unexpected HDF5 v2 decoded chunk length")
            dst = out[start : min(start + n, out.size)]
            if mask & 1:
                dst[:] = np.frombuffer(raw, "<u4")[:dst.size]
            else:
                _unshuffle_u32(np.frombuffer(raw, "u1"), dst, n)
    pending, batch = [], []
    batch_size = max(1, min(64, 4 * 1024**2 // (n * 4)))
    for start in range(0, out.size, n):
        mask, payload = dataset.id.read_direct_chunk((start,))
        batch.append((start, mask, payload))
        if len(batch) == batch_size:
            pending.append(pool.submit(work, batch))
            batch = []
            if len(pending) >= workers * 2:
                pending.pop(0).result()
    if batch:
        pending.append(pool.submit(work, batch))
    for future in pending:
        future.result()
    return out.view(dataset.dtype)


def full_scan(group: Any, ones_offsets: Any, multi_offsets: Any) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]] | None:
    """Read an eligible complete v2 payload, otherwise return ``None``."""
    workers = int(os.environ.get("EMCFILE_H5_FULL_SCAN_WORKERS", "4"))
    if workers <= 0 or not eligible(group):
        return None
    datasets = [group[name] for name in ("place_ones", "place_multi", "count_multi")]
    if sum(dataset.size * dataset.dtype.itemsize for dataset in datasets) < MIN_FULL_SCAN_BYTES:
        return None
    for dataset, offsets in zip(datasets, (ones_offsets, multi_offsets, multi_offsets)):
        if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0 or int(offsets[-1]) != dataset.size or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Pattern counts do not match HDF5 v2 payload size")
    workers = min(workers, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    _unshuffle_u32(np.frombuffer(bytes(4), "u1"), np.empty(1, "u4"), 1)
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dataset, offsets in zip(datasets, (ones_offsets, multi_offsets, None)):
            out = _read(dataset, pool, workers)
            if offsets is not None:
                boundaries = np.linspace(0, offsets.size - 1, workers + 1, dtype=np.int64)
                futures = [pool.submit(_decode_segmented_delta_njit, out, offsets[start:stop + 1], out) for start, stop in zip(boundaries[:-1], boundaries[1:]) if start < stop]
                for future in futures:
                    future.result()
            outputs.append(out)
    return outputs[0], outputs[1], outputs[2]
