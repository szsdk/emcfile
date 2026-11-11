"""Direct-chunk full scans for the standard HDF5 shuffle+Zstd layout."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from ._delta import decode_segmented_delta_inplace
from ._h5_accel import (
    H5AccelEntry,
    build_chunk_tables,
    get_entry_for_group,
    get_h5_rs,
)
from ._h5_workers import env_workers

MIN_FULL_SCAN_BYTES = 8 * 1024**2
_EMPTY_U64 = np.empty(0, dtype=np.uint64)


def _unshuffle_u32(src: Any, dst: Any, n: int) -> None:
    """Unshuffle 4 byte-planes of length *n* into *dst*."""
    valid = dst.size
    u = np.asarray(src, dtype=np.uint8).reshape(4, n)[:, :valid]
    dst_2d = dst.view(np.uint8).reshape(valid, 4)
    dst_2d[:, 0] = u[0]
    dst_2d[:, 1] = u[1]
    dst_2d[:, 2] = u[2]
    dst_2d[:, 3] = u[3]


def _eligible_dataset(dataset: h5py.Dataset) -> bool:
    if (
        sys.byteorder != "little"
        or dataset.is_virtual
        or dataset.ndim != 1
        or dataset.chunks is None
    ):
        return False
    if dataset.dtype.str not in ("<u4", "<i4"):
        return False
    plist = dataset.id.get_create_plist()
    if plist.get_nfilters() != 2:
        return False
    shuffle, zstd = plist.get_filter(0), plist.get_filter(1)
    n = dataset.chunks[0]
    return n > 0 and shuffle[0] == 2 and shuffle[2] == (4,) and zstd[0] == 32015


def _decode_chunk(
    mask: int,
    payload: bytes,
    n: int,
    *,
    decoder: Any = None,
    output: Any = None,
    selection: slice | None = None,
) -> npt.NDArray[np.uint32]:
    """Decode the exact two-filter layout, respecting per-chunk skip bits."""
    del decoder
    if mask & ~3:
        raise ValueError("Unexpected HDF5 v2 chunk filter mask")
    if (mask & 2) and len(payload) != n * 4:
        raise ValueError("Unexpected HDF5 v2 decoded chunk length")
    if selection is not None:
        start, stop, step = selection.indices(n)
        if step != 1:
            raise ValueError("Only unit-step slice selections are supported")
    else:
        start, stop = 0, (output.size if output is not None else n)
    rs = get_h5_rs()
    if rs is None:
        raise RuntimeError("Direct chunk decoding requires emcfile[fast]")
    decoded = rs.decode_raw_chunk(mask, memoryview(payload), n, start, stop)
    if output is not None:
        output[:] = decoded
        return output
    return decoded


def eligible(group: Any, *, datasets: list[h5py.Dataset] | None = None) -> bool:
    """Whether *group* uses the exact portable physical layout we own."""
    entry = get_entry_for_group(group)
    if entry is not None and entry.eligible is not None:
        return entry.eligible
    names = ("place_ones", "place_multi", "count_multi")
    if (
        str(group.attrs.get("version", "")) != "2"
        or str(group.attrs.get("position_encoding", "absolute")) != "delta"
        or any(name not in group for name in names)
    ):
        if entry is not None:
            entry.eligible = False
        return False
    if datasets is None:
        datasets = [group[name] for name in names]
    res = [dataset.dtype.str for dataset in datasets] == ["<u4", "<u4", "<i4"] and all(
        _eligible_dataset(dataset) for dataset in datasets
    )
    if entry is not None:
        entry.eligible = res
        if res:
            entry.chunk_lens = (
                int(datasets[0].chunks[0]),
                int(datasets[1].chunks[0]),
                int(datasets[2].chunks[0]),
            )
            entry.dataset_sizes = (
                int(datasets[0].size),
                int(datasets[1].size),
                int(datasets[2].size),
            )
    return res


def validate_offsets(
    ones_offsets: Any,
    multi_offsets: Any,
    sizes: tuple[int, int, int],
    entry: H5AccelEntry | None = None,
) -> None:
    """Validate pattern offset arrays, caching validation per object identity."""
    if (
        entry is not None
        and entry.validated_offsets is not None
        and entry.validated_offsets[0] is ones_offsets
        and entry.validated_offsets[1] is multi_offsets
        and not ones_offsets.flags.writeable
        and not multi_offsets.flags.writeable
        and ones_offsets.ndim == 1
        and multi_offsets.ndim == 1
        and ones_offsets.size > 0
        and ones_offsets.size == multi_offsets.size
        and int(ones_offsets[0]) == 0
        and int(multi_offsets[0]) == 0
        and int(ones_offsets[-1]) == sizes[0]
        and int(multi_offsets[-1]) == sizes[1]
        and int(multi_offsets[-1]) == sizes[2]
    ):
        return
    count = ones_offsets.size - 1
    for ds_size, offsets in zip(sizes, (ones_offsets, multi_offsets, multi_offsets)):
        if (
            offsets.ndim != 1
            or offsets.size == 0
            or offsets.size != count + 1
            or offsets[0] != 0
            or int(offsets[-1]) != ds_size
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("Pattern counts do not match HDF5 v2 payload size")
    if entry is not None:
        entry.validated_offsets = (ones_offsets, multi_offsets)


def _read(
    dataset: h5py.Dataset, pool: ThreadPoolExecutor, workers: int
) -> npt.NDArray[Any]:
    n = dataset.chunks[0]
    out = np.empty(dataset.size, dtype=np.uint32)

    def work(batch: list[tuple[int, int, bytes]]) -> None:
        for start, mask, payload in batch:
            dst = out[start : min(start + n, out.size)]
            _decode_chunk(mask, payload, n, output=dst)

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


def full_scan(
    group: Any, ones_offsets: Any, multi_offsets: Any
) -> (
    tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]] | None
):
    """Read an eligible complete v2 payload, otherwise return ``None``."""
    workers = env_workers("EMCFILE_H5_FULL_SCAN_WORKERS", 4, allow_zero=True)
    if workers <= 0 or not eligible(group):
        return None
    rs = get_h5_rs()
    if rs is None:
        return None

    entry = get_entry_for_group(group)
    datasets: list[h5py.Dataset] | None = None
    if entry is not None:
        if entry.tables is None or entry.mm is None:
            datasets = [
                group[name] for name in ("place_ones", "place_multi", "count_multi")
            ]
            build_chunk_tables(entry, datasets)
            if entry.eligible is False:
                return None
        if (
            entry.tables is not None
            and entry.mm is not None
            and entry.chunk_lens is not None
            and entry.dataset_sizes is not None
        ):
            sizes = entry.dataset_sizes
            if sum(s * 4 for s in sizes) < MIN_FULL_SCAN_BYTES:
                return None
            validate_offsets(ones_offsets, multi_offsets, sizes, entry)
            ones_u64 = np.ascontiguousarray(ones_offsets, dtype=np.uint64)
            multi_u64 = np.ascontiguousarray(multi_offsets, dtype=np.uint64)
            out_ones = np.empty(sizes[0], dtype=np.uint32)
            out_multi = np.empty(sizes[1], dtype=np.uint32)
            out_count = np.empty(sizes[2], dtype=np.uint32)
            rs.decode_dataset_mmap(
                entry.mm,
                entry.tables[0],
                entry.chunk_lens[0],
                out_ones,
                ones_u64,
                True,
                workers,
            )
            rs.decode_dataset_mmap(
                entry.mm,
                entry.tables[1],
                entry.chunk_lens[1],
                out_multi,
                multi_u64,
                True,
                workers,
            )
            rs.decode_dataset_mmap(
                entry.mm,
                entry.tables[2],
                entry.chunk_lens[2],
                out_count,
                _EMPTY_U64,
                False,
                workers,
            )
            return out_ones, out_multi, out_count.view(np.int32)

    if datasets is None:
        datasets = [
            group[name] for name in ("place_ones", "place_multi", "count_multi")
        ]
    if any(
        ds.chunks is None
        or ds.id.get_num_chunks() != (ds.size + ds.chunks[0] - 1) // ds.chunks[0]
        for ds in datasets
    ):
        if entry is not None:
            entry.eligible = False
        return None
    if (
        sum(dataset.size * dataset.dtype.itemsize for dataset in datasets)
        < MIN_FULL_SCAN_BYTES
    ):
        return None
    sizes = (int(datasets[0].size), int(datasets[1].size), int(datasets[2].size))
    validate_offsets(ones_offsets, multi_offsets, sizes, entry)
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dataset, offsets in zip(datasets, (ones_offsets, multi_offsets, None)):
            out = _read(dataset, pool, workers)
            if offsets is not None:
                boundaries = np.linspace(
                    0, offsets.size - 1, workers + 1, dtype=np.int64
                )
                futures = [
                    pool.submit(
                        decode_segmented_delta_inplace,
                        out,
                        offsets[start : stop + 1],
                        out,
                        accelerated=True,
                    )
                    for start, stop in zip(boundaries[:-1], boundaries[1:])
                    if start < stop
                ]
                for future in futures:
                    future.result()
            outputs.append(out)
    return outputs[0], outputs[1], outputs[2]
