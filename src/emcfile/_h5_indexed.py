"""Bounded direct-chunk indexed reads for standard shuffle+Zstd HDF5 v2.

Only the calling thread touches HDF5. Workers decode unique physical chunks
and scatter into disjoint output segments, including repeated selections.
"""

from __future__ import annotations

import importlib.util
import os
from collections import deque
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
from ._h5_full_scan import _decode_chunk, _unshuffle_u32, eligible, validate_offsets
from ._h5_workers import env_workers

__all__ = ["_unshuffle_u32", "indexed_read"]

MIN_INDEXED_ELEMENTS = 1024
_DEFAULT_DECODE_CHUNK = _decode_chunk
_ORIG_FIND_SPEC = importlib.util.find_spec


def _check_dependencies() -> bool:
    if importlib.util.find_spec is not _ORIG_FIND_SPEC:
        if importlib.util.find_spec("emcfile_fast") is None:
            return False
    return get_h5_rs() is not None


def _plan(ranges: Any, offsets: Any, chunk_len: int) -> tuple[Any, Any, Any]:
    """Map source chunks to output spans without expanding ranges into IDs."""
    chunks: dict[int, list[tuple[int, int, int]]] = {}
    boundaries = [0]
    destination = 0
    for first, last in ranges:
        first, last = int(first), int(last)
        begin, end = int(offsets[first]), int(offsets[last])
        boundaries.extend(
            (offsets[first + 1 : last + 1] - begin + destination).tolist()
        )
        while begin < end:
            chunk, local = divmod(begin, chunk_len)
            length = min(end - begin, chunk_len - local)
            chunks.setdefault(chunk * chunk_len, []).append(
                (destination, local, length)
            )
            begin += length
            destination += length
    return chunks, np.asarray(boundaries, dtype=np.uint64), destination


def indexed_read(group: Any, ranges: Any, ones_offsets: Any, multi_offsets: Any) -> Any:
    """Return decoded payloads, or ``None`` when the generic reader is needed.

    Single frames decode only selected byte-plane spans without workers/JIT.
    Multi-frame selections use parallel direct chunk decoding automatically on
    nontrivial datasets or when ``EMCFILE_H5_INDEXED_WORKERS`` is set.
    """
    ranges = np.asarray(ranges)
    single = ranges.shape == (1, 2) and int(ranges[0, 1]) == int(ranges[0, 0]) + 1
    if single:
        default_workers = 1
    elif "EMCFILE_H5_INDEXED_WORKERS" in os.environ:
        default_workers = 0
    else:
        total_elems = (
            int(ones_offsets[-1]) if getattr(ones_offsets, "size", 0) > 0 else 0
        ) + (int(multi_offsets[-1]) if getattr(multi_offsets, "size", 0) > 0 else 0)
        default_workers = 4 if total_elems >= MIN_INDEXED_ELEMENTS else 0

    workers = env_workers(
        "EMCFILE_H5_INDEXED_WORKERS", default_workers, allow_zero=True
    )
    if not workers:
        return None

    entry = get_entry_for_group(group)
    datasets: list[h5py.Dataset] | None = None
    if entry is not None and entry.eligible is not None:
        if not entry.eligible:
            return None
    else:
        datasets = [
            group[name] for name in ("place_ones", "place_multi", "count_multi")
        ]
        if not eligible(group, datasets=datasets):
            return None

    if not _check_dependencies():
        return None

    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("Pattern ranges must have shape (N, 2)")
    count = ones_offsets.size - 1
    if single:
        frame = int(ranges[0, 0])
        if frame < 0 or frame + 1 > count:
            raise IndexError("Pattern range is out of bounds")
    else:
        if (
            np.any(ranges < 0)
            or np.any(ranges[:, 1] > count)
            or np.any(ranges[:, 0] > ranges[:, 1])
        ):
            raise IndexError("Pattern range is out of bounds")

    if entry is not None and entry.dataset_sizes is not None:
        sizes = entry.dataset_sizes
    else:
        if datasets is None:
            datasets = [
                group[name] for name in ("place_ones", "place_multi", "count_multi")
            ]
        sizes = (int(datasets[0].size), int(datasets[1].size), int(datasets[2].size))
    validate_offsets(ones_offsets, multi_offsets, sizes, entry)

    if single:
        return _single_frame(
            datasets,
            int(ranges[0, 0]),
            ones_offsets,
            multi_offsets,
            entry=entry,
            group=group,
        )

    rs = get_h5_rs()
    if _decode_chunk is _DEFAULT_DECODE_CHUNK and rs is not None and entry is not None:
        if entry.tables is None or entry.mm is None:
            if datasets is None:
                datasets = [
                    group[name] for name in ("place_ones", "place_multi", "count_multi")
                ]
            build_chunk_tables(entry, datasets)
        if (
            entry.tables is not None
            and entry.mm is not None
            and entry.chunk_lens is not None
        ):
            ranges_i64 = np.ascontiguousarray(ranges, dtype=np.int64)
            ones_u64 = np.ascontiguousarray(ones_offsets, dtype=np.uint64)
            multi_u64 = np.ascontiguousarray(multi_offsets, dtype=np.uint64)
            out_ones = rs.decode_indexed_mmap(
                entry.mm,
                entry.tables[0],
                entry.chunk_lens[0],
                ranges_i64,
                ones_u64,
                True,
                workers,
            )
            out_multi = rs.decode_indexed_mmap(
                entry.mm,
                entry.tables[1],
                entry.chunk_lens[1],
                ranges_i64,
                multi_u64,
                True,
                workers,
            )
            out_count = rs.decode_indexed_mmap(
                entry.mm,
                entry.tables[2],
                entry.chunk_lens[2],
                ranges_i64,
                multi_u64,
                False,
                workers,
            )
            return out_ones, out_multi, out_count.view(np.int32)

    if datasets is None:
        datasets = [
            group[name] for name in ("place_ones", "place_multi", "count_multi")
        ]
    decode_segmented_delta_inplace(
        np.empty(0, "u4"), np.array([0], "u8"), np.empty(0, "u4"), accelerated=True
    )
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (dataset, offsets) in enumerate(
            zip(datasets, (ones_offsets, multi_offsets, multi_offsets))
        ):
            n = dataset.chunks[0]
            chunks, boundaries, size = _plan(ranges, offsets, n)
            output = np.empty(size, dtype=dataset.dtype)

            def work(
                mask: int,
                payload: bytes,
                spans: Any,
                target: Any = output,
                length: int = n,
            ) -> None:
                block = _decode_chunk(mask, payload, length)
                for destination, source, span in spans:
                    target[destination : destination + span] = block[
                        source : source + span
                    ]

            pending: Any = deque()
            for start, spans in sorted(chunks.items()):
                mask, payload = dataset.id.read_direct_chunk((start,))
                pending.append(pool.submit(work, mask, payload, spans))
                if len(pending) >= workers * 2:
                    pending.popleft().result()
            for future in pending:
                future.result()
            if index < 2:
                split = np.linspace(0, boundaries.size - 1, workers + 1, dtype=np.int64)
                futures = [
                    pool.submit(
                        decode_segmented_delta_inplace,
                        output,
                        boundaries[first : last + 1],
                        output,
                        accelerated=True,
                    )
                    for first, last in zip(split[:-1], split[1:])
                    if first < last
                ]
                for future in futures:
                    future.result()
            outputs.append(output)
    return tuple(outputs)


def _single_frame(
    datasets: list[h5py.Dataset] | None,
    frame: int,
    ones_offsets: npt.NDArray[np.uint64],
    multi_offsets: npt.NDArray[np.uint64],
    *,
    entry: H5AccelEntry | None = None,
    group: Any = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    rs = get_h5_rs()
    if _decode_chunk is _DEFAULT_DECODE_CHUNK and rs is not None and entry is not None:
        if entry.tables is None or entry.mm is None:
            if datasets is None and group is not None:
                datasets = [
                    group[name] for name in ("place_ones", "place_multi", "count_multi")
                ]
            if datasets is not None:
                build_chunk_tables(entry, datasets)
        if (
            entry.tables is not None
            and entry.mm is not None
            and entry.chunk_lens is not None
        ):
            begin_o = int(ones_offsets[frame])
            end_o = int(ones_offsets[frame + 1])
            begin_m = int(multi_offsets[frame])
            end_m = int(multi_offsets[frame + 1])
            p_ones, p_multi, c_multi = rs.decode_single_frame_mmap(
                entry.file_token,
                entry.mm,
                entry.tables[0],
                entry.chunk_lens[0],
                begin_o,
                end_o,
                entry.tables[1],
                entry.chunk_lens[1],
                begin_m,
                end_m,
                entry.tables[2],
                entry.chunk_lens[2],
                True,
            )
            return p_ones, p_multi, c_multi.view(np.int32)

    if datasets is None:
        assert group is not None
        datasets = [
            group[name] for name in ("place_ones", "place_multi", "count_multi")
        ]
    outputs = []
    for index, (dataset, offsets) in enumerate(
        zip(datasets, (ones_offsets, multi_offsets, multi_offsets))
    ):
        begin, end = int(offsets[frame]), int(offsets[frame + 1])
        output = np.empty(end - begin, dtype=dataset.dtype)
        n = dataset.chunks[0]
        destination = 0
        while begin < end:
            chunk, start = divmod(begin, n)
            length = min(end - begin, n - start)
            mask, payload = dataset.id.read_direct_chunk((chunk * n,))
            output[destination : destination + length] = _decode_chunk(
                mask,
                payload,
                n,
                selection=slice(start, start + length),
            ).view(dataset.dtype)
            begin += length
            destination += length
        if index < 2:
            np.cumsum(output, dtype=np.uint32, out=output)
        outputs.append(output)
    return outputs[0], outputs[1], outputs[2]
