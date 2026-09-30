"""Bounded direct-chunk indexed reads for standard shuffle+Zstd HDF5 v2.

Only the calling thread touches HDF5. Workers decode unique physical chunks
and scatter into disjoint output segments, including repeated selections.
"""

from __future__ import annotations

import importlib.util
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import local
from typing import Any

import numpy as np

from ._delta import decode_segmented_delta_inplace
from ._h5_full_scan import _decode_chunk, _unshuffle_u32, eligible
from ._h5_workers import env_workers


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

    Environment budget zero disables this path. Layout checks are shared with
    the full-scan reader; missing optional dependencies never require a fork.
    """
    workers = env_workers("EMCFILE_H5_INDEXED_WORKERS", 0, allow_zero=True)
    if not workers or not eligible(group):
        return None
    if any(importlib.util.find_spec(name) is None for name in ("numba", "zstandard")):
        return None
    ranges = np.asarray(ranges)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("Pattern ranges must have shape (N, 2)")
    count = ones_offsets.size - 1
    if (
        np.any(ranges < 0)
        or np.any(ranges[:, 1] > count)
        or np.any(ranges[:, 0] > ranges[:, 1])
    ):
        raise IndexError("Pattern range is out of bounds")
    datasets = [group[name] for name in ("place_ones", "place_multi", "count_multi")]
    for dataset, offsets in zip(datasets, (ones_offsets, multi_offsets, multi_offsets)):
        if (
            offsets.ndim != 1
            or offsets.size != count + 1
            or offsets[0] != 0
            or int(offsets[-1]) != dataset.size
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("Pattern counts do not match HDF5 v2 payload size")
    _unshuffle_u32(np.zeros(4, "u1"), np.empty(1, "u4"), 1)
    # Compile the delta kernel before worker dispatch as well.
    decode_segmented_delta_inplace(
        np.empty(0, "u4"), np.array([0], "u8"), np.empty(0, "u4"), accelerated=True
    )
    outputs = []
    import zstandard

    thread_state = local()
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
                if not hasattr(thread_state, "decoder"):
                    thread_state.decoder = zstandard.ZstdDecompressor()
                block = _decode_chunk(
                    mask, payload, length, decoder=thread_state.decoder
                )
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
