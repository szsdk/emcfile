"""Bounded direct-chunk indexed reads for standard shuffle+Zstd HDF5 v2.

Only the calling thread touches HDF5. Workers decode unique physical chunks
and scatter into disjoint output segments, including repeated selections.
"""

from __future__ import annotations

import importlib.util
import os
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import local
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from ._delta import decode_segmented_delta_inplace
from ._h5_full_scan import _decode_chunk, _unshuffle_u32, eligible
from ._h5_workers import env_workers

_THREAD_STATE = local()


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


def indexed_read(
    group: Any,
    ranges: Any,
    ones_offsets: Any,
    multi_offsets: Any,
    default_workers: int = 0,
) -> Any:
    """Return decoded payloads, or ``None`` when the generic reader is needed.

    Single frames decode only selected byte-plane spans without workers/JIT.
    Other selections remain opt-in for direct callers unless *default_workers*
    is supplied. An explicit budget zero disables both.
    """
    ranges = np.asarray(ranges)
    single = ranges.shape == (1, 2) and ranges[0, 1] == ranges[0, 0] + 1
    if single and "EMCFILE_H5_INDEXED_WORKERS" not in os.environ:
        workers = 1
    else:
        workers = env_workers(
            "EMCFILE_H5_INDEXED_WORKERS",
            1 if single else default_workers,
            allow_zero=True,
        )
    if not workers:
        return None
    cached_ref = getattr(_THREAD_STATE, "cached_group_ref", None)
    if cached_ref is not None and cached_ref() is group and bool(group.id):
        datasets = _THREAD_STATE.cached_datasets
        is_eligible = _THREAD_STATE.cached_eligible
        chunk_lens = _THREAD_STATE.cached_chunk_lens
        dtypes = _THREAD_STATE.cached_dtypes
    else:
        names = ("place_ones", "place_multi", "count_multi")
        if any(name not in group for name in names):
            return None
        datasets = [group[name] for name in names]
        is_eligible = eligible(group, datasets=datasets)
        chunk_lens = tuple(d.chunks[0] for d in datasets) if is_eligible else (0, 0, 0)
        dtypes = tuple(d.dtype for d in datasets)
        try:
            _THREAD_STATE.cached_group_ref = weakref.ref(group)
            _THREAD_STATE.cached_datasets = datasets
            _THREAD_STATE.cached_eligible = is_eligible
            _THREAD_STATE.cached_chunk_lens = chunk_lens
            _THREAD_STATE.cached_dtypes = dtypes
        except TypeError:
            pass
    if not is_eligible:
        return None
    if single:
        if getattr(_THREAD_STATE, "has_zstd", None) is None or (
            importlib.util.find_spec is not getattr(_THREAD_STATE, "find_spec_fn", None)
        ):
            _THREAD_STATE.has_zstd = importlib.util.find_spec("zstandard") is not None
            _THREAD_STATE.find_spec_fn = importlib.util.find_spec
        if not _THREAD_STATE.has_zstd:
            return None
    else:
        if any(
            importlib.util.find_spec(name) is None for name in ("numba", "zstandard")
        ):
            return None
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("Pattern ranges must have shape (N, 2)")
    count = ones_offsets.size - 1
    if single:
        f0 = int(ranges[0, 0])
        if f0 < 0 or f0 + 1 > count:
            raise IndexError("Pattern range is out of bounds")
    elif (
        np.any(ranges < 0)
        or np.any(ranges[:, 1] > count)
        or np.any(ranges[:, 0] > ranges[:, 1])
    ):
        raise IndexError("Pattern range is out of bounds")
    offsets_key = (
        id(ones_offsets),
        ones_offsets.size,
        int(ones_offsets[0]) if ones_offsets.size else -1,
        int(ones_offsets[-1]) if ones_offsets.size else -1,
        id(multi_offsets),
        multi_offsets.size,
        int(multi_offsets[0]) if multi_offsets.size else -1,
        int(multi_offsets[-1]) if multi_offsets.size else -1,
        datasets[0].size,
        datasets[1].size,
        datasets[2].size,
    )
    if getattr(_THREAD_STATE, "cached_offsets_key", None) != offsets_key:
        for dataset, offsets in zip(
            datasets, (ones_offsets, multi_offsets, multi_offsets)
        ):
            if (
                offsets.ndim != 1
                or offsets.size != count + 1
                or offsets[0] != 0
                or int(offsets[-1]) != dataset.size
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError("Pattern counts do not match HDF5 v2 payload size")
        _THREAD_STATE.cached_offsets_key = offsets_key
    if single:
        return _single_frame(
            datasets,
            int(ranges[0, 0]),
            ones_offsets,
            multi_offsets,
            chunk_lens=chunk_lens,
            dtypes=dtypes,
        )
    _unshuffle_u32(np.frombuffer(bytes(4), "u1"), np.empty(1, "u4"), 1)
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
            if size == 0:
                outputs.append(output)
                continue

            def work_batch(
                batch: list[tuple[int, bytes, Any]],
                target: Any = output,
                length: int = n,
            ) -> None:
                decoder = getattr(thread_state, "decoder", None)
                if decoder is None:
                    decoder = zstandard.ZstdDecompressor()
                    thread_state.decoder = decoder
                for mask, payload, spans in batch:
                    _decode_chunk(
                        mask,
                        payload,
                        length,
                        decoder=decoder,
                        output=target,
                        spans=spans,
                    )

            pending: Any = deque()
            batch: list[tuple[int, bytes, Any]] = []
            batch_size = max(
                1, min(16, (len(chunks) + workers * 4 - 1) // (workers * 4))
            )
            for start, spans in sorted(chunks.items()):
                mask, payload = dataset.id.read_direct_chunk((start,))
                batch.append((mask, payload, spans))
                if len(batch) == batch_size:
                    pending.append(pool.submit(work_batch, batch))
                    batch = []
                    if len(pending) >= workers * 3:
                        pending.popleft().result()
            if batch:
                pending.append(pool.submit(work_batch, batch))
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
    datasets: list[h5py.Dataset],
    frame: int,
    ones_offsets: npt.NDArray[np.uint64],
    multi_offsets: npt.NDArray[np.uint64],
    *,
    chunk_lens: tuple[int, int, int] | None = None,
    dtypes: tuple[Any, Any, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    import zstandard

    decoder = getattr(_THREAD_STATE, "decoder", None)
    if decoder is None:
        decoder = zstandard.ZstdDecompressor()
        _THREAD_STATE.decoder = decoder
    if chunk_lens is None:
        chunk_lens = (
            datasets[0].chunks[0],
            datasets[1].chunks[0],
            datasets[2].chunks[0],
        )
    if dtypes is None:
        dtypes = (datasets[0].dtype, datasets[1].dtype, datasets[2].dtype)
    outputs = []
    for index, (dataset, offsets, n, dt) in enumerate(
        zip(
            datasets,
            (ones_offsets, multi_offsets, multi_offsets),
            chunk_lens,
            dtypes,
        )
    ):
        begin, end = int(offsets[frame]), int(offsets[frame + 1])
        output = np.empty(end - begin, dtype=dt)
        destination = 0
        while begin < end:
            chunk, start = divmod(begin, n)
            length = min(end - begin, n - start)
            mask, payload = dataset.id.read_direct_chunk((chunk * n,))
            _decode_chunk(
                mask,
                payload,
                n,
                decoder=decoder,
                output=output[destination : destination + length],
                selection=slice(start, start + length),
            )
            begin += length
            destination += length
        if index < 2 and output.size > 1:
            np.cumsum(output, dtype=np.uint32, out=output)
        outputs.append(output)
    return outputs[0], outputs[1], outputs[2]
