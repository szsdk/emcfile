"""Standard-compatible prefiltered shuffle+Zstd HDF5 writer."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import h5py
import numpy as np

from ._h5_accel import get_h5_rs

_EMPTY_U64 = np.empty(0, dtype=np.uint64)


def available() -> bool:
    """Whether all optional direct-write acceleration is installed."""
    return sys.byteorder == "little" and get_h5_rs() is not None


def _as_dataset_u32(values: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Coerce an integer array safely to a 4-byte dataset dtype and view as uint32."""
    arr = np.asarray(values)
    le_u4 = np.dtype("<u4")
    le_i4 = np.dtype("<i4")
    target_le = np.dtype("<u4" if target_dtype.kind == "u" else "<i4")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"direct Zstd dataset writer expects integer values, got {arr.dtype}"
        )
    if arr.dtype == target_le:
        return np.ascontiguousarray(arr).view(le_u4)
    if arr.size > 0:
        info = np.iinfo(target_le)
        if np.issubdtype(arr.dtype, np.signedinteger) and target_le == le_u4:
            if int(arr.min()) < 0 or (
                arr.dtype.itemsize > 4 and int(arr.max()) > info.max
            ):
                raise ValueError("position values must fit in uint32")
        elif np.issubdtype(arr.dtype, np.unsignedinteger) and target_le == le_i4:
            if int(arr.max()) > info.max:
                raise ValueError(f"values exceed {target_le} maximum")
        elif arr.dtype.itemsize > 4:
            if int(arr.min()) < info.min or int(arr.max()) > info.max:
                raise ValueError(
                    f"values exceed {target_le} range [{info.min}, {info.max}]"
                )
    if arr.dtype in (le_u4, le_i4):
        return np.ascontiguousarray(arr).view(le_u4)
    return np.ascontiguousarray(arr, dtype=target_le).view(le_u4)


def _compress(
    values: np.ndarray,
    chunk_size: int,
    level: int,
    state: threading.local | None = None,
) -> bytes:
    """Shuffle a full uint32 chunk and compress it in a pool worker."""
    rs = get_h5_rs()
    if rs is None:
        raise RuntimeError("Direct Zstd writing requires emcfile[fast]")
    arr = np.ascontiguousarray(values, dtype=np.uint32)
    return rs.compress_chunk(arr, chunk_size, level)


class PrefilteredDatasetWriter:
    """Bounded ordered writer for one 32-bit dataset.

    A supplied executor and semaphore belong to the enclosing write operation.
    The semaphore limits *all* datasets together to ``2 * workers`` queued
    chunks, preventing five per-dataset queues from multiplying memory.
    """

    def __init__(
        self,
        dataset: h5py.Dataset,
        level: int,
        workers: int,
        *,
        pool: ThreadPoolExecutor | None = None,
        slots: threading.Semaphore | None = None,
    ) -> None:
        if dataset.ndim != 1 or dataset.chunks is None or dataset.dtype.itemsize != 4:
            raise ValueError(
                "direct Zstd writing requires a chunked 1-D 32-bit dataset"
            )
        self.dataset, self.level, self.chunk_size = dataset, level, dataset.chunks[0]
        self.workers = max(1, int(workers))
        self.buffer: np.ndarray | None = None
        self.used = self.offset = 0
        self.pool = pool or ThreadPoolExecutor(max_workers=workers)
        self._owns_pool = pool is None
        self.slots = slots or threading.Semaphore(max(2, workers * 2))
        self.pending: list[tuple[int, Future[bytes]]] = []
        self.state = threading.local()

    def _submit(self, values: np.ndarray) -> None:
        if not self.slots.acquire(blocking=False):
            # During one array write this writer owns at least one queued chunk.
            # The enclosing writer drains after each dataset array, so this never
            # waits behind a different dataset's ordered commit queue.
            self._commit_one()
            self.slots.acquire()
        start = self.offset
        try:
            future = self.pool.submit(
                _compress, values, self.chunk_size, self.level, self.state
            )
        except BaseException:
            self.slots.release()
            raise
        self.pending.append((start, future))
        self.offset += self.chunk_size
        self.used = 0

    def _commit_one(self) -> None:
        offset, future = self.pending.pop(0)
        try:
            self.dataset.id.write_direct_chunk(
                (offset,), future.result(), filter_mask=0
            )
        finally:
            self.slots.release()

    def _write_chunks_rs(
        self,
        values_u32: np.ndarray,
        offsets_u64: np.ndarray,
        n_chunks: int,
        *,
        is_delta: bool,
        check_sorted: bool,
    ) -> None:
        rs = get_h5_rs()
        assert rs is not None
        self.drain()
        batch_chunks = min(
            n_chunks,
            max(
                self.workers * 4,
                min(128, max(1, (32 << 20) // (self.chunk_size * 4))),
            ),
        )
        base_offset = self.offset
        ds_id = self.dataset.id
        ranges = [
            (c_first, min(c_first + batch_chunks, n_chunks))
            for c_first in range(0, n_chunks, batch_chunks)
        ]
        if not ranges:
            return
        c_bound = int(rs.compress_bound_u32(self.chunk_size))
        n_bufs = 2 if len(ranges) > 1 else 1
        bufs = [np.empty(batch_chunks * c_bound, dtype=np.uint8) for _ in range(n_bufs)]
        metas = [np.empty((batch_chunks, 3), dtype=np.int64) for _ in range(n_bufs)]
        mvs = [memoryview(b) for b in bufs]

        next_future = self.pool.submit(
            rs.encode_chunks_into,
            values_u32,
            offsets_u64,
            ranges[0][0],
            ranges[0][1],
            self.chunk_size,
            self.level,
            is_delta,
            check_sorted,
            self.workers,
            bufs[0],
            metas[0],
        )
        for idx, (c_first, c_last) in enumerate(ranges):
            cur_slot = idx & 1
            future = next_future
            n_done = future.result()
            if idx + 1 < len(ranges):
                nxt_slot = (idx + 1) & 1
                nf_first, nf_last = ranges[idx + 1]
                next_future = self.pool.submit(
                    rs.encode_chunks_into,
                    values_u32,
                    offsets_u64,
                    nf_first,
                    nf_last,
                    self.chunk_size,
                    self.level,
                    is_delta,
                    check_sorted,
                    self.workers,
                    bufs[nxt_slot],
                    metas[nxt_slot],
                )
                time.sleep(0)
            mv = mvs[cur_slot]
            meta = metas[cur_slot]
            for i in range(n_done):
                c_off = base_offset + int(meta[i, 0])
                b_off = int(meta[i, 1])
                b_len = int(meta[i, 2])
                ds_id.write_direct_chunk(
                    (c_off,), mv[b_off : b_off + b_len], filter_mask=0
                )
        self.offset = base_offset + n_chunks * self.chunk_size

    def write_delta(
        self,
        values: np.ndarray,
        offsets: np.ndarray,
        counts: np.ndarray,
        *,
        check_sorted: bool = False,
    ) -> None:
        """Fused pattern-local delta + byte-shuffle + Zstd direct chunk write."""
        values_u32 = _as_dataset_u32(values, self.dataset.dtype)
        rs = get_h5_rs()
        is_complete_tail = self.offset + values_u32.size == self.dataset.size
        if (
            rs is not None
            and self.buffer is None
            and self.used == 0
            and (values_u32.size % self.chunk_size == 0 or is_complete_tail)
        ):
            offsets_u64 = np.ascontiguousarray(offsets, dtype=np.uint64)
            n_chunks = (values_u32.size + self.chunk_size - 1) // self.chunk_size
            if n_chunks > 0:
                self._write_chunks_rs(
                    values_u32,
                    offsets_u64,
                    n_chunks,
                    is_delta=True,
                    check_sorted=check_sorted,
                )
            return
        from ._delta import encode_pattern_local_delta_parallel

        encoded = encode_pattern_local_delta_parallel(
            values_u32,
            counts,
            self.workers,
            check_sorted=check_sorted,
            accelerated=True,
        )
        self.write(encoded)

    def write(self, values: np.ndarray) -> None:
        values = _as_dataset_u32(values, self.dataset.dtype)
        rs = get_h5_rs()
        if rs is not None and self.buffer is None and self.used == 0:
            is_complete_tail = self.offset + values.size == self.dataset.size
            if is_complete_tail and values.size > 0:
                n_chunks = (values.size + self.chunk_size - 1) // self.chunk_size
                self._write_chunks_rs(
                    values,
                    _EMPTY_U64,
                    n_chunks,
                    is_delta=False,
                    check_sorted=False,
                )
                return
            n_full = values.size // self.chunk_size
            if n_full > 0:
                self._write_chunks_rs(
                    values[: n_full * self.chunk_size],
                    _EMPTY_U64,
                    n_full,
                    is_delta=False,
                    check_sorted=False,
                )
                values = values[n_full * self.chunk_size :]
                if values.size == 0:
                    return

        cursor = 0
        while cursor < values.size:
            if self.buffer is None:
                full_stop = values.size - (values.size - cursor) % self.chunk_size
                while cursor < full_stop:
                    self._submit(values[cursor : cursor + self.chunk_size])
                    cursor += self.chunk_size
                if cursor == values.size:
                    break
                self.buffer = np.zeros(self.chunk_size, dtype="u4")
            copied = min(self.chunk_size - self.used, values.size - cursor)
            self.buffer[self.used : self.used + copied] = values[
                cursor : cursor + copied
            ]
            self.used += copied
            cursor += copied
            if self.used == self.chunk_size:
                self._submit(self.buffer)
                self.buffer = None

    def close(self) -> None:
        if self.used:
            assert self.buffer is not None
            self._submit(self.buffer)
            self.buffer = None
        try:
            while self.pending:
                self._commit_one()
        finally:
            if self._owns_pool:
                self.pool.shutdown(cancel_futures=True)

    def drain(self) -> None:
        """Commit queued chunks while retaining the final partial buffer."""
        while self.pending:
            self._commit_one()
