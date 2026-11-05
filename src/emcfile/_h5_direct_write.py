"""Standard-compatible prefiltered shuffle+Zstd HDF5 writer."""

from __future__ import annotations

import importlib.util
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import h5py
import numpy as np

from ._delta import _check, _values

_shuffle_kernel: Any | None = None
_delta_shuffle_kernel: Any | None = None
_delta_slice_kernel: Any | None = None
_shuffle_lock = threading.Lock()


def available() -> bool:
    """Whether all optional direct-write acceleration is installed."""
    return sys.byteorder == "little" and all(
        importlib.util.find_spec(name) is not None for name in ("numba", "zstandard")
    )


def _ensure_kernels() -> tuple[Any, Any, Any]:
    global _shuffle_kernel, _delta_shuffle_kernel, _delta_slice_kernel
    if _shuffle_kernel is None:
        with _shuffle_lock:
            if _shuffle_kernel is None:
                from numba import njit

                @njit(nogil=True)
                def shuffle(source: Any, output: Any, size: int) -> None:
                    valid = source.size
                    b0 = output[0:valid]
                    b1 = output[size : size + valid]
                    b2 = output[2 * size : 2 * size + valid]
                    b3 = output[3 * size : 3 * size + valid]
                    for index in range(valid):
                        value = source[index]
                        b0[index] = np.uint8(value)
                        b1[index] = np.uint8(value >> np.uint32(8))
                        b2[index] = np.uint8(value >> np.uint32(16))
                        b3[index] = np.uint8(value >> np.uint32(24))
                    if valid < size:
                        output[valid:size] = 0
                        output[size + valid : 2 * size] = 0
                        output[2 * size + valid : 3 * size] = 0
                        output[3 * size + valid : 4 * size] = 0

                @njit(nogil=True)
                def delta_shuffle(
                    src_all: Any,
                    c_start: int,
                    c_stop: int,
                    offsets: Any,
                    p_idx: int,
                    dst_u8: Any,
                    chunk_n: int,
                ) -> int:
                    valid_n = c_stop - c_start
                    b0 = dst_u8[0:valid_n]
                    b1 = dst_u8[chunk_n : chunk_n + valid_n]
                    b2 = dst_u8[2 * chunk_n : 2 * chunk_n + valid_n]
                    b3 = dst_u8[3 * chunk_n : 3 * chunk_n + valid_n]
                    if valid_n > 0:
                        if c_start > 0:
                            s_cur = src_all[c_start:c_stop]
                            s_prv = src_all[c_start - 1 : c_stop - 1]
                            for i in range(valid_n):
                                v = s_cur[i] - s_prv[i]
                                b0[i] = np.uint8(v)
                                b1[i] = np.uint8(v >> np.uint32(8))
                                b2[i] = np.uint8(v >> np.uint32(16))
                                b3[i] = np.uint8(v >> np.uint32(24))
                        else:
                            v0 = src_all[0]
                            b0[0] = np.uint8(v0)
                            b1[0] = np.uint8(v0 >> np.uint32(8))
                            b2[0] = np.uint8(v0 >> np.uint32(16))
                            b3[0] = np.uint8(v0 >> np.uint32(24))
                            for i in range(1, valid_n):
                                v = src_all[i] - src_all[i - 1]
                                b0[i] = np.uint8(v)
                                b1[i] = np.uint8(v >> np.uint32(8))
                                b2[i] = np.uint8(v >> np.uint32(16))
                                b3[i] = np.uint8(v >> np.uint32(24))
                    if valid_n < chunk_n:
                        dst_u8[valid_n:chunk_n] = 0
                        dst_u8[chunk_n + valid_n : 2 * chunk_n] = 0
                        dst_u8[2 * chunk_n + valid_n : 3 * chunk_n] = 0
                        dst_u8[3 * chunk_n + valid_n : 4 * chunk_n] = 0
                    num_p = offsets.size - 1
                    while p_idx < num_p and np.int64(offsets[p_idx]) < c_start:
                        p_idx += 1
                    while p_idx < num_p and np.int64(offsets[p_idx]) < c_stop:
                        s = np.int64(offsets[p_idx])
                        if s < np.int64(offsets[p_idx + 1]):
                            idx = s - c_start
                            v = src_all[s]
                            b0[idx] = np.uint8(v)
                            b1[idx] = np.uint8(v >> np.uint32(8))
                            b2[idx] = np.uint8(v >> np.uint32(16))
                            b3[idx] = np.uint8(v >> np.uint32(24))
                        p_idx += 1
                    return p_idx

                @njit(nogil=True)
                def delta_slice(
                    src_all: Any,
                    c_start: int,
                    c_stop: int,
                    offsets: Any,
                    p_idx: int,
                    dst_u32: Any,
                ) -> int:
                    valid_n = c_stop - c_start
                    for i in range(valid_n):
                        g = c_start + i
                        prev = src_all[g - 1] if g > 0 else np.uint32(0)
                        dst_u32[i] = src_all[g] - prev
                    num_p = offsets.size - 1
                    while p_idx < num_p and np.int64(offsets[p_idx]) < c_start:
                        p_idx += 1
                    while p_idx < num_p and np.int64(offsets[p_idx]) < c_stop:
                        s = np.int64(offsets[p_idx])
                        if s < np.int64(offsets[p_idx + 1]):
                            dst_u32[s - c_start] = src_all[s]
                        p_idx += 1
                    return p_idx

                _shuffle_kernel = shuffle
                _delta_shuffle_kernel = delta_shuffle
                _delta_slice_kernel = delta_slice
    return _shuffle_kernel, _delta_shuffle_kernel, _delta_slice_kernel


def _get_worker_resources(
    chunk_size: int, level: int, state: threading.local
) -> tuple[Any, np.ndarray]:
    import zstandard

    compressor = getattr(state, "compressor", None)
    if compressor is None or getattr(state, "level", None) != level:
        compressor = zstandard.ZstdCompressor(level=level)
        state.compressor = compressor
        state.level = level
    buf = getattr(state, "buf", None)
    if buf is None or buf.size < chunk_size * 4:
        buf = np.empty(chunk_size * 4, dtype="u1")
        state.buf = buf
    return compressor, buf[: chunk_size * 4]


def _compress(
    values: np.ndarray, chunk_size: int, level: int, state: threading.local
) -> bytes:
    """Shuffle a full uint32 chunk and compress it in a pool worker."""
    compressor, shuffled = _get_worker_resources(chunk_size, level, state)
    shuffle_fn, _, _ = _ensure_kernels()
    shuffle_fn(values, shuffled, chunk_size)
    return compressor.compress(memoryview(shuffled))


def _compress_batch(
    values: np.ndarray,
    c_start: int,
    n_chunks: int,
    dataset_offset: int,
    chunk_size: int,
    level: int,
    state: threading.local,
) -> list[tuple[int, bytes]]:
    compressor, shuffled = _get_worker_resources(chunk_size, level, state)
    shuffle_fn, _, _ = _ensure_kernels()
    out: list[tuple[int, bytes]] = []
    for k in range(n_chunks):
        s = c_start + k * chunk_size
        shuffle_fn(values[s : s + chunk_size], shuffled, chunk_size)
        out.append(
            (dataset_offset + k * chunk_size, compressor.compress(memoryview(shuffled)))
        )
    return out


def _compress_delta_batch(
    values: np.ndarray,
    offsets: np.ndarray,
    c_start: int,
    n_chunks: int,
    dataset_offset: int,
    chunk_size: int,
    level: int,
    state: threading.local,
) -> list[tuple[int, bytes]]:
    compressor, shuffled = _get_worker_resources(chunk_size, level, state)
    _, delta_shuffle_fn, _ = _ensure_kernels()
    p_idx = int(np.searchsorted(offsets, c_start, side="left"))
    out: list[tuple[int, bytes]] = []
    for k in range(n_chunks):
        s = c_start + k * chunk_size
        p_idx = delta_shuffle_fn(
            values, s, s + chunk_size, offsets, p_idx, shuffled, chunk_size
        )
        out.append(
            (dataset_offset + k * chunk_size, compressor.compress(memoryview(shuffled)))
        )
    return out


class PrefilteredDatasetWriter:
    """Bounded ordered writer for one 32-bit dataset.

    A supplied executor and semaphore belong to the enclosing write operation.
    The semaphore limits *all* datasets together to ``2 * workers`` queued
    batches, preventing five per-dataset queues from multiplying memory.
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
        self.buffer: np.ndarray | None = None
        self.used = self.offset = 0
        self.pool = pool or ThreadPoolExecutor(max_workers=workers)
        self._owns_pool = pool is None
        self.slots = slots or threading.Semaphore(max(2, workers * 2))
        self.pending: list[tuple[int | None, Future[Any]]] = []
        self.state = threading.local()
        self.batch_chunks = max(1, min(16, (8 << 20) // max(4, self.chunk_size * 4)))

    def _acquire_slot(self) -> None:
        if not self.slots.acquire(blocking=False):
            self._commit_one()
            self.slots.acquire()

    def _submit(self, values: np.ndarray) -> None:
        self._acquire_slot()
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

    def _submit_batch(self, values: np.ndarray, c_start: int, n_chunks: int) -> None:
        self._acquire_slot()
        start = self.offset
        try:
            future = self.pool.submit(
                _compress_batch,
                values,
                c_start,
                n_chunks,
                start,
                self.chunk_size,
                self.level,
                self.state,
            )
        except BaseException:
            self.slots.release()
            raise
        self.pending.append((None, future))
        self.offset += n_chunks * self.chunk_size
        self.used = 0

    def _submit_delta_batch(
        self,
        values: np.ndarray,
        offsets: np.ndarray,
        c_start: int,
        n_chunks: int,
    ) -> None:
        self._acquire_slot()
        start = self.offset
        try:
            future = self.pool.submit(
                _compress_delta_batch,
                values,
                offsets,
                c_start,
                n_chunks,
                start,
                self.chunk_size,
                self.level,
                self.state,
            )
        except BaseException:
            self.slots.release()
            raise
        self.pending.append((None, future))
        self.offset += n_chunks * self.chunk_size
        self.used = 0

    def _commit_one(self) -> None:
        offset, future = self.pending.pop(0)
        try:
            res = future.result()
            if offset is not None:
                self.dataset.id.write_direct_chunk((offset,), res, filter_mask=0)
            else:
                for chunk_offset, payload in res:
                    self.dataset.id.write_direct_chunk(
                        (chunk_offset,), payload, filter_mask=0
                    )
        finally:
            self.slots.release()

    def write(self, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values).view("u4")
        cursor = 0
        while cursor < values.size:
            if self.buffer is None:
                rem = (values.size - cursor) % self.chunk_size
                full_stop = values.size - rem
                while cursor < full_stop:
                    n_chunks = min(
                        self.batch_chunks, (full_stop - cursor) // self.chunk_size
                    )
                    if n_chunks == 1:
                        self._submit(values[cursor : cursor + self.chunk_size])
                    else:
                        self._submit_batch(values, cursor, n_chunks)
                    cursor += n_chunks * self.chunk_size
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

    def write_delta(
        self,
        values: np.ndarray,
        offsets: np.ndarray,
        *,
        check_sorted: bool = False,
    ) -> None:
        values = np.ascontiguousarray(_values(values))
        offsets = np.ascontiguousarray(offsets, dtype=np.uint64)
        if int(offsets[-1]) != values.size:
            raise ValueError("pattern counts do not match the number of values")
        if check_sorted:
            _check(values, offsets)
        if values.size == 0:
            return
        _, _, delta_slice_fn = _ensure_kernels()
        cursor = 0
        p_idx = 0
        while cursor < values.size:
            if self.buffer is None:
                rem = (values.size - cursor) % self.chunk_size
                full_stop = values.size - rem
                while cursor < full_stop:
                    n_chunks = min(
                        self.batch_chunks, (full_stop - cursor) // self.chunk_size
                    )
                    self._submit_delta_batch(values, offsets, cursor, n_chunks)
                    cursor += n_chunks * self.chunk_size
                if cursor == values.size:
                    break
                self.buffer = np.zeros(self.chunk_size, dtype="u4")
                p_idx = int(np.searchsorted(offsets, cursor, side="left"))
            copied = min(self.chunk_size - self.used, values.size - cursor)
            p_idx = delta_slice_fn(
                values,
                cursor,
                cursor + copied,
                offsets,
                p_idx,
                self.buffer[self.used : self.used + copied],
            )
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
