"""Standard-compatible prefiltered shuffle+Zstd HDF5 writer."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import importlib.util
import sys
import threading
from typing import Any

import h5py
import numpy as np

_shuffle_kernel: Any | None = None
_shuffle_lock = threading.Lock()


def available() -> bool:
    """Whether all optional direct-write acceleration is installed."""
    return sys.byteorder == "little" and all(importlib.util.find_spec(name) is not None for name in ("numba", "zstandard"))


def _compress(values: np.ndarray, chunk_size: int, level: int, state: threading.local) -> bytes:
    """Shuffle a full uint32 chunk and compress it in a pool worker."""
    import zstandard

    compressor = getattr(state, "compressor", None)
    if compressor is None:
        compressor = zstandard.ZstdCompressor(level=level)
        state.compressor = compressor
    global _shuffle_kernel
    if _shuffle_kernel is None:
        with _shuffle_lock:
            if _shuffle_kernel is None:
                from numba import njit

                @njit(nogil=True)
                def shuffle(source: Any, output: Any, size: int) -> None:
                    for index in range(size):
                        value = source[index]
                        output[index] = value & np.uint32(0xFF)
                        output[size + index] = (value >> np.uint32(8)) & np.uint32(0xFF)
                        output[2 * size + index] = (value >> np.uint32(16)) & np.uint32(0xFF)
                        output[3 * size + index] = (value >> np.uint32(24)) & np.uint32(0xFF)

                _shuffle_kernel = shuffle
    shuffled = np.empty(chunk_size * 4, dtype="u1")
    _shuffle_kernel(values, shuffled, chunk_size)
    return compressor.compress(memoryview(shuffled))


class PrefilteredDatasetWriter:
    """Bounded ordered writer for one 32-bit dataset.

    A supplied executor and semaphore belong to the enclosing write operation.
    The semaphore limits *all* datasets together to ``2 * workers`` queued
    chunks, preventing five per-dataset queues from multiplying memory.
    """

    def __init__(self, dataset: h5py.Dataset, level: int, workers: int, *, pool: ThreadPoolExecutor | None = None, slots: threading.Semaphore | None = None) -> None:
        if dataset.ndim != 1 or dataset.chunks is None or dataset.dtype.itemsize != 4:
            raise ValueError("direct Zstd writing requires a chunked 1-D 32-bit dataset")
        self.dataset, self.level, self.chunk_size = dataset, level, dataset.chunks[0]
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
            future = self.pool.submit(_compress, values, self.chunk_size, self.level, self.state)
        except BaseException:
            self.slots.release()
            raise
        self.pending.append((start, future))
        self.offset += self.chunk_size
        self.used = 0

    def _commit_one(self) -> None:
        offset, future = self.pending.pop(0)
        try:
            self.dataset.id.write_direct_chunk((offset,), future.result(), filter_mask=0)
        finally:
            self.slots.release()

    def write(self, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values).view("u4")
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
            self.buffer[self.used : self.used + copied] = values[cursor : cursor + copied]
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
