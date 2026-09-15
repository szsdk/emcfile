"""Standard-compatible prefiltered shuffle+Zstd HDF5 writer."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading

import h5py
import numpy as np
import zstandard
from numba import njit


@njit(nogil=True)
def _shuffle_u32(src: np.ndarray, dst: np.ndarray, size: int) -> None:
    for index in range(size):
        value = src[index]
        dst[index] = value & np.uint32(0xFF)
        dst[size + index] = (value >> np.uint32(8)) & np.uint32(0xFF)
        dst[2 * size + index] = (value >> np.uint32(16)) & np.uint32(0xFF)
        dst[3 * size + index] = (value >> np.uint32(24)) & np.uint32(0xFF)


class PrefilteredDatasetWriter:
    """Bounded ordered writer for fragments of one 32-bit HDF5 dataset."""

    def __init__(self, dataset: h5py.Dataset, level: int, workers: int) -> None:
        if dataset.ndim != 1 or dataset.chunks is None or dataset.dtype.itemsize != 4:
            raise ValueError("direct Zstd writing requires a chunked 1-D 32-bit dataset")
        self.dataset, self.level, self.chunk_size = dataset, level, dataset.chunks[0]
        self.buffer: np.ndarray | None = None
        self.used = self.offset = 0
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending: list[tuple[int, Future[bytes]]] = []
        self.limit = max(2, workers * 2)
        self.state = threading.local()

    def _submit(self, values: np.ndarray) -> None:
        start = self.offset
        def compress() -> bytes:
            compressor = getattr(self.state, "compressor", None)
            if compressor is None:
                compressor = zstandard.ZstdCompressor(level=self.level)
                self.state.compressor = compressor
            shuffled = np.empty(self.chunk_size * 4, dtype="u1")
            _shuffle_u32(values.view("u4"), shuffled, self.chunk_size)
            return compressor.compress(memoryview(shuffled))
        self.pending.append((start, self.pool.submit(compress)))
        self.offset += self.chunk_size
        self.used = 0
        if len(self.pending) >= self.limit:
            self._commit_one()

    def _commit_one(self) -> None:
        offset, future = self.pending.pop(0)
        self.dataset.id.write_direct_chunk((offset,), future.result(), filter_mask=0)

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
        while self.pending:
            self._commit_one()
        self.pool.shutdown()
