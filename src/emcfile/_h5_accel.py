"""Low-overhead metadata, mmap, and Rust acceleration helpers for HDF5 v2."""

from __future__ import annotations

import itertools
import mmap
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

_CACHE_LOCK = threading.Lock()
_MAX_CACHE_ENTRIES = 16
_TOKEN_COUNTER = itertools.count(1)


def get_h5_rs() -> Any:
    """Return the compiled ``emcfile_fast.h5`` extension module if available."""
    if sys.byteorder != "little":
        return None
    try:
        from emcfile_fast import h5 as mod  # type: ignore[import-not-found]

        return mod
    except ImportError:
        return None


class H5AccelEntry:
    """Cached metadata, offsets, chunk index tables, and read-only mmap for an HDF5 group."""

    __slots__ = (
        "key",
        "file_token",
        "fn_path",
        "group_name",
        "version",
        "num_data",
        "num_pix",
        "position_encoding",
        "ones",
        "multi",
        "ones_idx",
        "multi_idx",
        "eligible",
        "validated_offsets",
        "mm",
        "tables",
        "chunk_lens",
        "dataset_sizes",
        "lock",
    )

    def __init__(
        self,
        key: tuple[str, str, int, int],
        fn_path: Path,
        group_name: str,
    ) -> None:
        self.key = key
        self.file_token = next(_TOKEN_COUNTER)
        self.fn_path = fn_path
        self.group_name = group_name
        self.version: str | None = None
        self.num_data: int | None = None
        self.num_pix: int | None = None
        self.position_encoding: str | None = None
        self.ones: npt.NDArray[np.uint32] | None = None
        self.multi: npt.NDArray[np.uint32] | None = None
        self.ones_idx: npt.NDArray[np.uint64] | None = None
        self.multi_idx: npt.NDArray[np.uint64] | None = None
        self.eligible: bool | None = None
        self.validated_offsets: tuple[int, int] | None = None
        self.mm: mmap.mmap | None = None
        self.tables: (
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]
            | None
        ) = None
        self.chunk_lens: tuple[int, int, int] | None = None
        self.dataset_sizes: tuple[int, int, int] | None = None
        self.lock = threading.Lock()

    def ensure_mmap(self) -> mmap.mmap | None:
        if self.mm is not None:
            return self.mm
        with self.lock:
            if self.mm is not None:
                return self.mm
            try:
                fd = os.open(self.fn_path, os.O_RDONLY)
                try:
                    self.mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                finally:
                    os.close(fd)
            except OSError:
                self.mm = None
            return self.mm

    def close(self) -> None:
        with self.lock:
            if self.mm is not None:
                try:
                    self.mm.close()
                except Exception:
                    pass
                self.mm = None


_H5_META_CACHE: OrderedDict[tuple[str, str, int, int], H5AccelEntry] = OrderedDict()


def _normalize_group_name(gn: str) -> str:
    if not gn or gn == "/":
        return "/"
    return "/" + gn.strip("/")


def get_cache_entry(fn: str | Path, gn: str = "/") -> H5AccelEntry | None:
    """Return a cached :class:`H5AccelEntry` validated by file ``(mtime_ns, size)``."""
    try:
        path = Path(fn).resolve()
        st = path.stat()
    except OSError:
        return None
    norm_gn = _normalize_group_name(gn)
    key = (str(path), norm_gn, st.st_mtime_ns, st.st_size)
    with _CACHE_LOCK:
        entry = _H5_META_CACHE.get(key)
        if entry is not None:
            _H5_META_CACHE.move_to_end(key)
            return entry
        stale_keys = [
            k for k in _H5_META_CACHE if k[0] == str(path) and k[1] == norm_gn
        ]
        for k in stale_keys:
            old = _H5_META_CACHE.pop(k, None)
            if old is not None:
                old.close()
        entry = H5AccelEntry(key, path, norm_gn)
        _H5_META_CACHE[key] = entry
        while len(_H5_META_CACHE) > _MAX_CACHE_ENTRIES:
            _, evicted = _H5_META_CACHE.popitem(last=False)
            evicted.close()
        return entry


def invalidate_cache_for_path(fn: str | Path) -> None:
    """Invalidate any cached entries for *fn* before/after writing."""
    try:
        path_str = str(Path(fn).resolve())
    except OSError:
        path_str = str(fn)
    with _CACHE_LOCK:
        keys = [k for k in _H5_META_CACHE if k[0] == path_str]
        for k in keys:
            old = _H5_META_CACHE.pop(k, None)
            if old is not None:
                old.close()


def get_entry_for_group(group: Any) -> H5AccelEntry | None:
    """Return the :class:`H5AccelEntry` for an ``h5py.Group``, ``h5py.File``, or :class:`LazyH5Group`."""
    if isinstance(group, LazyH5Group):
        return group.entry
    try:
        h5file = group.file
        filename = h5file.filename
        if not filename:
            return None
        return get_cache_entry(filename, group.name)
    except Exception:
        return None


def build_chunk_tables(
    entry: H5AccelEntry,
    datasets: list[h5py.Dataset],
) -> bool:
    """Populate ``entry.tables``, ``entry.chunk_lens``, ``entry.dataset_sizes``, and ``entry.mm``."""
    if entry.tables is not None and entry.mm is not None:
        return True
    mm = entry.ensure_mmap()
    if mm is None:
        return False
    with entry.lock:
        if entry.tables is not None:
            return True
        tables: list[npt.NDArray[np.int64]] = []
        chunk_lens: list[int] = []
        sizes: list[int] = []
        for ds in datasets:
            if not hasattr(ds.id, "chunk_iter") or ds.chunks is None:
                return False
            n = int(ds.chunks[0])
            sz = int(ds.size)
            n_chunks = (sz + n - 1) // n
            tbl = np.full((n_chunks, 3), -1, dtype=np.int64)
            if n_chunks > 0:

                def _cb(
                    info: Any, _tbl: npt.NDArray[np.int64] = tbl, _n: int = n
                ) -> None:
                    idx = info.chunk_offset[0] // _n
                    if 0 <= idx < _tbl.shape[0]:
                        _tbl[idx, 0] = info.byte_offset
                        _tbl[idx, 1] = info.size
                        _tbl[idx, 2] = info.filter_mask

                ds.id.chunk_iter(_cb)
                if np.any(tbl[:, 1] < 0):
                    return False
            tables.append(tbl)
            chunk_lens.append(n)
            sizes.append(sz)
        entry.tables = (tables[0], tables[1], tables[2])
        entry.chunk_lens = (chunk_lens[0], chunk_lens[1], chunk_lens[2])
        entry.dataset_sizes = (sizes[0], sizes[1], sizes[2])
        return True


class LazyH5Group:
    """Deferred HDF5 group wrapper that avoids opening ``h5py.File`` when cached mmap tables suffice."""

    __slots__ = ("_h5path", "entry", "_reader", "_own_fp", "_gp")

    def __init__(
        self,
        h5path_obj: Any,
        entry: H5AccelEntry | None,
        *,
        reader: Any = None,
    ) -> None:
        self._h5path = h5path_obj
        self.entry = entry
        self._reader = reader
        self._own_fp: h5py.File | None = None
        self._gp: h5py.Group | h5py.File | None = None

    def real_group(self) -> h5py.Group | h5py.File:
        if self._gp is not None:
            return self._gp
        if self._reader is not None:
            fp = self._reader._ensure_file_handle()
            self._gp = fp[self._h5path.gn]
            return self._gp
        self._own_fp = h5py.File(self._h5path.fn, "r")
        self._gp = self._own_fp[self._h5path.gn]
        return self._gp

    @property
    def attrs(self) -> Any:
        return self.real_group().attrs

    @property
    def file(self) -> Any:
        return self.real_group().file

    @property
    def name(self) -> str:
        if self.entry is not None:
            return self.entry.group_name
        return self.real_group().name

    def __contains__(self, key: str) -> bool:
        return key in self.real_group()

    def __getitem__(self, key: str) -> Any:
        return self.real_group()[key]

    def __enter__(self) -> LazyH5Group:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._own_fp is not None:
            try:
                self._own_fp.close()
            finally:
                self._own_fp = None
                self._gp = None
