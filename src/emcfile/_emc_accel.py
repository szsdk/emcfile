"""Metadata, mmap, and Rust acceleration helpers for binary .emc files and EMCPatternArray."""

from __future__ import annotations

import importlib.util
import mmap
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._h5_workers import env_workers

_ORIG_FIND_SPEC = importlib.util.find_spec
_EMC_RS_CACHE: Any = False

_CACHE_LOCK = threading.Lock()
_MAX_CACHE_ENTRIES = 16
_I32_MAX = int(np.iinfo(np.int32).max)


def get_emc_rs() -> Any:
    """Return the compiled ``emcfile_fast.emc`` extension module if enabled and available."""
    global _EMC_RS_CACHE
    if sys.byteorder != "little":
        return None
    if os.environ.get("EMCFILE_DISABLE_EMC_ACCEL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None
    if importlib.util.find_spec is not _ORIG_FIND_SPEC:
        if importlib.util.find_spec("emcfile_fast") is None:
            return None
    if _EMC_RS_CACHE is not False:
        return _EMC_RS_CACHE
    try:
        from emcfile_fast import emc as mod  # type: ignore[import-not-found]

        _EMC_RS_CACHE = mod
    except ImportError:
        _EMC_RS_CACHE = None
    return _EMC_RS_CACHE


_WORKERS_CACHE_KEY: tuple[str | None, int] | None = None
_WORKERS_CACHE_VAL: int = 4


def emc_workers(default: int = 4) -> int:
    """Return worker thread count for EMC Rust operations (clamped >= 1)."""
    global _WORKERS_CACHE_KEY, _WORKERS_CACHE_VAL
    raw_env = os.environ.get("EMCFILE_EMC_WORKERS")
    key = (raw_env, default)
    if _WORKERS_CACHE_KEY == key:
        return _WORKERS_CACHE_VAL
    w = max(1, env_workers("EMCFILE_EMC_WORKERS", default))
    _WORKERS_CACHE_KEY = key
    _WORKERS_CACHE_VAL = w
    return w


def as_c_u32_positions(arr: np.ndarray, num_pix: int) -> npt.NDArray[np.uint32] | None:
    """Return a zero-copy C-contiguous ``uint32`` view of position array if compatible."""
    if arr.ndim != 1 or not arr.flags.c_contiguous:
        return None
    if arr.dtype == np.uint32:
        return cast(npt.NDArray[np.uint32], arr)
    if arr.dtype == np.int32 and num_pix <= _I32_MAX:
        return arr.view(np.uint32)
    return None


def as_c_u32_counts(arr: np.ndarray) -> npt.NDArray[np.uint32] | None:
    """Return a zero-copy C-contiguous ``uint32`` view of count array if compatible."""
    if arr.ndim != 1 or not arr.flags.c_contiguous:
        return None
    if arr.dtype == np.uint32:
        return cast(npt.NDArray[np.uint32], arr)
    return None


def as_c_i32_counts(arr: np.ndarray) -> npt.NDArray[np.int32] | None:
    """Return a zero-copy C-contiguous ``int32`` view of multi-count array if compatible."""
    if arr.ndim != 1 or not arr.flags.c_contiguous:
        return None
    if arr.dtype == np.int32:
        return cast(npt.NDArray[np.int32], arr)
    return None


def as_c_u64_offsets(arr: np.ndarray) -> npt.NDArray[np.uint64] | None:
    """Return a zero-copy C-contiguous ``uint64`` view of offset array if compatible."""
    if arr.ndim != 1 or not arr.flags.c_contiguous:
        return None
    if arr.dtype == np.uint64:
        return cast(npt.NDArray[np.uint64], arr)
    return None


class EmcAccelEntry:
    """Cached header metadata, offsets, and read-only mmap for a binary ``.emc`` file."""

    __slots__ = (
        "key",
        "fn_path",
        "num_data",
        "num_pix",
        "ones",
        "multi",
        "ones_idx",
        "multi_idx",
        "mm",
        "pinned",
        "lock",
    )

    def __init__(
        self,
        key: tuple[str, int, int],
        fn_path: Path,
        *,
        pinned: bool = False,
    ) -> None:
        self.key = key
        self.fn_path = fn_path
        self.num_data: int | None = None
        self.num_pix: int | None = None
        self.ones: npt.NDArray[np.uint32] | None = None
        self.multi: npt.NDArray[np.uint32] | None = None
        self.ones_idx: npt.NDArray[np.uint64] | None = None
        self.multi_idx: npt.NDArray[np.uint64] | None = None
        self.mm: mmap.mmap | None = None
        self.pinned = pinned
        self.lock = threading.Lock()

    def ensure_mmap(self, *, allow_open: bool = False) -> mmap.mmap | None:
        if self.mm is not None:
            return self.mm
        if self.pinned and not allow_open:
            return None
        with self.lock:
            if self.mm is not None:
                return self.mm
            try:
                with self.fn_path.open("rb") as fp:
                    self.mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                self.mm = None
            return self.mm

    @contextmanager
    def borrow_mmap(self) -> Iterator[mmap.mmap | None]:
        """Yield a read-only ``mmap``; avoids holding open handles on Windows when unpinned."""
        if self.pinned or sys.platform != "win32":
            yield self.ensure_mmap()
            return
        mm_local: mmap.mmap | None = None
        try:
            with self.fn_path.open("rb") as fp:
                mm_local = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):
            mm_local = None
        try:
            yield mm_local
        finally:
            if mm_local is not None:
                try:
                    mm_local.close()
                except Exception:
                    pass

    def close(self) -> None:
        with self.lock:
            if self.mm is not None:
                try:
                    self.mm.close()
                except Exception:
                    pass
                self.mm = None


_EMC_META_CACHE: OrderedDict[tuple[str, int, int], EmcAccelEntry] = OrderedDict()


def _reset_emc_fork_state() -> None:
    global _CACHE_LOCK, _EMC_META_CACHE
    old_entries = list(_EMC_META_CACHE.values())
    _CACHE_LOCK = threading.Lock()
    _EMC_META_CACHE = OrderedDict()
    for entry in old_entries:
        entry.lock = threading.Lock()
        entry.mm = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_emc_fork_state)


def get_emc_cache_entry(
    fn: str | Path, *, _resolved: bool = False
) -> EmcAccelEntry | None:
    """Return a cached :class:`EmcAccelEntry` validated by file ``(mtime_ns, size)``."""
    try:
        path = fn if (_resolved and isinstance(fn, Path)) else Path(fn).resolve()
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _CACHE_LOCK:
        entry = _EMC_META_CACHE.get(key)
        if entry is not None:
            _EMC_META_CACHE.move_to_end(key)
            return entry
        stale_keys = [k for k in _EMC_META_CACHE if k[0] == str(path)]
        for k in stale_keys:
            old = _EMC_META_CACHE.pop(k, None)
            if old is not None:
                old.close()
        entry = EmcAccelEntry(key, path)
        _EMC_META_CACHE[key] = entry
        while len(_EMC_META_CACHE) > _MAX_CACHE_ENTRIES:
            _, evicted = _EMC_META_CACHE.popitem(last=False)
            evicted.close()
        return entry


def create_private_emc_entry(fn: str | Path) -> EmcAccelEntry:
    """Create a private :class:`EmcAccelEntry` pinned to an open reader."""
    path = Path(fn).resolve()
    entry = EmcAccelEntry((str(path), 0, 0), path, pinned=True)
    entry.ensure_mmap(allow_open=True)
    return entry


def invalidate_emc_cache_for_path(fn: str | Path) -> None:
    """Invalidate any cached entries for *fn* before/after writing."""
    try:
        path_str = str(Path(fn).resolve())
    except OSError:
        path_str = str(fn)
    with _CACHE_LOCK:
        keys = [k for k in _EMC_META_CACHE if k[0] == path_str]
        for k in keys:
            old = _EMC_META_CACHE.pop(k, None)
            if old is not None:
                old.close()
