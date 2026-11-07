"""Parquet 1-row-per-pattern reader and writer backed by Rust SIMD+Zstd."""

from __future__ import annotations

import importlib.util
import mmap
import os
import sys
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ._h5_workers import env_workers

if TYPE_CHECKING:
    from ._emc_patterns import PatternsSOne, PatternsSOneBase

PARQUET_SUFFIXES = (".parquet", ".pq")
_PARQUET_MAGIC = b"PAR1"
PARQUET_EXTRA_MESSAGE = (
    "Parquet support requires the optional 'fast' dependencies. "
    "Install with: pip install 'emcfile[fast]'"
)

_ORIG_FIND_SPEC = importlib.util.find_spec
_parquet_rs_cache: Any = False

_file_info_lock = threading.Lock()
_file_info_lru: dict[
    tuple[str, int, int, bytes, bytes],
    tuple[dict[str, Any], npt.NDArray[np.int64]],
] = {}


def _get_parquet_rs() -> Any:
    global _parquet_rs_cache
    if sys.byteorder != "little":
        return None
    if importlib.util.find_spec is not _ORIG_FIND_SPEC:
        if importlib.util.find_spec("emcfile_fast") is None:
            return None
    if _parquet_rs_cache is not False:
        return _parquet_rs_cache
    try:
        from emcfile_fast import parquet as mod  # type: ignore[import-not-found]

        _parquet_rs_cache = mod
    except ImportError:
        _parquet_rs_cache = None
    return _parquet_rs_cache


def _require_parquet_rs() -> Any:
    rs = _get_parquet_rs()
    if rs is None:
        raise RuntimeError(PARQUET_EXTRA_MESSAGE)
    return rs


def available() -> bool:
    """Whether native Rust Parquet support is available."""
    return _get_parquet_rs() is not None


def _ensure_decode_kernel() -> Any:
    return None


def is_parquet_file(path: Path) -> bool:
    """Return True if ``path`` has a Parquet suffix or ``PAR1`` magic bytes."""
    if path.suffix.lower() in PARQUET_SUFFIXES:
        return True
    try:
        with path.open("rb") as fp:
            if fp.read(4) != _PARQUET_MAGIC:
                return False
            fp.seek(-4, os.SEEK_END)
            return fp.read(4) == _PARQUET_MAGIC
    except OSError:
        return False


def _normalize_parquet_compression(
    compression: None | int | str,
    compression_opts: Any,
) -> tuple[str, int]:
    if compression is None or compression in ("none", "uncompressed", 0):
        return "NONE", 0
    if isinstance(compression, str):
        name = compression.strip().lower()
        if name in {"zstd", "zstandard"}:
            level = 3 if compression_opts is None else int(compression_opts)
            if level <= 0:
                level = 1
            return "ZSTD", level
    if compression == 32015:
        level = 3 if compression_opts is None else int(compression_opts)
        if level <= 0:
            level = 1
        return "ZSTD", level
    raise ValueError(f"Unsupported Parquet compression {compression!r}")


def _get_file_info(
    path: Path, cache: dict[str, Any] | None = None
) -> tuple[dict[str, Any], npt.NDArray[np.int64]]:
    if cache is not None and "file_info" in cache:
        return cache["file_info"]

    rs = _require_parquet_rs()
    resolved = path.resolve()
    st = resolved.stat()
    if st.st_size < 12:
        raise ValueError(f"Invalid or truncated Parquet file: {path}")
    with resolved.open("rb") as fp:
        head_sig = fp.read(min(64, st.st_size))
        fp.seek(-8, os.SEEK_END)
        tail8 = fp.read(8)
        if tail8[4:8] != _PARQUET_MAGIC:
            raise ValueError(f"Invalid Parquet magic bytes in {path}")
        fl = int.from_bytes(tail8[:4], "little")
        if fl <= 0 or fl > st.st_size - 12:
            raise ValueError(f"Invalid Parquet footer length in {path}")
        fp.seek(st.st_size - fl - 8)
        footer_bytes = fp.read(fl)
        tail_sig = footer_bytes[-min(32, fl) :] + tail8
        cache_key = (
            str(resolved),
            st.st_mtime_ns,
            st.st_size,
            head_sig,
            tail_sig,
        )
        with _file_info_lock:
            cached = _file_info_lru.get(cache_key)
            if cached is not None:
                if cache is not None:
                    cache["file_info"] = cached
                return cached

        parsed = rs.parse_parquet_footer(footer_bytes)
        if parsed is None:
            raise ValueError(f"Unsupported or corrupt Parquet v2 layout in {path}")
        with _file_info_lock:
            _file_info_lru[cache_key] = parsed
            if len(_file_info_lru) > 16:
                _file_info_lru.pop(next(iter(_file_info_lru)))
        if cache is not None:
            cache["file_info"] = parsed
        return parsed


def _get_or_open_mmap(
    path: Path, cache: dict[str, Any] | None
) -> tuple[mmap.mmap, Any, mmap.mmap | None]:
    if cache is not None:
        mm_cached = cache.get("mm")
        if mm_cached is not None:
            return mm_cached, None, None
        fd = os.open(path, os.O_RDONLY)
        try:
            mm_new = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        cache["mm"] = mm_new
        return mm_new, None, None
    fp_local = path.open("rb")
    mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
    return mm_local, fp_local, mm_local


def _iter_rg_batches(
    datas: Sequence[PatternsSOneBase], rg_pats: int, max_rgs_per_batch: int
) -> Iterator[PatternsSOne]:
    from ._emc_patterns import PatternsSOne

    batch_target_pats = max(1, rg_pats * max(1, max_rgs_per_batch))
    pending: list[PatternsSOne] = []
    pending_pats = 0

    for data in datas:
        cursor = 0
        n = data.num_data
        while cursor < n:
            need = batch_target_pats - pending_pats
            take = min(need, n - cursor)
            chunk = (
                data
                if (cursor == 0 and take == n and isinstance(data, PatternsSOne))
                else cast(PatternsSOne, data[cursor : cursor + take])
            )
            pending.append(chunk)
            pending_pats += take
            cursor += take
            if pending_pats == batch_target_pats:
                yield (
                    pending[0]
                    if len(pending) == 1
                    else cast(PatternsSOne, np.concatenate(pending))
                )
                pending = []
                pending_pats = 0

    if pending_pats > 0:
        yield (
            pending[0]
            if len(pending) == 1
            else cast(PatternsSOne, np.concatenate(pending))
        )


def write_parquet_patterns(
    datas: Sequence[PatternsSOneBase],
    path: Path,
    *,
    overwrite: bool = False,
    buffer_size: int = 1073741824,
    compression: None | int | str = None,
    compression_opts: Any = None,
    shuffle: bool = False,
    position_encoding: str = "absolute",
    check_sorted: bool = False,
) -> None:
    """Write pattern sources in order to a 1-row-per-pattern Parquet v2 file."""
    rs = _require_parquet_rs()
    if position_encoding not in {"absolute", "delta"}:
        raise ValueError("position_encoding must be 'absolute' or 'delta'")
    if not datas:
        raise ValueError("at least one pattern source is required")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists")

    num_pix = datas[0].num_pix
    if any(data.num_pix != num_pix for data in datas):
        raise ValueError("all pattern sources must have the same number of pixels")

    pq_codec, pq_level = _normalize_parquet_compression(compression, compression_opts)

    num_ones = int(sum(int(d.ones.sum(dtype=np.uint64)) for d in datas))
    num_multi = int(sum(int(d.multi.sum(dtype=np.uint64)) for d in datas))
    num_data = int(sum(data.num_data for data in datas))

    total_bytes = num_data * 8 + (num_ones + 2 * num_multi) * 4
    env_chunk_bytes = os.environ.get("EMCFILE_PARQUET_CHUNK_BYTES")
    if env_chunk_bytes is not None:
        chunk_bytes = int(env_chunk_bytes)
        if chunk_bytes <= 0 or chunk_bytes % np.dtype("u4").itemsize:
            raise ValueError(
                "EMCFILE_PARQUET_CHUNK_BYTES must be a positive multiple of 4"
            )
    else:
        chunk_bytes = 4 << 20

    bytes_per_pat = max(4, total_bytes // max(1, num_data))
    rg_pats = max(1, min(max(1, num_data), chunk_bytes // bytes_per_pat))

    env_page_bytes = os.environ.get("EMCFILE_PARQUET_PAGE_BYTES")
    if env_page_bytes is not None:
        page_bytes = max(4, int(env_page_bytes))
    else:
        page_bytes = 64 << 10
    page_pats = max(1, min(rg_pats, page_bytes // bytes_per_pat))

    workers = max(1, env_workers("EMCFILE_PARQUET_WRITE_WORKERS", 4, allow_zero=True))
    is_delta = position_encoding == "delta"

    meta_kv = {
        b"version": b"2",
        b"num_pix": str(num_pix).encode("ascii"),
        b"num_data": str(num_data).encode("ascii"),
        b"rg_pats": str(rg_pats).encode("ascii"),
        b"position_encoding": position_encoding.encode("ascii"),
        b"shuffle": b"1" if shuffle else b"0",
        b"payload_codec": b"none",
        b"compression": pq_codec.lower().encode("ascii"),
        b"num_ones": str(num_ones).encode("ascii"),
        b"num_multi": str(num_multi).encode("ascii"),
    }

    from ._emc_patterns import PatternsSOne

    rg_bytes = max(1, rg_pats * bytes_per_pat)
    max_rgs_per_batch = max(1, buffer_size // rg_bytes)
    if len(datas) == 1 and isinstance(datas[0], PatternsSOne):
        batches_iter: Iterator[PatternsSOne] = iter([datas[0]])
    else:
        batches_iter = _iter_rg_batches(datas, rg_pats, max_rgs_per_batch)

    with path.open("wb") as fp:
        fp.write(_PARQUET_MAGIC)
        fp.flush()
        batch_pats_list: list[int] = []
        meta_batches_list: list[npt.NDArray[np.int64]] = []

        for batch in batches_iter:
            b_pats = batch.num_data
            if b_pats == 0:
                continue
            b_rgs = (b_pats + rg_pats - 1) // rg_pats
            po_src = np.ascontiguousarray(batch.place_ones, dtype=np.uint32)
            pm_src = np.ascontiguousarray(batch.place_multi, dtype=np.uint32)
            cm_src = np.ascontiguousarray(batch.count_multi).view(np.uint32)
            ones_idx = np.ascontiguousarray(batch.ones_idx, dtype=np.uint64)
            multi_idx = np.ascontiguousarray(batch.multi_idx, dtype=np.uint64)

            meta_batch = rs.write_batch_zstd(
                fp.fileno(),
                po_src,
                pm_src,
                cm_src,
                ones_idx,
                multi_idx,
                b_pats,
                rg_pats,
                page_pats,
                pq_level,
                is_delta,
                shuffle,
                check_sorted,
                min(workers, b_rgs),
            )
            batch_pats_list.append(b_pats)
            meta_batches_list.append(meta_batch)

        rs.write_parquet_footer(
            fp.fileno(),
            num_data,
            rg_pats,
            batch_pats_list,
            meta_batches_list,
            list(meta_kv.items()),
        )


def read_parquet_metadata(
    path: Path, *, cache: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Read header metadata from a Parquet v2 pattern file."""
    meta_dict, _ = _get_file_info(path, cache=cache)
    return meta_dict


def read_parquet_ones_multi(
    path: Path,
    num_data: int,
    *,
    pf: Any | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Derive ``ones`` and ``multi`` count arrays from a Parquet v2 file."""
    del pf
    if num_data == 0:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

    rs = _require_parquet_rs()
    meta_dict, col_info = _get_file_info(path, cache=cache)
    workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    rg_pats = int(meta_dict["rg_pats"])
    n_workers = max(1, min(max(1, workers), max(1, len(col_info))))
    mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
    try:
        return rs.read_ones_multi(mm_obj, col_info, rg_pats, num_data, n_workers)
    finally:
        if mm_local is not None:
            mm_local.close()
        if fp_local is not None:
            fp_local.close()


def read_single_frame_parquet(
    pf: Any,
    frame: int,
    ones_idx: npt.NDArray[np.uint64] | None = None,
    multi_idx: npt.NDArray[np.uint64] | None = None,
    *,
    path: Path | None = None,
    col_info: npt.NDArray[np.int64] | None = None,
    num_data: int,
    rg_pats: int,
    is_delta: bool,
    is_shuffled: bool,
    payload_codec: str = "none",
    cache: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    """Fast single-frame read using Rust page-level Zstd + SIMD decoding."""
    del pf, ones_idx, multi_idx, payload_codec
    rs = _require_parquet_rs()
    assert path is not None
    if col_info is None:
        _, col_info = _get_file_info(path, cache=cache)
    rg = frame // rg_pats
    ps = rg * rg_pats
    pe = min(num_data, (rg + 1) * rg_pats)

    mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
    try:
        po, pm, cm = rs.decode_single_frame(
            mm_obj,
            col_info,
            rg,
            ps,
            pe,
            frame,
            is_delta,
            is_shuffled,
        )
        return po, pm, cm.view(np.int32)
    finally:
        if mm_local is not None:
            mm_local.close()
        if fp_local is not None:
            fp_local.close()


def read_parquet_patterns_with_counts(
    path: Path,
    index_ranges: npt.NDArray[np.integer[Any]],
    ones_idx: npt.NDArray[np.uint64] | None = None,
    multi_idx: npt.NDArray[np.uint64] | None = None,
    *,
    num_data: int,
    rg_pats: int,
    position_encoding: str,
    shuffle: bool,
    payload_codec: str = "none",
    pf: Any | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.int32],
]:
    """Read selected pattern ranges and their per-pattern counts from Parquet v2."""
    del payload_codec, pf
    rs = _require_parquet_rs()
    ranges = np.asarray(index_ranges)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("Pattern ranges must have shape (N, 2)")
    if (
        np.any(ranges < 0)
        or np.any(ranges[:, 1] > num_data)
        or np.any(ranges[:, 0] > ranges[:, 1])
    ):
        raise IndexError("Pattern range is out of bounds")

    if ones_idx is not None or multi_idx is not None:
        if ones_idx is None or multi_idx is None:
            raise ValueError("Pattern counts do not match Parquet v2 payload size")
        idx_pair = (id(ones_idx), id(multi_idx))
        already_validated = cache is not None and cache.get("valid_idx") == idx_pair
        if not already_validated:
            if (
                ones_idx.ndim != 1
                or multi_idx.ndim != 1
                or ones_idx.size != num_data + 1
                or multi_idx.size != num_data + 1
                or ones_idx[0] != 0
                or multi_idx[0] != 0
                or np.any(ones_idx[1:] < ones_idx[:-1])
                or np.any(multi_idx[1:] < multi_idx[:-1])
            ):
                raise ValueError("Pattern counts do not match Parquet v2 payload size")
            if cache is not None:
                cache["valid_idx"] = idx_pair

    if num_data == 0 or len(ranges) == 0:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int32),
        )

    is_delta = position_encoding == "delta"
    single = ranges.shape == (1, 2) and int(ranges[0, 1]) == int(ranges[0, 0]) + 1
    is_full = (
        ranges.shape == (1, 2)
        and int(ranges[0, 0]) == 0
        and int(ranges[0, 1]) == num_data
    )

    _, col_info = _get_file_info(path, cache=cache)

    if single:
        single_workers = env_workers(
            "EMCFILE_PARQUET_INDEXED_WORKERS", 1, allow_zero=True
        )
        if single_workers > 0:
            po, pm, cm = read_single_frame_parquet(
                None,
                int(ranges[0, 0]),
                path=path,
                col_info=col_info,
                num_data=num_data,
                rg_pats=rg_pats,
                is_delta=is_delta,
                is_shuffled=shuffle,
                cache=cache,
            )
            return (
                np.array([len(po)], dtype=np.uint32),
                np.array([len(pm)], dtype=np.uint32),
                po,
                pm,
                cm,
            )

    if is_full:
        workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    else:
        workers = env_workers("EMCFILE_PARQUET_INDEXED_WORKERS", 4, allow_zero=True)

    ranges_i64 = np.ascontiguousarray(ranges, dtype=np.int64)
    ones_u64 = (
        np.ascontiguousarray(ones_idx, dtype=np.uint64)
        if ones_idx is not None
        else None
    )
    multi_u64 = (
        np.ascontiguousarray(multi_idx, dtype=np.uint64)
        if multi_idx is not None
        else None
    )
    mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
    try:
        out_ones, out_multi, out_po, out_pm, out_cm = rs.read_patterns(
            mm_obj,
            col_info,
            ranges_i64,
            ones_u64,
            multi_u64,
            rg_pats,
            num_data,
            is_delta,
            shuffle,
            max(1, workers),
        )
        return out_ones, out_multi, out_po, out_pm, out_cm.view(np.int32)
    finally:
        if mm_local is not None:
            mm_local.close()
        if fp_local is not None:
            fp_local.close()


def read_parquet_patterns(
    path: Path,
    index_ranges: npt.NDArray[np.integer[Any]],
    ones_idx: npt.NDArray[np.uint64] | None = None,
    multi_idx: npt.NDArray[np.uint64] | None = None,
    *,
    num_data: int,
    rg_pats: int,
    position_encoding: str,
    shuffle: bool,
    payload_codec: str = "none",
    pf: Any | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    """Read selected pattern ranges from a 1-row-per-pattern Parquet v2 file."""
    _, _, po, pm, cm = read_parquet_patterns_with_counts(
        path,
        index_ranges,
        ones_idx,
        multi_idx,
        num_data=num_data,
        rg_pats=rg_pats,
        position_encoding=position_encoding,
        shuffle=shuffle,
        payload_codec=payload_codec,
        pf=pf,
        cache=cache,
    )
    return po, pm, cm


def read_parquet_column(
    path: Path,
    column: str,
    offsets: npt.NDArray[np.uint64] | None = None,
    *,
    num_data: int,
    rg_pats: int,
    is_delta: bool,
    shuffle: bool,
    payload_codec: str = "none",
    dtype: str = "u4",
    cache: dict[str, Any] | None = None,
) -> npt.NDArray[Any]:
    """Read and decode a single full event column from a Parquet v2 file."""
    del offsets, payload_codec
    if num_data == 0:
        return np.empty(0, dtype=dtype)

    rs = _require_parquet_rs()
    workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    _, col_info = _get_file_info(path, cache=cache)
    n_rgs = (num_data + rg_pats - 1) // rg_pats

    col_idx_map = {"place_ones": 0, "place_multi": 1, "count_multi": 2}
    if column not in col_idx_map:
        raise ValueError(f"Unknown Parquet payload column {column!r}")
    c_idx = col_idx_map[column]
    n_workers = max(1, min(max(1, workers), max(1, n_rgs)))
    mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
    try:
        out_u32 = rs.decode_col(
            mm_obj,
            col_info,
            c_idx,
            rg_pats,
            num_data,
            is_delta,
            shuffle,
            n_workers,
        )
    finally:
        if mm_local is not None:
            mm_local.close()
        if fp_local is not None:
            fp_local.close()
    return out_u32.view(dtype)


__all__ = [
    "PARQUET_EXTRA_MESSAGE",
    "PARQUET_SUFFIXES",
    "ThreadPoolExecutor",
    "_ensure_decode_kernel",
    "available",
    "is_parquet_file",
    "read_parquet_column",
    "read_parquet_metadata",
    "read_parquet_ones_multi",
    "read_parquet_patterns",
    "read_parquet_patterns_with_counts",
    "read_single_frame_parquet",
    "write_parquet_patterns",
]
