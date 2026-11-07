"""Parquet 1-row-per-pattern reader and Rust SIMD+Zstd fast paths."""

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


def _load_parquet_rs() -> Any:
    try:
        from . import _parquet_rs as mod  # type: ignore[attr-defined]

        return mod
    except ImportError:
        return None


_parquet_rs = _load_parquet_rs()

if TYPE_CHECKING:
    from ._emc_patterns import PatternsSOne, PatternsSOneBase

PARQUET_SUFFIXES = (".parquet", ".pq")
_PARQUET_MAGIC = b"PAR1"
PARQUET_EXTRA_MESSAGE = (
    "Parquet support requires the optional 'parquet' dependencies. "
    "Install with: pip install 'emcfile[parquet]'"
)

_PAYLOAD_COLUMNS = ["place_ones", "place_multi", "count_multi"]

_file_info_lock = threading.Lock()
_file_info_lru: dict[
    tuple[str, int, int, bytes, bytes],
    tuple[dict[str, Any], npt.NDArray[np.int64] | None],
] = {}

_pool_lock = threading.Lock()
_shared_pool: ThreadPoolExecutor | None = None
_shared_pool_workers: int = 0
_shared_pool_cls: Any = None


def _get_worker_pool(workers: int) -> ThreadPoolExecutor:
    global _shared_pool, _shared_pool_workers, _shared_pool_cls
    if (
        ThreadPoolExecutor is not _shared_pool_cls
        or _shared_pool is None
        or _shared_pool_workers != workers
    ):
        with _pool_lock:
            if (
                ThreadPoolExecutor is not _shared_pool_cls
                or _shared_pool is None
                or _shared_pool_workers != workers
            ):
                if _shared_pool is not None:
                    _shared_pool.shutdown(wait=False)
                _shared_pool = ThreadPoolExecutor(max_workers=workers)
                _shared_pool_workers = workers
                _shared_pool_cls = ThreadPoolExecutor
    return _shared_pool


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(PARQUET_EXTRA_MESSAGE) from exc
    return pa, pq


def available() -> bool:
    """Whether native Rust + Zstandard acceleration is available."""
    return sys.byteorder == "little" and all(
        importlib.util.find_spec(name) is not None
        for name in ("pyarrow", "numba", "zstandard")
    )


def _accel_enabled() -> bool:
    try:
        import zstandard as _zstd  # noqa: F401
    except ImportError:
        pass
    return (
        _parquet_rs is not None
        and sys.byteorder == "little"
        and importlib.util.find_spec("numba") is not None
        and importlib.util.find_spec("zstandard") is not None
    )


def _ensure_decode_kernel() -> Any:
    return None


def _encode_rg_numpy(
    src: npt.NDArray[Any],
    pat_offsets: npt.NDArray[np.uint64],
    ps: int,
    pe: int,
    *,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
    dtype: str,
) -> npt.NDArray[np.uint8]:
    del dtype
    src_u32 = np.ascontiguousarray(src).view(np.uint32)
    offs_u64 = np.ascontiguousarray(pat_offsets, dtype=np.uint64)
    return _parquet_rs.encode_rg_u8(
        src_u32, offs_u64, ps, pe, is_delta, is_shuffled, check_sorted
    )


def _decode_rg_spans_numpy(
    rg_u8: npt.NDArray[np.uint8],
    rg_ps: int,
    rg_local_offs: npt.NDArray[np.int64],
    out_u32: npt.NDArray[np.uint32],
    p_firsts: npt.NDArray[np.int64],
    p_lasts: npt.NDArray[np.int64],
    dst_offs: npt.NDArray[np.int64],
    *,
    is_delta: bool,
    is_shuffled: bool,
) -> None:
    _parquet_rs.decode_rg_spans_u8(
        np.ascontiguousarray(rg_u8, dtype=np.uint8),
        rg_ps,
        np.ascontiguousarray(rg_local_offs, dtype=np.int64),
        out_u32,
        np.ascontiguousarray(p_firsts, dtype=np.int64),
        np.ascontiguousarray(p_lasts, dtype=np.int64),
        np.ascontiguousarray(dst_offs, dtype=np.int64),
        is_delta,
        is_shuffled,
    )


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
) -> tuple[str, int | None]:
    if compression is None or compression in ("none", "uncompressed", 0):
        return "NONE", None
    if isinstance(compression, str):
        name = compression.strip().lower()
        if name in {"zstd", "zstandard"}:
            level = 3 if compression_opts is None else int(compression_opts)
            return "ZSTD", level
        if name in {"lz4", "lz4_raw"}:
            return "LZ4", None
        if name == "snappy":
            return "SNAPPY", None
        if name in {"gzip", "zlib"}:
            level = 4 if compression_opts is None else int(compression_opts)
            return "GZIP", level
        if name == "brotli":
            level = 1 if compression_opts is None else int(compression_opts)
            return "BROTLI", level
    if compression == 32015:
        level = 3 if compression_opts is None else int(compression_opts)
        return "ZSTD", level
    raise ValueError(f"Unsupported Parquet compression {compression!r}")


def _meta_dict_from_kv(
    meta: dict[bytes, bytes], col_info: npt.NDArray[np.int64] | None = None
) -> dict[str, Any]:
    version = meta.get(b"version", b"2").decode("ascii")
    if version != "2":
        raise ValueError(f"Unsupported Parquet EMC format version {version!r}")
    position_encoding = meta.get(b"position_encoding", b"absolute").decode("ascii")
    if position_encoding not in {"absolute", "delta"}:
        raise ValueError(f"Unsupported Parquet position encoding {position_encoding!r}")
    num_pix = int(meta[b"num_pix"].decode("ascii"))
    num_data = int(meta[b"num_data"].decode("ascii"))
    rg_pats = int(meta.get(b"rg_pats", b"1").decode("ascii"))
    shuffle = meta.get(b"shuffle", b"0") == b"1"
    payload_codec = meta.get(b"payload_codec", b"none").decode("ascii")
    if col_info is not None:
        num_ones: int | None = int(col_info[:, 0, 2].sum()) if len(col_info) > 0 else 0
        num_multi: int | None = int(col_info[:, 1, 2].sum()) if len(col_info) > 0 else 0
    else:
        num_ones = (
            int(meta[b"num_ones"].decode("ascii")) if b"num_ones" in meta else None
        )
        num_multi = (
            int(meta[b"num_multi"].decode("ascii")) if b"num_multi" in meta else None
        )
    return {
        "version": version,
        "num_pix": num_pix,
        "num_data": num_data,
        "rg_pats": rg_pats,
        "position_encoding": position_encoding,
        "shuffle": shuffle,
        "payload_codec": payload_codec,
        "num_ones": num_ones,
        "num_multi": num_multi,
    }


def _get_file_info(
    path: Path, cache: dict[str, Any] | None = None
) -> tuple[dict[str, Any], npt.NDArray[np.int64] | None]:
    if cache is not None and "file_info" in cache:
        return cache["file_info"]

    resolved = path.resolve()
    if _parquet_rs is not None:
        try:
            st = resolved.stat()
            if st.st_size >= 16:
                with resolved.open("rb") as fp:
                    head_sig = fp.read(min(64, st.st_size))
                    fp.seek(-8, os.SEEK_END)
                    tail8 = fp.read(8)
                    if tail8[4:8] == _PARQUET_MAGIC:
                        fl = int.from_bytes(tail8[:4], "little")
                        if 0 < fl <= st.st_size - 12:
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

                            parsed = _parquet_rs.parse_parquet_footer(footer_bytes)
                            if parsed is not None:
                                with _file_info_lock:
                                    _file_info_lru[cache_key] = parsed
                                    if len(_file_info_lru) > 16:
                                        _file_info_lru.pop(next(iter(_file_info_lru)))
                                if cache is not None:
                                    cache["file_info"] = parsed
                                return parsed
        except OSError:
            pass

    _, pq = _require_pyarrow()
    meta = pq.read_metadata(path).metadata or {}
    meta_dict = _meta_dict_from_kv(meta, None)
    res_fallback: tuple[dict[str, Any], npt.NDArray[np.int64] | None] = (
        meta_dict,
        None,
    )
    if cache is not None:
        cache["file_info"] = res_fallback
    return res_fallback


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


def _split_ranges_by_rg(
    ranges: npt.NDArray[np.integer[Any]], rg_pats: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    return _parquet_rs.split_ranges_by_rg(
        np.ascontiguousarray(ranges, dtype=np.int64), rg_pats
    )


def _extract_chunk_offs_and_u8(
    col_chunked: Any,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8]]:
    if col_chunked.num_chunks == 0 or len(col_chunked) == 0:
        return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.uint8)
    chunk = (
        col_chunked.chunk(0)
        if col_chunked.num_chunks == 1
        else col_chunked.combine_chunks()
    )
    off_buf = chunk.buffers()[1]
    val_buf = chunk.buffers()[2]
    if off_buf is None:
        return np.zeros(len(chunk) + 1, dtype=np.int64), np.empty(0, dtype=np.uint8)
    offs_i32 = np.frombuffer(off_buf, dtype=np.int32, count=len(chunk) + 1)
    s_b = int(offs_i32[0])
    e_b = int(offs_i32[-1])
    local_offs = ((offs_i32.astype(np.int64) - s_b) >> 2).astype(np.int64, copy=False)
    if val_buf is None or e_b <= s_b:
        return local_offs, np.empty(0, dtype=np.uint8)
    u8 = np.frombuffer(val_buf, dtype=np.uint8, count=e_b - s_b, offset=s_b)
    return local_offs, u8


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

    workers = env_workers("EMCFILE_PARQUET_WRITE_WORKERS", 4, allow_zero=True)
    use_rs = num_data > 0 and workers > 0 and _accel_enabled()
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

    if pq_codec == "ZSTD" and use_rs:
        zstd_level = 1 if pq_level is None else int(pq_level)
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

                meta_batch = _parquet_rs.write_batch_zstd(
                    fp.fileno(),
                    po_src,
                    pm_src,
                    cm_src,
                    ones_idx,
                    multi_idx,
                    b_pats,
                    rg_pats,
                    page_pats,
                    zstd_level,
                    is_delta,
                    shuffle,
                    check_sorted,
                    min(workers, b_rgs),
                )
                batch_pats_list.append(b_pats)
                meta_batches_list.append(meta_batch)

            _parquet_rs.write_parquet_footer(
                fp.fileno(),
                num_data,
                rg_pats,
                batch_pats_list,
                meta_batches_list,
                list(meta_kv.items()),
            )
        return

    pa, pq = _require_pyarrow()
    schema = pa.schema(
        [
            pa.field("place_ones", pa.binary(), nullable=False),
            pa.field("place_multi", pa.binary(), nullable=False),
            pa.field("count_multi", pa.binary(), nullable=False),
        ],
        metadata=meta_kv,
    )

    if num_data == 0:
        empty_tbl = pa.Table.from_arrays(
            [
                pa.array([], type=pa.binary()),
                pa.array([], type=pa.binary()),
                pa.array([], type=pa.binary()),
            ],
            schema=schema,
        )
        pq.write_table(
            empty_tbl,
            path,
            compression=pq_codec,
            compression_level=pq_level,
            use_dictionary=False,
            write_statistics=False,
        )
        return

    rg_bytes = max(1, rg_pats * bytes_per_pat)
    max_rgs_per_batch = max(1, buffer_size // rg_bytes)

    def encode_batch_arrow(batch: PatternsSOne) -> Any:
        b_pats = batch.num_data
        b_rgs = (b_pats + rg_pats - 1) // rg_pats
        ones_b = batch.ones_idx * 4
        multi_b = batch.multi_idx * 4
        po_chunks = [None] * b_rgs
        pm_chunks = [None] * b_rgs
        cm_chunks = [None] * b_rgs

        for rg in range(b_rgs):
            ps = rg * rg_pats
            pe = min(b_pats, (rg + 1) * rg_pats)
            nr = pe - ps
            u8_po = _encode_rg_numpy(
                batch.place_ones,
                batch.ones_idx,
                ps,
                pe,
                is_delta=is_delta,
                is_shuffled=shuffle,
                check_sorted=check_sorted,
                dtype="<u4",
            )
            u8_pm = _encode_rg_numpy(
                batch.place_multi,
                batch.multi_idx,
                ps,
                pe,
                is_delta=is_delta,
                is_shuffled=shuffle,
                check_sorted=check_sorted,
                dtype="<u4",
            )
            u8_cm = _encode_rg_numpy(
                batch.count_multi,
                batch.multi_idx,
                ps,
                pe,
                is_delta=False,
                is_shuffled=shuffle,
                check_sorted=False,
                dtype="<i4",
            )
            off_o = (ones_b[ps : pe + 1] - ones_b[ps]).astype(np.int32)
            off_m = (multi_b[ps : pe + 1] - multi_b[ps]).astype(np.int32)
            buf_off_o = pa.py_buffer(off_o)
            buf_off_m = pa.py_buffer(off_m)
            po_chunks[rg] = pa.BinaryArray.from_buffers(
                pa.binary(), nr, [None, buf_off_o, pa.py_buffer(u8_po)]
            )
            pm_chunks[rg] = pa.BinaryArray.from_buffers(
                pa.binary(), nr, [None, buf_off_m, pa.py_buffer(u8_pm)]
            )
            cm_chunks[rg] = pa.BinaryArray.from_buffers(
                pa.binary(), nr, [None, buf_off_m, pa.py_buffer(u8_cm)]
            )

        return pa.Table.from_arrays(
            [
                pa.chunked_array(po_chunks, type=pa.binary()),
                pa.chunked_array(pm_chunks, type=pa.binary()),
                pa.chunked_array(cm_chunks, type=pa.binary()),
            ],
            schema=schema,
        )

    with pq.ParquetWriter(
        path,
        schema,
        compression=pq_codec,
        compression_level=pq_level,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=max(4096, page_bytes),
    ) as writer:
        for batch in _iter_rg_batches(datas, rg_pats, max_rgs_per_batch):
            tbl = encode_batch_arrow(batch)
            writer.write_table(tbl, row_group_size=max(1, rg_pats))


def read_parquet_metadata(
    path: Path, *, cache: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Read header metadata from a Parquet v2 pattern file."""
    meta_dict, _ = _get_file_info(path, cache=cache)
    return meta_dict


def _counts_from_chunked_binary(
    col_chunked: Any, num_data: int
) -> npt.NDArray[np.uint32]:
    out = np.empty(num_data, dtype=np.uint32)
    pos = 0
    for chunk in col_chunked.chunks:
        n_c = len(chunk)
        if n_c == 0:
            continue
        off_buf = chunk.buffers()[1]
        offs = np.frombuffer(off_buf, dtype=np.int32, count=n_c + 1)
        out[pos : pos + n_c] = ((offs[1:] - offs[:-1]) >> 2).astype(
            np.uint32, copy=False
        )
        pos += n_c
    return out


def read_parquet_ones_multi(
    path: Path,
    num_data: int,
    *,
    pf: Any | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Derive ``ones`` and ``multi`` count arrays from a Parquet v2 file."""
    if num_data == 0:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

    meta_dict, col_info = _get_file_info(path, cache=cache)
    workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    if col_info is not None and _accel_enabled():
        rg_pats = int(meta_dict["rg_pats"])
        n_workers = max(1, min(max(1, workers), len(col_info)))
        mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
        try:
            return _parquet_rs.read_ones_multi(
                mm_obj, col_info, rg_pats, num_data, n_workers
            )
        finally:
            if mm_local is not None:
                mm_local.close()
            if fp_local is not None:
                fp_local.close()

    _, pq = _require_pyarrow()
    if pf is not None:
        tbl = pf.read(columns=["place_ones", "place_multi"], use_threads=True)
    else:
        tbl = pq.read_table(
            path,
            columns=["place_ones", "place_multi"],
            use_threads=True,
            memory_map=True,
        )
    ones = _counts_from_chunked_binary(tbl.column("place_ones"), num_data)
    multi = _counts_from_chunked_binary(tbl.column("place_multi"), num_data)
    return ones, multi


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
    del ones_idx, multi_idx, payload_codec
    rg = frame // rg_pats
    ps = rg * rg_pats
    pe = min(num_data, (rg + 1) * rg_pats)

    if col_info is not None and _accel_enabled():
        assert path is not None
        mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
        try:
            po, pm, cm = _parquet_rs.decode_single_frame(
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

    if pf is None:
        assert path is not None
        _, pq = _require_pyarrow()
        pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)

    cached_vecs = None
    if cache is not None:
        lru = cache.get("lru")
        if lru is not None and rg in lru:
            cached_vecs = lru.pop(rg)
            lru[rg] = cached_vecs

    if cached_vecs is None:
        rt = pf.read_row_group(rg, columns=_PAYLOAD_COLUMNS, use_threads=False)
        off_o, u8_po = _extract_chunk_offs_and_u8(rt.column("place_ones"))
        off_m, u8_pm = _extract_chunk_offs_and_u8(rt.column("place_multi"))
        _, u8_cm = _extract_chunk_offs_and_u8(rt.column("count_multi"))
        cached_vecs = (off_o, u8_po, off_m, u8_pm, u8_cm)
        if cache is not None:
            lru = cache.setdefault("lru", {})
            lru[rg] = cached_vecs
            if len(lru) > 2:
                lru.pop(next(iter(lru)))
    else:
        off_o, u8_po, off_m, u8_pm, u8_cm = cached_vecs

    loc_f = frame - ps
    k_o = int(off_o[loc_f + 1]) - int(off_o[loc_f])
    po = np.empty(k_o, dtype=np.uint32)
    if k_o > 0:
        _decode_rg_spans_numpy(
            u8_po,
            ps,
            off_o,
            po,
            np.array([frame], dtype=np.int64),
            np.array([frame + 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            is_delta=is_delta,
            is_shuffled=is_shuffled,
        )

    k_m = int(off_m[loc_f + 1]) - int(off_m[loc_f])
    pm = np.empty(k_m, dtype=np.uint32)
    cm_u32 = np.empty(k_m, dtype=np.uint32)
    if k_m > 0:
        f_arr = np.array([frame], dtype=np.int64)
        l_arr = np.array([frame + 1], dtype=np.int64)
        d_arr = np.array([0], dtype=np.int64)
        _decode_rg_spans_numpy(
            u8_pm,
            ps,
            off_m,
            pm,
            f_arr,
            l_arr,
            d_arr,
            is_delta=is_delta,
            is_shuffled=is_shuffled,
        )
        _decode_rg_spans_numpy(
            u8_cm,
            ps,
            off_m,
            cm_u32,
            f_arr,
            l_arr,
            d_arr,
            is_delta=False,
            is_shuffled=is_shuffled,
        )

    return po, pm, cm_u32.view(np.int32)


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
    del payload_codec
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
                pf,
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

    if col_info is not None and workers > 0 and _accel_enabled():
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
            out_ones, out_multi, out_po, out_pm, out_cm = _parquet_rs.read_patterns(
                mm_obj,
                col_info,
                ranges_i64,
                ones_u64,
                multi_u64,
                rg_pats,
                num_data,
                is_delta,
                shuffle,
                workers,
            )
            return out_ones, out_multi, out_po, out_pm, out_cm.view(np.int32)
        finally:
            if mm_local is not None:
                mm_local.close()
            if fp_local is not None:
                fp_local.close()

    rgs, firsts, lasts = _split_ranges_by_rg(ranges, rg_pats)
    if len(rgs) == 0:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int32),
        )

    pat_lens = lasts - firsts
    dst_pats = np.empty(len(firsts) + 1, dtype=np.int64)
    dst_pats[0] = 0
    np.cumsum(pat_lens, out=dst_pats[1:])
    total_sel_pats = int(dst_pats[-1])

    out_ones = np.empty(total_sel_pats, dtype=np.uint32)
    out_multi = np.empty(total_sel_pats, dtype=np.uint32)

    order = (
        np.arange(len(rgs), dtype=np.int64) if is_full else np.lexsort((firsts, rgs))
    )
    sorted_rgs = rgs[order]
    unique_rgs, split_pts = np.unique(sorted_rgs, return_index=True)

    _, pq = _require_pyarrow()
    if pf is None:
        pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)

    groups = np.split(order, split_pts[1:])
    n_groups = len(unique_rgs)
    group_tables: list[Any] = [None] * n_groups

    def read_group_tbl(g_idx: int) -> None:
        rg = int(unique_rgs[g_idx])
        group_tables[g_idx] = pf.read_row_group(
            rg, columns=_PAYLOAD_COLUMNS, use_threads=False
        )

    if workers > 1 and n_groups > 1:
        pool = _get_worker_pool(min(workers, n_groups))
        list(pool.map(read_group_tbl, range(n_groups)))
    else:
        for g_idx in range(n_groups):
            read_group_tbl(g_idx)

    ones_lens = np.empty(len(firsts), dtype=np.int64)
    multi_lens = np.empty(len(firsts), dtype=np.int64)
    group_parsed: list[
        tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.uint8],
            npt.NDArray[np.int64],
            npt.NDArray[np.uint8],
            npt.NDArray[np.uint8],
        ]
    ] = [None] * n_groups  # type: ignore[list-item]

    for g_idx in range(n_groups):
        rg = int(unique_rgs[g_idx])
        ps = rg * rg_pats
        rt = group_tables[g_idx]
        off_o, u8_po = _extract_chunk_offs_and_u8(rt.column("place_ones"))
        off_m, u8_pm = _extract_chunk_offs_and_u8(rt.column("place_multi"))
        _, u8_cm = _extract_chunk_offs_and_u8(rt.column("count_multi"))
        group_parsed[g_idx] = (off_o, u8_po, off_m, u8_pm, u8_cm)

        span_ids = groups[g_idx]
        for sid in span_ids:
            f = int(firsts[sid]) - ps
            last_p = int(lasts[sid]) - ps
            dp = int(dst_pats[sid])
            out_ones[dp : dp + (last_p - f)] = (
                off_o[f + 1 : last_p + 1] - off_o[f:last_p]
            ).astype(np.uint32)
            out_multi[dp : dp + (last_p - f)] = (
                off_m[f + 1 : last_p + 1] - off_m[f:last_p]
            ).astype(np.uint32)
            ones_lens[sid] = int(off_o[last_p]) - int(off_o[f])
            multi_lens[sid] = int(off_m[last_p]) - int(off_m[f])

    dst_ones = np.empty(len(firsts) + 1, dtype=np.int64)
    dst_ones[0] = 0
    np.cumsum(ones_lens, out=dst_ones[1:])

    dst_multi = np.empty(len(firsts) + 1, dtype=np.int64)
    dst_multi[0] = 0
    np.cumsum(multi_lens, out=dst_multi[1:])

    out_po = np.empty(int(dst_ones[-1]), dtype=np.uint32)
    out_pm = np.empty(int(dst_multi[-1]), dtype=np.uint32)
    out_cm = np.empty(int(dst_multi[-1]), dtype=np.uint32)

    def proc_group(g_idx: int) -> None:
        rg = int(unique_rgs[g_idx])
        ps = rg * rg_pats
        span_ids = groups[g_idx]
        p_firsts = firsts[span_ids]
        p_lasts = lasts[span_ids]
        off_o, u8_po, off_m, u8_pm, u8_cm = group_parsed[g_idx]

        if len(u8_po) > 0:
            _decode_rg_spans_numpy(
                u8_po,
                ps,
                off_o,
                out_po,
                p_firsts,
                p_lasts,
                dst_ones[span_ids],
                is_delta=is_delta,
                is_shuffled=shuffle,
            )

        if len(u8_pm) > 0:
            _decode_rg_spans_numpy(
                u8_pm,
                ps,
                off_m,
                out_pm,
                p_firsts,
                p_lasts,
                dst_multi[span_ids],
                is_delta=is_delta,
                is_shuffled=shuffle,
            )
            _decode_rg_spans_numpy(
                u8_cm,
                ps,
                off_m,
                out_cm,
                p_firsts,
                p_lasts,
                dst_multi[span_ids],
                is_delta=False,
                is_shuffled=shuffle,
            )

    if workers > 1 and n_groups > 1:
        pool = _get_worker_pool(min(workers, n_groups))
        list(pool.map(proc_group, range(n_groups)))
    else:
        for g_idx in range(n_groups):
            proc_group(g_idx)

    return out_ones, out_multi, out_po, out_pm, out_cm.view(np.int32)


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

    workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    use_rs = workers > 0 and _accel_enabled()
    _, col_info = _get_file_info(path, cache=cache)
    n_rgs = (num_data + rg_pats - 1) // rg_pats

    col_idx_map = {"place_ones": 0, "place_multi": 1, "count_multi": 2}
    if col_info is not None and column in col_idx_map and use_rs:
        c_idx = col_idx_map[column]
        n_workers = max(1, min(workers, n_rgs))
        mm_obj, fp_local, mm_local = _get_or_open_mmap(path, cache)
        try:
            out_u32 = _parquet_rs.decode_col(
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

    _, pq = _require_pyarrow()
    pf_main = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    rg_Data: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8]]] = [None] * n_rgs  # type: ignore[list-item]
    rg_lens = np.empty(n_rgs, dtype=np.int64)

    for rg in range(n_rgs):
        rt = pf_main.read_row_group(rg, columns=[column], use_threads=False)
        local_offs, u8 = _extract_chunk_offs_and_u8(rt.column(column))
        rg_Data[rg] = (local_offs, u8)
        rg_lens[rg] = int(local_offs[-1])

    rg_offs = np.empty(n_rgs + 1, dtype=np.int64)
    rg_offs[0] = 0
    np.cumsum(rg_lens, out=rg_offs[1:])
    total_elems = int(rg_offs[-1])
    if total_elems == 0:
        return np.empty(0, dtype=dtype)

    out_u32 = np.empty(total_elems, dtype=np.uint32)

    def proc_rg(rg: int) -> None:
        ps = rg * rg_pats
        pe = min(num_data, (rg + 1) * rg_pats)
        local_offs, u8 = rg_Data[rg]
        if len(u8) == 0:
            return
        p_first = np.array([ps], dtype=np.int64)
        p_last = np.array([pe], dtype=np.int64)
        dst_off = np.array([rg_offs[rg]], dtype=np.int64)
        _decode_rg_spans_numpy(
            u8,
            ps,
            local_offs,
            out_u32,
            p_first,
            p_last,
            dst_off,
            is_delta=is_delta,
            is_shuffled=shuffle,
        )

    if workers > 1 and n_rgs > 1:
        pool = _get_worker_pool(min(workers, n_rgs))
        list(pool.map(proc_rg, range(n_rgs)))
    else:
        for rg in range(n_rgs):
            proc_rg(rg)

    return out_u32.view(dtype)
