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

from ._delta import encode_pattern_local_delta
from ._h5_workers import env_workers


def _load_parquet_rs() -> Any:
    try:
        from . import _parquet_rs as mod  # type: ignore[attr-defined]

        return mod
    except ImportError:
        import shutil
        import subprocess

        pkg_dir = Path(__file__).resolve().parent
        rust_manifest = pkg_dir.parent.parent / "rust" / "Cargo.toml"
        if rust_manifest.is_file() and shutil.which("cargo") is not None:
            env = os.environ.copy()
            link_arg = (
                "-Wl,-undefined,dynamic_lookup"
                if sys.platform == "darwin"
                else "-Wl,--unresolved-symbols=ignore-in-object-files"
            )
            env.setdefault("RUSTFLAGS", f"-C target-cpu=native -C link-arg={link_arg}")
            subprocess.run(
                ["cargo", "build", "--release", "--manifest-path", str(rust_manifest)],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ext = ".dylib" if sys.platform == "darwin" else ".so"
            built = rust_manifest.parent / "target" / "release" / f"lib_parquet_rs{ext}"
            if built.is_file():
                shutil.copy2(built, pkg_dir / "_parquet_rs.so")
                from . import _parquet_rs as mod  # type: ignore[attr-defined]

                return mod
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

_COUNT_COLUMNS = ["ones", "multi"]
_PAYLOAD_COLUMNS = ["place_ones", "place_multi", "count_multi"]
_ALL_COLUMNS = ["ones", "multi", "place_ones", "place_multi", "count_multi"]

_template_lock = threading.Lock()
_zstd_footer_template: tuple[bytes, bytes] | None = None

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
    s_rg = int(pat_offsets[ps])
    e_rg = int(pat_offsets[pe])
    n_rg = e_rg - s_rg
    if n_rg == 0:
        return np.empty(0, dtype=np.uint8)

    sub_src = src[s_rg:e_rg]
    sub_offs = pat_offsets[ps : pe + 1] - pat_offsets[ps]
    if is_delta:
        sub_counts = pat_offsets[ps + 1 : pe + 1] - pat_offsets[ps:pe]
        encoded = encode_pattern_local_delta(
            sub_src, sub_counts, check_sorted=check_sorted
        )
    else:
        if check_sorted:
            sub_counts = pat_offsets[ps + 1 : pe + 1] - pat_offsets[ps:pe]
            encode_pattern_local_delta(sub_src, sub_counts, check_sorted=True)
        encoded = np.ascontiguousarray(sub_src, dtype=dtype)

    out_u8 = np.empty(n_rg * 4, dtype=np.uint8)
    for p in range(pe - ps):
        s = int(sub_offs[p])
        e = int(sub_offs[p + 1])
        k = e - s
        if k <= 0:
            continue
        chunk_u8 = encoded[s:e].view(np.uint8)
        b_s = s * 4
        if is_shuffled:
            out_u8[b_s : b_s + 4 * k] = chunk_u8.reshape(k, 4).T.reshape(-1)
        else:
            out_u8[b_s : b_s + 4 * k] = chunk_u8
    return out_u8


def _decode_rg_spans_numpy(
    rg_u8: npt.NDArray[np.uint8],
    rg_base_elem: int,
    pat_offsets: npt.NDArray[np.int64],
    out_u32: npt.NDArray[np.uint32],
    p_firsts: npt.NDArray[np.int64],
    p_lasts: npt.NDArray[np.int64],
    dst_offs: npt.NDArray[np.int64],
    *,
    is_delta: bool,
    is_shuffled: bool,
) -> None:
    for s_i in range(len(p_firsts)):
        f = int(p_firsts[s_i])
        last_p = int(p_lasts[s_i])
        out_idx = int(dst_offs[s_i])
        for p in range(f, last_p):
            s = int(pat_offsets[p])
            e = int(pat_offsets[p + 1])
            k = e - s
            if k <= 0:
                continue
            b_s = (s - rg_base_elem) * 4
            if is_shuffled:
                pat_u32 = (
                    rg_u8[b_s : b_s + 4 * k]
                    .reshape(4, k)
                    .T.copy()
                    .view("<u4")
                    .reshape(-1)
                )
            else:
                pat_u32 = rg_u8[b_s : b_s + 4 * k].view("<u4").copy()
            if is_delta:
                np.cumsum(pat_u32, dtype=np.uint32, out=out_u32[out_idx : out_idx + k])
            else:
                out_u32[out_idx : out_idx + k] = pat_u32
            out_idx += k


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


def _enc_uvarint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _enc_zigzag(n: int) -> bytes:
    return _enc_uvarint((n << 1) ^ (n >> 63))


def _dec_uvarint_py(mv: memoryview | bytes, pos: int) -> tuple[int, int]:
    b0 = mv[pos]
    if b0 < 0x80:
        return b0, pos + 1
    b1 = mv[pos + 1]
    if b1 < 0x80:
        return (b0 & 0x7F) | (b1 << 7), pos + 2
    n = (b0 & 0x7F) | ((b1 & 0x7F) << 7)
    shift = 14
    pos += 2
    while True:
        b = mv[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if b < 0x80:
            return n, pos
        shift += 7


def _dec_zigzag_py(mv: memoryview | bytes, pos: int) -> tuple[int, int]:
    n, pos = _dec_uvarint_py(mv, pos)
    return (n >> 1) ^ -(n & 1), pos


_COL_ENCODING_STATS_BYTES = bytes.fromhex("491c15001500150200")


def _make_col_chunk_meta(
    col_type: int,
    col_name: bytes,
    codec: int,
    num_values: int,
    total_uncomp: int,
    total_comp: int,
    page_offset: int,
) -> bytes:
    return (
        b"\x26\x00\x1c"
        + b"\x15"
        + _enc_zigzag(col_type)
        + b"\x19\x25\x00\x06"
        + b"\x19\x18"
        + _enc_uvarint(len(col_name))
        + col_name
        + b"\x15"
        + _enc_zigzag(codec)
        + b"\x16"
        + _enc_zigzag(num_values)
        + b"\x16"
        + _enc_zigzag(total_uncomp)
        + b"\x16"
        + _enc_zigzag(total_comp)
        + b"\x26"
        + _enc_zigzag(page_offset)
        + _COL_ENCODING_STATS_BYTES
        + b"\x00\x00"
    )


def _get_zstd_footer_template() -> tuple[bytes, bytes]:
    global _zstd_footer_template
    if _zstd_footer_template is None:
        with _template_lock:
            if _zstd_footer_template is None:
                pa, pq = _require_pyarrow()
                schema = pa.schema(
                    [
                        pa.field("ones", pa.uint32(), nullable=False),
                        pa.field("multi", pa.uint32(), nullable=False),
                        pa.field("place_ones", pa.binary(), nullable=False),
                        pa.field("place_multi", pa.binary(), nullable=False),
                        pa.field("count_multi", pa.binary(), nullable=False),
                    ]
                )
                sink = pa.BufferOutputStream()
                with pq.ParquetWriter(
                    sink,
                    schema,
                    use_dictionary=False,
                    write_statistics=False,
                    compression="ZSTD",
                    store_schema=False,
                ) as w:
                    w.write_table(
                        pa.Table.from_pydict(
                            {
                                "ones": [0],
                                "multi": [0],
                                "place_ones": [b""],
                                "place_multi": [b""],
                                "count_multi": [b""],
                            },
                            schema=schema,
                        )
                    )
                tb = sink.getvalue().to_pybytes()
                tfl = int.from_bytes(tb[-8:-4], "little")
                tf = tb[len(tb) - tfl - 8 : -8]
                schema_end = tf.index(b"count_multi\x00") + len(b"count_multi\x00")
                created_by_pos = tf.index(b"\x28\x20parquet-cpp", schema_end)
                _zstd_footer_template = (tf[:schema_end], tf[created_by_pos:])
    return _zstd_footer_template


def _encode_kv_metadata(kv: dict[bytes, bytes]) -> bytes:
    n_kv = len(kv)
    if n_kv == 0:
        return b""
    hdr = (
        bytes([0x19, (n_kv << 4) | 0x0C])
        if n_kv < 15
        else b"\x19\xfc" + _enc_uvarint(n_kv)
    )
    parts = [hdr]
    for k, v in kv.items():
        parts.append(b"\x18" + _enc_uvarint(len(k)) + k)
        parts.append(b"\x18" + _enc_uvarint(len(v)) + v)
        parts.append(b"\x00")
    return b"".join(parts)


def _decode_kv_metadata_py(
    mv: memoryview | bytes, pos: int
) -> dict[bytes, bytes] | None:
    if pos + 2 > len(mv) or mv[pos] != 0x19:
        return None
    hdr_b = mv[pos + 1]
    if hdr_b == 0xFC:
        n_kv, pos = _dec_uvarint_py(mv, pos + 2)
    else:
        n_kv = hdr_b >> 4
        pos += 2
    out: dict[bytes, bytes] = {}
    for _ in range(n_kv):
        if pos >= len(mv) or mv[pos] != 0x18:
            return None
        k_len, pos = _dec_uvarint_py(mv, pos + 1)
        k = bytes(mv[pos : pos + k_len])
        pos += k_len
        if pos >= len(mv) or mv[pos] != 0x18:
            return None
        v_len, pos = _dec_uvarint_py(mv, pos + 1)
        v = bytes(mv[pos : pos + v_len])
        pos += v_len
        if pos >= len(mv) or mv[pos] != 0x00:
            return None
        pos += 1
        out[k] = v
    return out


def _parse_our_footer_py(
    footer_bytes: bytes,
) -> tuple[dict[bytes, bytes], npt.NDArray[np.int64]] | None:
    schema_prefix, _ = _get_zstd_footer_template()
    if not footer_bytes.startswith(schema_prefix):
        return None
    pos = len(schema_prefix)
    n_len = len(footer_bytes)
    if pos >= n_len or footer_bytes[pos] != 0x16:
        return None
    _, pos = _dec_zigzag_py(footer_bytes, pos + 1)
    if pos + 1 >= n_len or footer_bytes[pos] != 0x19:
        return None
    hdr_b = footer_bytes[pos + 1]
    if hdr_b == 0xFC:
        n_rgs, pos = _dec_uvarint_py(footer_bytes, pos + 2)
    else:
        n_rgs = hdr_b >> 4
        pos += 2
    if n_rgs < 0:
        return None

    col_info = np.empty((n_rgs, 5, 2), dtype=np.int64)
    col_name_lens = (4, 5, 10, 11, 11)
    for rg in range(n_rgs):
        if (
            pos + 1 >= n_len
            or footer_bytes[pos] != 0x19
            or footer_bytes[pos + 1] != 0x5C
        ):
            return None
        pos += 2
        for c in range(5):
            if (
                pos + 4 >= n_len
                or footer_bytes[pos] != 0x26
                or footer_bytes[pos + 3] != 0x15
            ):
                return None
            _, pos = _dec_zigzag_py(footer_bytes, pos + 4)
            pos += 7 + col_name_lens[c]
            if pos >= n_len or footer_bytes[pos] != 0x15:
                return None
            codec, pos = _dec_zigzag_py(footer_bytes, pos + 1)
            if codec != 6:
                return None
            if pos >= n_len or footer_bytes[pos] != 0x16:
                return None
            _, pos = _dec_zigzag_py(footer_bytes, pos + 1)
            if pos >= n_len or footer_bytes[pos] != 0x16:
                return None
            _, pos = _dec_zigzag_py(footer_bytes, pos + 1)
            if pos >= n_len or footer_bytes[pos] != 0x16:
                return None
            tc, pos = _dec_zigzag_py(footer_bytes, pos + 1)
            if pos >= n_len or footer_bytes[pos] != 0x26:
                return None
            poff, pos = _dec_zigzag_py(footer_bytes, pos + 1)
            col_info[rg, c, 0] = poff
            col_info[rg, c, 1] = tc
            pos += 11
        for _ in range(4):
            if pos >= n_len:
                return None
            _, pos = _dec_zigzag_py(footer_bytes, pos + 1)
        if pos >= n_len or footer_bytes[pos] != 0x00:
            return None
        pos += 1

    kv = _decode_kv_metadata_py(footer_bytes, pos)
    if kv is None:
        return None
    return kv, col_info


def _meta_dict_from_kv(meta: dict[bytes, bytes]) -> dict[str, Any]:
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
    return {
        "version": version,
        "num_pix": num_pix,
        "num_data": num_data,
        "rg_pats": rg_pats,
        "position_encoding": position_encoding,
        "shuffle": shuffle,
        "payload_codec": payload_codec,
    }


def _get_file_info(
    path: Path, cache: dict[str, Any] | None = None
) -> tuple[dict[str, Any], npt.NDArray[np.int64] | None]:
    if cache is not None and "file_info" in cache:
        return cache["file_info"]

    resolved = path.resolve()
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

                        parsed = _parse_our_footer_py(footer_bytes)
                        if parsed is not None:
                            kv, col_info = parsed
                            meta_dict = _meta_dict_from_kv(kv)
                            res: tuple[dict[str, Any], npt.NDArray[np.int64] | None] = (
                                meta_dict,
                                col_info,
                            )
                            with _file_info_lock:
                                _file_info_lru[cache_key] = res
                                if len(_file_info_lru) > 16:
                                    _file_info_lru.pop(next(iter(_file_info_lru)))
                            if cache is not None:
                                cache["file_info"] = res
                            return res
    except OSError:
        pass

    _, pq = _require_pyarrow()
    meta = pq.read_metadata(path).metadata or {}
    meta_dict = _meta_dict_from_kv(meta)
    res_fallback: tuple[dict[str, Any], npt.NDArray[np.int64] | None] = (
        meta_dict,
        None,
    )
    if cache is not None:
        cache["file_info"] = res_fallback
    return res_fallback


def _split_ranges_by_rg(
    ranges: npt.NDArray[np.integer[Any]], rg_pats: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r = np.asarray(ranges, dtype=np.int64)
    valid = r[:, 1] > r[:, 0]
    if not np.all(valid):
        r = r[valid]
    if len(r) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    starts = r[:, 0]
    stops = r[:, 1]
    rg_starts = starts // rg_pats
    rg_stops = (stops - 1) // rg_pats
    counts = rg_stops - rg_starts + 1

    if np.all(counts == 1):
        return (
            np.ascontiguousarray(rg_starts, dtype=np.int64),
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(stops, dtype=np.int64),
        )

    total_spans = int(counts.sum())
    span_range_idx = np.repeat(np.arange(len(r), dtype=np.int64), counts)
    prefix = np.zeros(len(r) + 1, dtype=np.int64)
    np.cumsum(counts, out=prefix[1:])
    intra_offset = np.arange(total_spans, dtype=np.int64) - prefix[span_range_idx]
    rgs = rg_starts[span_range_idx] + intra_offset

    firsts = np.maximum(starts[span_range_idx], rgs * rg_pats)
    lasts = np.minimum(stops[span_range_idx], (rgs + 1) * rg_pats)
    return rgs, firsts, lasts


def _extract_chunk_val_u8(col_chunked: Any) -> npt.NDArray[np.uint8]:
    if col_chunked.num_chunks == 0 or len(col_chunked) == 0:
        return np.empty(0, dtype=np.uint8)
    chunk = (
        col_chunked.chunk(0)
        if col_chunked.num_chunks == 1
        else col_chunked.combine_chunks()
    )
    off_buf = chunk.buffers()[1]
    val_buf = chunk.buffers()[2]
    if val_buf is None or off_buf is None:
        return np.empty(0, dtype=np.uint8)
    offs = np.frombuffer(off_buf, dtype=np.int32, count=len(chunk) + 1)
    s_b = int(offs[0])
    e_b = int(offs[-1])
    if e_b <= s_b:
        return np.empty(0, dtype=np.uint8)
    return np.frombuffer(val_buf, dtype=np.uint8, count=e_b - s_b, offset=s_b)


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

    pa, pq = _require_pyarrow()
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
        chunk_bytes = max(4 << 20, ((total_bytes // 512) + 3) & ~3)

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

        schema_prefix, created_by_suffix = _get_zstd_footer_template()
        kv_bytes = _encode_kv_metadata(meta_kv)
        c_names = (b"ones", b"multi", b"place_ones", b"place_multi", b"count_multi")
        c_types = (1, 1, 6, 6, 6)

        with path.open("wb") as fp:
            fp.write(_PARQUET_MAGIC)
            fp.flush()
            file_pos = 4
            col_metas_all: list[bytes] = []
            total_rgs = 0

            for batch in batches_iter:
                b_pats = batch.num_data
                if b_pats == 0:
                    continue
                b_rgs = (b_pats + rg_pats - 1) // rg_pats
                ones_arr = np.ascontiguousarray(batch.ones, dtype=np.uint32)
                multi_arr = np.ascontiguousarray(batch.multi, dtype=np.uint32)
                po_src = np.ascontiguousarray(batch.place_ones, dtype=np.uint32)
                pm_src = np.ascontiguousarray(batch.place_multi, dtype=np.uint32)
                cm_src = np.ascontiguousarray(batch.count_multi).view(np.uint32)
                ones_idx = np.ascontiguousarray(batch.ones_idx, dtype=np.uint64)
                multi_idx = np.ascontiguousarray(batch.multi_idx, dtype=np.uint64)

                meta_batch = np.empty((b_rgs, 5, 2), dtype=np.int64)
                _parquet_rs.write_batch_zstd(
                    fp.fileno(),
                    ones_arr,
                    multi_arr,
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
                    meta_batch,
                )

                for rg in range(b_rgs):
                    ps = rg * rg_pats
                    pe = min(b_pats, (rg + 1) * rg_pats)
                    nr = pe - ps
                    total_rgs += 1
                    rg_start_pos = file_pos
                    total_uncomp_rg = 0
                    total_comp_rg = 0
                    col_metas = [b"\x19\x5c"]
                    for c in range(5):
                        tu = int(meta_batch[rg, c, 0])
                        tc = int(meta_batch[rg, c, 1])
                        col_metas.append(
                            _make_col_chunk_meta(
                                c_types[c],
                                c_names[c],
                                6,
                                nr,
                                tu,
                                tc,
                                file_pos,
                            )
                        )
                        file_pos += tc
                        total_uncomp_rg += tu
                        total_comp_rg += tc
                    col_metas.append(
                        b"\x16"
                        + _enc_zigzag(total_uncomp_rg)
                        + b"\x16"
                        + _enc_zigzag(nr)
                        + b"\x26"
                        + _enc_zigzag(rg_start_pos)
                        + b"\x16"
                        + _enc_zigzag(total_comp_rg)
                        + b"\x00"
                    )
                    col_metas_all.extend(col_metas)

            list_hdr = (
                bytes([0x19, (total_rgs << 4) | 0x0C])
                if total_rgs < 15
                else b"\x19\xfc" + _enc_uvarint(total_rgs)
            )
            footer = b"".join(
                [
                    schema_prefix,
                    b"\x16",
                    _enc_zigzag(num_data),
                    list_hdr,
                    *col_metas_all,
                    kv_bytes,
                    created_by_suffix,
                ]
            )
            fp.write(footer)
            fp.write(len(footer).to_bytes(4, "little"))
            fp.write(_PARQUET_MAGIC)
        return

    schema = pa.schema(
        [
            pa.field("ones", pa.uint32(), nullable=False),
            pa.field("multi", pa.uint32(), nullable=False),
            pa.field("place_ones", pa.binary(), nullable=False),
            pa.field("place_multi", pa.binary(), nullable=False),
            pa.field("count_multi", pa.binary(), nullable=False),
        ],
        metadata=meta_kv,
    )

    if num_data == 0:
        empty_tbl = pa.Table.from_arrays(
            [
                pa.array(np.empty(0, dtype=np.uint32), type=pa.uint32()),
                pa.array(np.empty(0, dtype=np.uint32), type=pa.uint32()),
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
        ones_arr = pa.array(
            np.ascontiguousarray(batch.ones, dtype=np.uint32), type=pa.uint32()
        )
        multi_arr = pa.array(
            np.ascontiguousarray(batch.multi, dtype=np.uint32), type=pa.uint32()
        )
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
                ones_arr,
                multi_arr,
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


def read_parquet_ones_multi(
    path: Path,
    num_data: int,
    *,
    pf: Any | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Read the ``ones`` and ``multi`` count arrays from a Parquet v2 file."""
    if num_data == 0:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

    meta_dict, col_info = _get_file_info(path, cache=cache)
    if col_info is not None and _accel_enabled():
        rg_pats = int(meta_dict["rg_pats"])
        n_rgs = len(col_info)
        out_ones = np.empty(num_data, dtype=np.uint32)
        out_multi = np.empty(num_data, dtype=np.uint32)

        mm_obj = cache.get("mm") if cache is not None else None
        fp_local = None
        mm_local = None
        try:
            if mm_obj is None:
                fp_local = path.open("rb")
                mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                mm_obj = mm_local
            _parquet_rs.read_ones_multi(
                mm_obj, col_info, n_rgs, rg_pats, num_data, out_ones, out_multi
            )
            return out_ones, out_multi
        finally:
            if mm_local is not None:
                mm_local.close()
            if fp_local is not None:
                fp_local.close()

    _, pq = _require_pyarrow()
    if pf is not None:
        tbl = pf.read(columns=_COUNT_COLUMNS, use_threads=True)
    else:
        tbl = pq.read_table(
            path, columns=_COUNT_COLUMNS, use_threads=True, memory_map=True
        )
    ones = np.ascontiguousarray(
        tbl.column("ones").to_numpy(zero_copy_only=False), dtype=np.uint32
    )
    multi = np.ascontiguousarray(
        tbl.column("multi").to_numpy(zero_copy_only=False), dtype=np.uint32
    )
    return ones, multi


def read_single_frame_parquet(
    pf: Any,
    frame: int,
    ones_idx: npt.NDArray[np.uint64],
    multi_idx: npt.NDArray[np.uint64],
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
    del payload_codec
    rg = frame // rg_pats
    ps = rg * rg_pats
    pe = min(num_data, (rg + 1) * rg_pats)

    if col_info is not None and _accel_enabled():
        k_o = int(ones_idx[frame + 1]) - int(ones_idx[frame])
        k_m = int(multi_idx[frame + 1]) - int(multi_idx[frame])
        po = np.empty(k_o, dtype=np.uint32)
        pm = np.empty(k_m, dtype=np.uint32)
        cm = np.empty(k_m, dtype=np.uint32)
        if k_o == 0 and k_m == 0:
            return po, pm, cm.view(np.int32)

        mm_obj = cache.get("mm") if cache is not None else None
        fp_local = None
        mm_local = None
        try:
            if mm_obj is None:
                assert path is not None
                fp_local = path.open("rb")
                mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                mm_obj = mm_local

            ones_i64 = (
                ones_idx.view(np.int64)
                if ones_idx.dtype == np.uint64
                else ones_idx.astype(np.int64, copy=False)
            )
            multi_i64 = (
                multi_idx.view(np.int64)
                if multi_idx.dtype == np.uint64
                else multi_idx.astype(np.int64, copy=False)
            )
            _parquet_rs.decode_single_frame(
                mm_obj,
                col_info,
                rg,
                ps,
                pe,
                frame,
                ones_i64,
                multi_i64,
                po,
                pm,
                cm,
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

    rg_base_o = int(ones_idx[ps])
    rg_base_m = int(multi_idx[ps])

    cached_vecs = None
    if cache is not None:
        lru = cache.get("lru")
        if lru is not None and rg in lru:
            cached_vecs = lru.pop(rg)
            lru[rg] = cached_vecs

    if cached_vecs is None:
        rt = pf.read_row_group(rg, columns=_PAYLOAD_COLUMNS, use_threads=False)
        u8_po = _extract_chunk_val_u8(rt.column("place_ones"))
        u8_pm = _extract_chunk_val_u8(rt.column("place_multi"))
        u8_cm = _extract_chunk_val_u8(rt.column("count_multi"))
        cached_vecs = (u8_po, u8_pm, u8_cm)
        if cache is not None:
            lru = cache.setdefault("lru", {})
            lru[rg] = cached_vecs
            if len(lru) > 2:
                lru.pop(next(iter(lru)))
    else:
        u8_po, u8_pm, u8_cm = cached_vecs

    s_o = int(ones_idx[frame]) - rg_base_o
    e_o = int(ones_idx[frame + 1]) - rg_base_o
    k_o = e_o - s_o
    if k_o > 0:
        b_o = s_o * 4
        if is_shuffled:
            po = (
                u8_po[b_o : b_o + 4 * k_o]
                .reshape(4, k_o)
                .T.copy()
                .view("<u4")
                .reshape(-1)
            )
        else:
            po = u8_po[b_o : b_o + 4 * k_o].view("<u4").copy()
        if is_delta:
            np.cumsum(po, dtype=np.uint32, out=po)
    else:
        po = np.empty(0, dtype=np.uint32)

    s_m = int(multi_idx[frame]) - rg_base_m
    e_m = int(multi_idx[frame + 1]) - rg_base_m
    k_m = e_m - s_m
    if k_m > 0:
        b_m = s_m * 4
        if is_shuffled:
            pm = (
                u8_pm[b_m : b_m + 4 * k_m]
                .reshape(4, k_m)
                .T.copy()
                .view("<u4")
                .reshape(-1)
            )
            cm = (
                u8_cm[b_m : b_m + 4 * k_m]
                .reshape(4, k_m)
                .T.copy()
                .view("<i4")
                .reshape(-1)
            )
        else:
            pm = u8_pm[b_m : b_m + 4 * k_m].view("<u4").copy()
            cm = u8_cm[b_m : b_m + 4 * k_m].view("<i4").copy()
        if is_delta:
            np.cumsum(pm, dtype=np.uint32, out=pm)
    else:
        pm = np.empty(0, dtype=np.uint32)
        cm = np.empty(0, dtype=np.int32)

    return po, pm, cm


def read_parquet_patterns(
    path: Path,
    index_ranges: npt.NDArray[np.integer[Any]],
    ones_idx: npt.NDArray[np.uint64],
    multi_idx: npt.NDArray[np.uint64],
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
    ranges = np.asarray(index_ranges)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("Pattern ranges must have shape (N, 2)")
    if (
        np.any(ranges < 0)
        or np.any(ranges[:, 1] > num_data)
        or np.any(ranges[:, 0] > ranges[:, 1])
    ):
        raise IndexError("Pattern range is out of bounds")

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
            return read_single_frame_parquet(
                pf,
                int(ranges[0, 0]),
                ones_idx,
                multi_idx,
                path=path,
                col_info=col_info,
                num_data=num_data,
                rg_pats=rg_pats,
                is_delta=is_delta,
                is_shuffled=shuffle,
                payload_codec=payload_codec,
                cache=cache,
            )

    if is_full:
        workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    else:
        workers = env_workers("EMCFILE_PARQUET_INDEXED_WORKERS", 4, allow_zero=True)

    rgs, firsts, lasts = _split_ranges_by_rg(ranges, rg_pats)
    if len(rgs) == 0:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int32),
        )

    ones_idx_i64 = (
        ones_idx.view(np.int64)
        if ones_idx.dtype == np.uint64
        else ones_idx.astype(np.int64, copy=False)
    )
    multi_idx_i64 = (
        multi_idx.view(np.int64)
        if multi_idx.dtype == np.uint64
        else multi_idx.astype(np.int64, copy=False)
    )

    ones_lens = ones_idx_i64[lasts] - ones_idx_i64[firsts]
    multi_lens = multi_idx_i64[lasts] - multi_idx_i64[firsts]

    dst_ones = np.empty(len(firsts) + 1, dtype=np.int64)
    dst_ones[0] = 0
    np.cumsum(ones_lens, out=dst_ones[1:])

    dst_multi = np.empty(len(firsts) + 1, dtype=np.int64)
    dst_multi[0] = 0
    np.cumsum(multi_lens, out=dst_multi[1:])

    total_ones = int(dst_ones[-1])
    total_multi = int(dst_multi[-1])

    out_po = np.empty(total_ones, dtype=np.uint32)
    out_pm = np.empty(total_multi, dtype=np.uint32)
    out_cm = np.empty(total_multi, dtype=np.uint32)

    order = (
        np.arange(len(rgs), dtype=np.int64) if is_full else np.lexsort((firsts, rgs))
    )
    sorted_rgs = rgs[order]
    unique_rgs, split_pts = np.unique(sorted_rgs, return_index=True)

    use_rs = workers > 0 and _accel_enabled()
    if col_info is not None and use_rs:
        group_starts = np.ascontiguousarray(split_pts, dtype=np.int64)
        group_stops = np.empty_like(group_starts)
        group_stops[:-1] = group_starts[1:]
        group_stops[-1] = len(order)
        group_span_ids = np.ascontiguousarray(order, dtype=np.int64)
        unique_rgs_i64 = np.ascontiguousarray(unique_rgs, dtype=np.int64)
        n_groups = len(unique_rgs_i64)
        n_workers = max(1, min(workers, n_groups))

        mm_obj = cache.get("mm") if cache is not None else None
        fp_local = None
        mm_local = None
        try:
            if mm_obj is None:
                fp_local = path.open("rb")
                mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                mm_obj = mm_local

            _parquet_rs.decode_rg_batch(
                mm_obj,
                col_info,
                unique_rgs_i64,
                group_starts,
                group_stops,
                group_span_ids,
                rg_pats,
                num_data,
                ones_idx_i64,
                multi_idx_i64,
                firsts,
                lasts,
                dst_ones,
                dst_multi,
                out_po,
                out_pm,
                out_cm,
                is_delta,
                shuffle,
                n_workers,
            )
        finally:
            if mm_local is not None:
                mm_local.close()
            if fp_local is not None:
                fp_local.close()
        return out_po, out_pm, out_cm.view(np.int32)

    _, pq = _require_pyarrow()
    if pf is None:
        pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)

    groups = np.split(order, split_pts[1:])

    def proc_group(g_idx: int) -> None:
        rg = int(unique_rgs[g_idx])
        span_ids = groups[g_idx]
        p_firsts = firsts[span_ids]
        p_lasts = lasts[span_ids]

        ps = rg * rg_pats
        pe = min(num_data, (rg + 1) * rg_pats)
        rg_base_o = int(ones_idx_i64[ps])
        rg_no = int(ones_idx_i64[pe]) - rg_base_o
        rg_base_m = int(multi_idx_i64[ps])
        rg_nm = int(multi_idx_i64[pe]) - rg_base_m

        rt = pf.read_row_group(rg, columns=_PAYLOAD_COLUMNS, use_threads=False)

        if rg_no > 0:
            u8_po = _extract_chunk_val_u8(rt.column("place_ones"))
            _decode_rg_spans_numpy(
                u8_po,
                rg_base_o,
                ones_idx_i64,
                out_po,
                p_firsts,
                p_lasts,
                dst_ones[span_ids],
                is_delta=is_delta,
                is_shuffled=shuffle,
            )

        if rg_nm > 0:
            u8_pm = _extract_chunk_val_u8(rt.column("place_multi"))
            u8_cm = _extract_chunk_val_u8(rt.column("count_multi"))
            _decode_rg_spans_numpy(
                u8_pm,
                rg_base_m,
                multi_idx_i64,
                out_pm,
                p_firsts,
                p_lasts,
                dst_multi[span_ids],
                is_delta=is_delta,
                is_shuffled=shuffle,
            )
            _decode_rg_spans_numpy(
                u8_cm,
                rg_base_m,
                multi_idx_i64,
                out_cm,
                p_firsts,
                p_lasts,
                dst_multi[span_ids],
                is_delta=False,
                is_shuffled=shuffle,
            )

    n_groups = len(unique_rgs)
    if workers > 1 and n_groups > 1:
        pool = _get_worker_pool(min(workers, n_groups))
        list(pool.map(proc_group, range(n_groups)))
    else:
        for g_idx in range(n_groups):
            proc_group(g_idx)

    return out_po, out_pm, out_cm.view(np.int32)


def read_parquet_column(
    path: Path,
    column: str,
    offsets: npt.NDArray[np.uint64],
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
    del payload_codec
    total_elems = int(offsets[-1]) if offsets.size > 0 else 0
    if num_data == 0 or total_elems == 0:
        return np.empty(0, dtype=dtype)

    workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    use_rs = workers > 0 and _accel_enabled()
    _, col_info = _get_file_info(path, cache=cache)

    offsets_i64 = (
        offsets.view(np.int64)
        if offsets.dtype == np.uint64
        else offsets.astype(np.int64, copy=False)
    )
    out_u32 = np.empty(total_elems, dtype=np.uint32)
    n_rgs = (num_data + rg_pats - 1) // rg_pats

    col_idx_map = {"place_ones": 2, "place_multi": 3, "count_multi": 4}
    if col_info is not None and column in col_idx_map and use_rs:
        c_idx = col_idx_map[column]
        n_workers = max(1, min(workers, n_rgs))
        mm_obj = cache.get("mm") if cache is not None else None
        fp_local = None
        mm_local = None
        try:
            if mm_obj is None:
                fp_local = path.open("rb")
                mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                mm_obj = mm_local
            _parquet_rs.decode_col(
                mm_obj,
                col_info,
                c_idx,
                n_rgs,
                rg_pats,
                num_data,
                offsets_i64,
                out_u32,
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

    def proc_rg(rg: int) -> None:
        ps = rg * rg_pats
        pe = min(num_data, (rg + 1) * rg_pats)
        rg_base = int(offsets_i64[ps])
        rg_n = int(offsets_i64[pe]) - rg_base
        if rg_n <= 0:
            return
        rt = pf_main.read_row_group(rg, columns=[column], use_threads=False)
        u8 = _extract_chunk_val_u8(rt.column(column))
        p_first = np.array([ps], dtype=np.int64)
        p_last = np.array([pe], dtype=np.int64)
        dst_off = np.array([rg_base], dtype=np.int64)
        _decode_rg_spans_numpy(
            u8,
            rg_base,
            offsets_i64,
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
