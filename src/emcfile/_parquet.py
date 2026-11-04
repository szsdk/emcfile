"""Parquet 1-row-per-pattern reader and parallel shuffle+Zstd writer/reader fast paths."""

from __future__ import annotations

import importlib.util
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

_kernels_lock = threading.Lock()
_encode_page_kernel: Any = None
_encode_raw_kernel: Any = None
_decode_spans_kernel: Any = None

_template_lock = threading.Lock()
_zstd_footer_template: tuple[bytes, bytes] | None = None


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(PARQUET_EXTRA_MESSAGE) from exc
    return pa, pq


def available() -> bool:
    """Whether optional Numba + Zstandard acceleration is available."""
    return sys.byteorder == "little" and all(
        importlib.util.find_spec(name) is not None
        for name in ("pyarrow", "numba", "zstandard")
    )


def _ensure_encode_kernels() -> tuple[Any, Any]:
    global _encode_page_kernel, _encode_raw_kernel
    if _encode_page_kernel is None or _encode_raw_kernel is None:
        with _kernels_lock:
            if _encode_page_kernel is None or _encode_raw_kernel is None:
                from numba import njit

                @njit(nogil=True)
                def _shuffle_4(
                    sub_src: np.ndarray,
                    k: int,
                    b0: np.ndarray,
                    b1: np.ndarray,
                    b2: np.ndarray,
                    b3: np.ndarray,
                    is_delta: bool,
                ) -> None:
                    if is_delta:
                        v0 = sub_src[0]
                        b0[0] = v0 & np.uint32(0xFF)
                        b1[0] = (v0 >> np.uint32(8)) & np.uint32(0xFF)
                        b2[0] = (v0 >> np.uint32(16)) & np.uint32(0xFF)
                        b3[0] = (v0 >> np.uint32(24)) & np.uint32(0xFF)
                        for i in range(1, k):
                            val = np.uint32(sub_src[i] - sub_src[i - 1])
                            b0[i] = val & np.uint32(0xFF)
                            b1[i] = (val >> np.uint32(8)) & np.uint32(0xFF)
                            b2[i] = (val >> np.uint32(16)) & np.uint32(0xFF)
                            b3[i] = (val >> np.uint32(24)) & np.uint32(0xFF)
                    else:
                        for i in range(k):
                            val = sub_src[i]
                            b0[i] = val & np.uint32(0xFF)
                            b1[i] = (val >> np.uint32(8)) & np.uint32(0xFF)
                            b2[i] = (val >> np.uint32(16)) & np.uint32(0xFF)
                            b3[i] = (val >> np.uint32(24)) & np.uint32(0xFF)

                @njit(nogil=True)
                def _validate_pattern(
                    src_i32: np.ndarray,
                    s: int,
                    e: int,
                    is_delta: bool,
                    check_sorted: bool,
                ) -> int:
                    if s >= e:
                        return 0
                    prev = np.int64(src_i32[s])
                    if is_delta and prev < 0:
                        return -1
                    if check_sorted:
                        for idx in range(s + 1, e):
                            cur = np.int64(src_i32[idx])
                            if is_delta and cur < 0:
                                return -1
                            if cur < prev:
                                return -2
                            prev = cur
                    elif is_delta:
                        for idx in range(s + 1, e):
                            if src_i32[idx] < 0:
                                return -1
                    return 0

                @njit(nogil=True)
                def page_kernel(
                    src_i32: np.ndarray,
                    pat_offsets: np.ndarray,
                    p_start: int,
                    p_end: int,
                    out_u8: np.ndarray,
                    is_delta: bool,
                    is_shuffled: bool,
                    check_sorted: bool,
                ) -> int:
                    src_u32 = src_i32.view(np.uint32)
                    base_elem = pat_offsets[p_start]
                    for r in range(p_end - p_start):
                        p = p_start + r
                        s = pat_offsets[p]
                        e = pat_offsets[p + 1]
                        if check_sorted:
                            rc = _validate_pattern(src_i32, s, e, is_delta, True)
                            if rc != 0:
                                return rc
                        elif is_delta and s < e and src_i32[s] < 0:
                            return -1
                        k = e - s
                        hdr_off = (s - base_elem + r) * 4
                        byte_len = np.uint32(k * 4)
                        out_u8[hdr_off] = byte_len & np.uint32(0xFF)
                        out_u8[hdr_off + 1] = (byte_len >> np.uint32(8)) & np.uint32(
                            0xFF
                        )
                        out_u8[hdr_off + 2] = (byte_len >> np.uint32(16)) & np.uint32(
                            0xFF
                        )
                        out_u8[hdr_off + 3] = (byte_len >> np.uint32(24)) & np.uint32(
                            0xFF
                        )
                        if k <= 0:
                            continue
                        o = hdr_off + 4
                        if is_shuffled:
                            _shuffle_4(
                                src_u32[s:e],
                                k,
                                out_u8[o : o + k],
                                out_u8[o + k : o + 2 * k],
                                out_u8[o + 2 * k : o + 3 * k],
                                out_u8[o + 3 * k : o + 4 * k],
                                is_delta,
                            )
                        else:
                            out_u32 = out_u8[o : o + 4 * k].view(np.uint32)
                            if is_delta:
                                out_u32[0] = src_u32[s]
                                for i in range(1, k):
                                    out_u32[i] = np.uint32(
                                        src_u32[s + i] - src_u32[s + i - 1]
                                    )
                            else:
                                for i in range(k):
                                    out_u32[i] = src_u32[s + i]
                    return 0

                @njit(nogil=True)
                def raw_kernel(
                    src_i32: np.ndarray,
                    pat_offsets: np.ndarray,
                    p_start: int,
                    p_end: int,
                    out_u8: np.ndarray,
                    is_delta: bool,
                    is_shuffled: bool,
                    check_sorted: bool,
                ) -> int:
                    src_u32 = src_i32.view(np.uint32)
                    base_elem = pat_offsets[p_start]
                    for r in range(p_end - p_start):
                        p = p_start + r
                        s = pat_offsets[p]
                        e = pat_offsets[p + 1]
                        if check_sorted:
                            rc = _validate_pattern(src_i32, s, e, is_delta, True)
                            if rc != 0:
                                return rc
                        elif is_delta and s < e and src_i32[s] < 0:
                            return -1
                        k = e - s
                        if k <= 0:
                            continue
                        o = (s - base_elem) * 4
                        if is_shuffled:
                            _shuffle_4(
                                src_u32[s:e],
                                k,
                                out_u8[o : o + k],
                                out_u8[o + k : o + 2 * k],
                                out_u8[o + 2 * k : o + 3 * k],
                                out_u8[o + 3 * k : o + 4 * k],
                                is_delta,
                            )
                        else:
                            out_u32 = out_u8[o : o + 4 * k].view(np.uint32)
                            if is_delta:
                                out_u32[0] = src_u32[s]
                                for i in range(1, k):
                                    out_u32[i] = np.uint32(
                                        src_u32[s + i] - src_u32[s + i - 1]
                                    )
                            else:
                                for i in range(k):
                                    out_u32[i] = src_u32[s + i]
                    return 0

                dummy_src = np.zeros(2, dtype=np.int32)
                dummy_offs = np.array([0, 2], dtype=np.int64)
                dummy_page = np.zeros(12, dtype=np.uint8)
                dummy_raw = np.zeros(8, dtype=np.uint8)
                page_kernel(dummy_src, dummy_offs, 0, 1, dummy_page, True, True, False)
                raw_kernel(dummy_src, dummy_offs, 0, 1, dummy_raw, True, True, False)
                _encode_page_kernel = page_kernel
                _encode_raw_kernel = raw_kernel
    return _encode_page_kernel, _encode_raw_kernel


def _ensure_decode_kernel() -> Any:
    global _decode_spans_kernel
    if _decode_spans_kernel is None:
        with _kernels_lock:
            if _decode_spans_kernel is None:
                from numba import njit

                @njit(nogil=True)
                def _unshuffle_two_pass(
                    b0: np.ndarray,
                    b1: np.ndarray,
                    b2: np.ndarray,
                    b3: np.ndarray,
                    k: int,
                    sub_out: np.ndarray,
                    is_delta: bool,
                ) -> None:
                    for i in range(k):
                        sub_out[i] = (
                            np.uint32(b0[i])
                            | (np.uint32(b1[i]) << np.uint32(8))
                            | (np.uint32(b2[i]) << np.uint32(16))
                            | (np.uint32(b3[i]) << np.uint32(24))
                        )
                    if is_delta:
                        acc = np.uint32(0)
                        for i in range(k):
                            acc = np.uint32(acc + sub_out[i])
                            sub_out[i] = acc

                @njit(nogil=True)
                def kernel(
                    val_u8: np.ndarray,
                    rg_base_elem: int,
                    pat_offsets: np.ndarray,
                    out_u32: np.ndarray,
                    pat_firsts: np.ndarray,
                    pat_lasts: np.ndarray,
                    dst_offsets: np.ndarray,
                    is_delta: bool,
                    is_shuffled: bool,
                ) -> None:
                    n_spans = len(pat_firsts)
                    for s_idx in range(n_spans):
                        p_start = pat_firsts[s_idx]
                        p_end = pat_lasts[s_idx]
                        out_pos = dst_offsets[s_idx]
                        for p in range(p_start, p_end):
                            s = pat_offsets[p]
                            e = pat_offsets[p + 1]
                            k = e - s
                            if k <= 0:
                                continue
                            b_s = (s - rg_base_elem) * 4
                            sub_out = out_u32[out_pos : out_pos + k]
                            if is_shuffled:
                                _unshuffle_two_pass(
                                    val_u8[b_s : b_s + k],
                                    val_u8[b_s + k : b_s + 2 * k],
                                    val_u8[b_s + 2 * k : b_s + 3 * k],
                                    val_u8[b_s + 3 * k : b_s + 4 * k],
                                    k,
                                    sub_out,
                                    is_delta,
                                )
                            else:
                                src_u32 = val_u8[b_s : b_s + 4 * k].view(np.uint32)
                                if is_delta:
                                    acc = np.uint32(0)
                                    for i in range(k):
                                        acc = np.uint32(acc + src_u32[i])
                                        sub_out[i] = acc
                                else:
                                    for i in range(k):
                                        sub_out[i] = src_u32[i]
                            out_pos += k

                dummy_in = np.zeros(8, dtype=np.uint8)
                dummy_offs = np.array([0, 2], dtype=np.int64)
                dummy_out = np.zeros(2, dtype=np.uint32)
                dummy_p = np.array([0], dtype=np.int64)
                dummy_q = np.array([1], dtype=np.int64)
                dummy_d = np.array([0], dtype=np.int64)
                kernel(
                    dummy_in,
                    0,
                    dummy_offs,
                    dummy_out,
                    dummy_p,
                    dummy_q,
                    dummy_d,
                    True,
                    True,
                )
                kernel(
                    dummy_in,
                    0,
                    dummy_offs,
                    dummy_out,
                    dummy_p,
                    dummy_q,
                    dummy_d,
                    False,
                    True,
                )
                _decode_spans_kernel = kernel
    return _decode_spans_kernel


def _encode_rg_numpy(
    src_arr: np.ndarray,
    pat_offsets: np.ndarray,
    p_start: int,
    p_end: int,
    *,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
) -> np.ndarray:
    s0 = int(pat_offsets[p_start])
    e0 = int(pat_offsets[p_end])
    n_elems = e0 - s0
    if n_elems <= 0:
        return np.empty(0, dtype=np.uint8)
    counts = pat_offsets[p_start + 1 : p_end + 1] - pat_offsets[p_start:p_end]
    sub_src = np.ascontiguousarray(src_arr[s0:e0])
    if is_delta:
        encoded = encode_pattern_local_delta(
            sub_src.astype(np.uint32, copy=False),
            counts,
            check_sorted=check_sorted,
        )
    else:
        encoded = sub_src.view(np.uint32)
    if not is_shuffled:
        return np.ascontiguousarray(encoded).view(np.uint8)
    out_u8 = np.empty(n_elems * 4, dtype=np.uint8)
    local_offs = pat_offsets[p_start : p_end + 1] - s0
    for r in range(p_end - p_start):
        ls = int(local_offs[r])
        le = int(local_offs[r + 1])
        k = le - ls
        if k <= 0:
            continue
        pat_u8 = encoded[ls:le].view(np.uint8).reshape(k, 4).T.reshape(-1)
        out_u8[ls * 4 : le * 4] = pat_u8
    return out_u8


def _decode_rg_spans_numpy(
    val_u8: np.ndarray,
    rg_base_elem: int,
    pat_offsets: np.ndarray,
    out_u32: np.ndarray,
    pat_firsts: np.ndarray,
    pat_lasts: np.ndarray,
    dst_offsets: np.ndarray,
    *,
    is_delta: bool,
    is_shuffled: bool,
) -> None:
    for s_idx in range(len(pat_firsts)):
        p_start = int(pat_firsts[s_idx])
        p_end = int(pat_lasts[s_idx])
        out_pos = int(dst_offsets[s_idx])
        for p in range(p_start, p_end):
            s = int(pat_offsets[p])
            e = int(pat_offsets[p + 1])
            k = e - s
            if k <= 0:
                continue
            b_s = (s - rg_base_elem) * 4
            if is_shuffled:
                vals = (
                    val_u8[b_s : b_s + 4 * k]
                    .reshape(4, k)
                    .T.copy()
                    .view("<u4")
                    .reshape(-1)
                )
            else:
                vals = val_u8[b_s : b_s + 4 * k].view("<u4").copy()
            if is_delta:
                np.cumsum(vals, dtype=np.uint32, out=vals)
            out_u32[out_pos : out_pos + k] = vals
            out_pos += k


def is_parquet_file(path: Path) -> bool:
    """Return whether *path* is a Parquet file by suffix or magic header."""
    if path.suffix.lower() in PARQUET_SUFFIXES:
        return True
    if not path.is_file():
        return False
    try:
        with path.open("rb") as fp:
            return fp.read(4) == _PARQUET_MAGIC
    except OSError:
        return False


def _normalize_parquet_compression(
    compression: str | int | None,
    compression_opts: Any,
) -> tuple[str, int | None]:
    if compression is None or compression == "none":
        return "NONE", None
    if not isinstance(compression, str):
        raise ValueError(f"Unsupported Parquet compression: {compression!r}")
    key = compression.strip().lower()
    if key == "none":
        return "NONE", None
    if key == "zstd":
        level = 1 if compression_opts is None else int(compression_opts)
        return "ZSTD", level
    if key == "gzip":
        level = 4 if compression_opts is None else int(compression_opts)
        return "GZIP", level
    if key == "lz4":
        return "LZ4", None
    if key == "snappy":
        return "SNAPPY", None
    raise ValueError(
        f"Unsupported Parquet compression {compression!r}. "
        "Supported codecs: None, 'zstd', 'lz4', 'snappy', 'gzip'."
    )


def _enc_uvarint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _enc_zigzag(n: int) -> bytes:
    return _enc_uvarint((n << 1) ^ (n >> 63))


def _make_data_page_header(uncomp_len: int, comp_len: int, num_values: int) -> bytes:
    return (
        b"\x15\x00"
        + b"\x15"
        + _enc_zigzag(uncomp_len)
        + b"\x15"
        + _enc_zigzag(comp_len)
        + b"\x2c\x15"
        + _enc_zigzag(num_values)
        + b"\x15\x00\x15\x06\x15\x06\x1c\x00\x00\x00"
    )


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


def _split_ranges_by_rg(
    ranges: npt.NDArray[np.integer[Any]], rg_pats: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    out_rgs: list[int] = []
    out_firsts: list[int] = []
    out_lasts: list[int] = []
    for start, end in ranges:
        s = int(start)
        e = int(end)
        if e <= s:
            continue
        cur = s
        while cur < e:
            rg = cur // rg_pats
            rg_end = (rg + 1) * rg_pats
            nxt = e if e < rg_end else rg_end
            out_rgs.append(rg)
            out_firsts.append(cur)
            out_lasts.append(nxt)
            cur = nxt
    return (
        np.asarray(out_rgs, dtype=np.int64),
        np.asarray(out_firsts, dtype=np.int64),
        np.asarray(out_lasts, dtype=np.int64),
    )


def _extract_chunk_val_u8(col_chunked: Any) -> npt.NDArray[np.uint8]:
    if len(col_chunked) == 0:
        return np.empty(0, dtype=np.uint8)
    arr = (
        col_chunked.chunk(0)
        if col_chunked.num_chunks == 1
        else col_chunked.combine_chunks()
    )
    bufs = arr.buffers()
    off_arr = np.frombuffer(
        bufs[1], dtype=np.int32, count=len(arr) + 1, offset=arr.offset * 4
    )
    b0 = int(off_arr[0])
    b_end = int(off_arr[-1])
    if b_end <= b0:
        return np.empty(0, dtype=np.uint8)
    return np.frombuffer(bufs[2], dtype=np.uint8, count=b_end - b0, offset=b0)


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
        chunk_bytes = max(4 << 20, ((total_bytes // 2048) + 3) & ~3)

    bytes_per_pat = max(4, total_bytes // max(1, num_data))
    rg_pats = max(1, min(max(1, num_data), chunk_bytes // bytes_per_pat))
    workers = env_workers("EMCFILE_PARQUET_WRITE_WORKERS", 4, allow_zero=True)
    use_numba = (
        num_data > 0
        and workers > 0
        and sys.byteorder == "little"
        and importlib.util.find_spec("numba") is not None
    )
    has_zstd = importlib.util.find_spec("zstandard") is not None
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

    if pq_codec == "ZSTD" and use_numba and has_zstd and num_data > 0:
        import zstandard

        page_kernel, _ = _ensure_encode_kernels()
        zstd_level = 1 if pq_level is None else int(pq_level)
        tls = threading.local()

        def encode_batch_zstd(batch: PatternsSOne) -> list[Any]:
            b_n = int(batch.num_data)
            b_rgs = (b_n + rg_pats - 1) // rg_pats
            ones_arr = np.ascontiguousarray(batch.ones, dtype=np.uint32)
            multi_arr = np.ascontiguousarray(batch.multi, dtype=np.uint32)
            ones_idx = batch.ones_idx.astype(np.int64, copy=False)
            multi_idx = batch.multi_idx.astype(np.int64, copy=False)
            po_src = np.ascontiguousarray(batch.place_ones).view(np.int32)
            pm_src = np.ascontiguousarray(batch.place_multi).view(np.int32)
            cm_src = np.ascontiguousarray(batch.count_multi).view(np.int32)

            def proc_rg_zstd(rg: int) -> Any:
                compressor = getattr(tls, "compressor", None)
                if compressor is None:
                    compressor = zstandard.ZstdCompressor(level=zstd_level)
                    tls.compressor = compressor

                ps = rg * rg_pats
                pe = min(b_n, (rg + 1) * rg_pats)
                nr = pe - ps
                no = int(ones_idx[pe] - ones_idx[ps])
                nm = int(multi_idx[pe] - multi_idx[ps])

                u_ones = nr * 4
                u_multi = nr * 4
                u_po = (no + nr) * 4
                u_pm = (nm + nr) * 4
                u_cm = (nm + nr) * 4

                buf = getattr(tls, "buf", None)
                need = max(u_po, u_pm)
                if buf is None or len(buf) < need:
                    buf = np.empty(max(need, 65536), dtype=np.uint8)
                    tls.buf = buf

                c_ones = compressor.compress(memoryview(ones_arr[ps:pe]).cast("B"))
                c_multi = compressor.compress(memoryview(multi_arr[ps:pe]).cast("B"))

                b_po = buf[:u_po]
                rc = int(
                    page_kernel(
                        po_src,
                        ones_idx,
                        ps,
                        pe,
                        b_po,
                        is_delta,
                        shuffle,
                        check_sorted,
                    )
                )
                if rc == -1:
                    raise ValueError("position values must fit in uint32")
                if rc == -2:
                    raise ValueError(
                        "position values must be sorted within every pattern"
                    )
                c_po = compressor.compress(memoryview(b_po))

                b_pm = buf[:u_pm]
                rc = int(
                    page_kernel(
                        pm_src,
                        multi_idx,
                        ps,
                        pe,
                        b_pm,
                        is_delta,
                        shuffle,
                        check_sorted,
                    )
                )
                if rc == -1:
                    raise ValueError("position values must fit in uint32")
                if rc == -2:
                    raise ValueError(
                        "position values must be sorted within every pattern"
                    )
                c_pm = compressor.compress(memoryview(b_pm))

                b_cm = buf[:u_cm]
                page_kernel(cm_src, multi_idx, ps, pe, b_cm, False, shuffle, False)
                c_cm = compressor.compress(memoryview(b_cm))

                return nr, (
                    (
                        1,
                        b"ones",
                        u_ones,
                        _make_data_page_header(u_ones, len(c_ones), nr),
                        c_ones,
                    ),
                    (
                        1,
                        b"multi",
                        u_multi,
                        _make_data_page_header(u_multi, len(c_multi), nr),
                        c_multi,
                    ),
                    (
                        6,
                        b"place_ones",
                        u_po,
                        _make_data_page_header(u_po, len(c_po), nr),
                        c_po,
                    ),
                    (
                        6,
                        b"place_multi",
                        u_pm,
                        _make_data_page_header(u_pm, len(c_pm), nr),
                        c_pm,
                    ),
                    (
                        6,
                        b"count_multi",
                        u_cm,
                        _make_data_page_header(u_cm, len(c_cm), nr),
                        c_cm,
                    ),
                )

            if workers > 1 and b_rgs > 1:
                with ThreadPoolExecutor(max_workers=min(workers, b_rgs)) as pool:
                    return list(pool.map(proc_rg_zstd, range(b_rgs)))
            return [proc_rg_zstd(rg) for rg in range(b_rgs)]

        rg_bytes = max(1, rg_pats * bytes_per_pat)
        max_rgs_per_batch = max(1, buffer_size // rg_bytes)
        if len(datas) == 1 and isinstance(datas[0], PatternsSOne):
            batches_iter: Iterator[PatternsSOne] = iter([datas[0]])
        else:
            batches_iter = _iter_rg_batches(datas, rg_pats, max_rgs_per_batch)

        schema_prefix, created_by_suffix = _get_zstd_footer_template()
        kv_bytes = _encode_kv_metadata(meta_kv)
        suffix_bytes = kv_bytes + b"\x18" + created_by_suffix[1:]

        col_metas_all: list[bytes] = []
        total_rgs = 0
        file_pos = 4
        with path.open("wb") as fp:
            fp.write(_PARQUET_MAGIC)
            for batch in batches_iter:
                for nr, cols in encode_batch_zstd(batch):
                    total_rgs += 1
                    rg_start_pos = file_pos
                    total_uncomp_rg = 0
                    total_comp_rg = 0
                    col_metas = [b"\x19\x5c"]
                    for c_type, c_name, u_len, hdr, comp_data in cols:
                        page_off = file_pos
                        fp.write(hdr)
                        fp.write(comp_data)
                        tc = len(hdr) + len(comp_data)
                        tu = len(hdr) + u_len
                        file_pos += tc
                        total_uncomp_rg += tu
                        total_comp_rg += tc
                        col_metas.append(
                            _make_col_chunk_meta(
                                c_type, c_name, 6, nr, tu, tc, page_off
                            )
                        )
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
            footer_bytes = b"".join(
                [
                    schema_prefix,
                    b"\x16",
                    _enc_zigzag(num_data),
                    list_hdr,
                    *col_metas_all,
                    suffix_bytes,
                ]
            )
            fp.write(footer_bytes)
            fp.write(len(footer_bytes).to_bytes(4, "little"))
            fp.write(_PARQUET_MAGIC)
        return

    _, raw_kernel = _ensure_encode_kernels() if use_numba else (None, None)
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
        tbl = pa.Table.from_pydict(
            {
                "ones": pa.array([], type=pa.uint32()),
                "multi": pa.array([], type=pa.uint32()),
                "place_ones": pa.array([], type=pa.binary()),
                "place_multi": pa.array([], type=pa.binary()),
                "count_multi": pa.array([], type=pa.binary()),
            },
            schema=schema,
        )
        pq.write_table(
            tbl,
            path,
            compression=pq_codec,
            compression_level=pq_level,
            use_dictionary=False,
            write_statistics=False,
        )
        return

    def encode_batch_arrow(batch: PatternsSOne) -> Any:
        b_n = int(batch.num_data)
        b_rgs = (b_n + rg_pats - 1) // rg_pats
        ones_arr = np.ascontiguousarray(batch.ones, dtype=np.uint32)
        multi_arr = np.ascontiguousarray(batch.multi, dtype=np.uint32)
        ones_idx = batch.ones_idx.astype(np.int64, copy=False)
        multi_idx = batch.multi_idx.astype(np.int64, copy=False)
        po_src = np.ascontiguousarray(batch.place_ones).view(np.int32)
        pm_src = np.ascontiguousarray(batch.place_multi).view(np.int32)
        cm_src = np.ascontiguousarray(batch.count_multi).view(np.int32)

        def build_rg_batch(rg: int) -> Any:
            ps = rg * rg_pats
            pe = min(b_n, (rg + 1) * rg_pats)
            nr = pe - ps
            no = int(ones_idx[pe] - ones_idx[ps])
            nm = int(multi_idx[pe] - multi_idx[ps])

            if raw_kernel is not None:
                po_u8 = np.empty(no * 4, dtype=np.uint8)
                pm_u8 = np.empty(nm * 4, dtype=np.uint8)
                cm_u8 = np.empty(nm * 4, dtype=np.uint8)
                rc = int(
                    raw_kernel(
                        po_src,
                        ones_idx,
                        ps,
                        pe,
                        po_u8,
                        is_delta,
                        shuffle,
                        check_sorted,
                    )
                )
                if rc == -1:
                    raise ValueError("position values must fit in uint32")
                if rc == -2:
                    raise ValueError(
                        "position values must be sorted within every pattern"
                    )
                rc = int(
                    raw_kernel(
                        pm_src,
                        multi_idx,
                        ps,
                        pe,
                        pm_u8,
                        is_delta,
                        shuffle,
                        check_sorted,
                    )
                )
                if rc == -1:
                    raise ValueError("position values must fit in uint32")
                if rc == -2:
                    raise ValueError(
                        "position values must be sorted within every pattern"
                    )
                raw_kernel(cm_src, multi_idx, ps, pe, cm_u8, False, shuffle, False)
            else:
                po_u8 = _encode_rg_numpy(
                    batch.place_ones,
                    ones_idx,
                    ps,
                    pe,
                    is_delta=is_delta,
                    is_shuffled=shuffle,
                    check_sorted=check_sorted,
                )
                pm_u8 = _encode_rg_numpy(
                    batch.place_multi,
                    multi_idx,
                    ps,
                    pe,
                    is_delta=is_delta,
                    is_shuffled=shuffle,
                    check_sorted=check_sorted,
                )
                cm_u8 = _encode_rg_numpy(
                    batch.count_multi,
                    multi_idx,
                    ps,
                    pe,
                    is_delta=False,
                    is_shuffled=shuffle,
                    check_sorted=False,
                )

            po_off = ((ones_idx[ps : pe + 1] - ones_idx[ps]) * 4).astype(np.int32)
            pm_off = ((multi_idx[ps : pe + 1] - multi_idx[ps]) * 4).astype(np.int32)

            a_ones = pa.array(ones_arr[ps:pe], type=pa.uint32())
            a_multi = pa.array(multi_arr[ps:pe], type=pa.uint32())
            a_po = pa.BinaryArray.from_buffers(
                pa.binary(), nr, [None, pa.py_buffer(po_off), pa.py_buffer(po_u8)]
            )
            a_pm = pa.BinaryArray.from_buffers(
                pa.binary(), nr, [None, pa.py_buffer(pm_off), pa.py_buffer(pm_u8)]
            )
            a_cm = pa.BinaryArray.from_buffers(
                pa.binary(), nr, [None, pa.py_buffer(pm_off), pa.py_buffer(cm_u8)]
            )
            return pa.RecordBatch.from_arrays(
                [a_ones, a_multi, a_po, a_pm, a_cm], schema=schema
            )

        if workers > 1 and b_rgs > 1:
            with ThreadPoolExecutor(max_workers=min(workers, b_rgs)) as pool:
                batches = list(pool.map(build_rg_batch, range(b_rgs)))
        else:
            batches = [build_rg_batch(rg) for rg in range(b_rgs)]
        return pa.Table.from_batches(batches, schema=schema)

    rg_bytes = max(1, rg_pats * bytes_per_pat)
    max_rgs_per_batch = max(1, buffer_size // rg_bytes)

    if len(datas) == 1 and isinstance(datas[0], PatternsSOne):
        tbl = encode_batch_arrow(datas[0])
        pq.write_table(
            tbl,
            path,
            row_group_size=max(1, rg_pats),
            compression=pq_codec,
            compression_level=pq_level,
            use_dictionary=False,
            write_statistics=False,
            data_page_size=64 * 1024 * 1024,
        )
        return

    with pq.ParquetWriter(
        path,
        schema,
        compression=pq_codec,
        compression_level=pq_level,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=64 * 1024 * 1024,
    ) as writer:
        for batch in _iter_rg_batches(datas, rg_pats, max_rgs_per_batch):
            tbl = encode_batch_arrow(batch)
            writer.write_table(tbl, row_group_size=max(1, rg_pats))


def read_parquet_metadata(path: Path) -> dict[str, Any]:
    """Read header metadata from a Parquet v2 pattern file."""
    _, pq = _require_pyarrow()
    file_meta = pq.read_metadata(path)
    meta = file_meta.metadata or {}
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


def read_parquet_ones_multi(
    path: Path,
    num_data: int,
    *,
    pf: Any | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Read the ``ones`` and ``multi`` count arrays from a Parquet v2 file."""
    if num_data == 0:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)
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
    num_data: int,
    rg_pats: int,
    is_delta: bool,
    is_shuffled: bool,
    payload_codec: str = "none",
    cache: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    """Fast single-frame read using a 2-Row-Group LRU cache and pure NumPy."""
    del num_data, payload_codec
    rg = frame // rg_pats
    ps = rg * rg_pats
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

    _, pq = _require_pyarrow()
    if pf is None:
        pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)

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

    ones_idx_i64 = ones_idx.astype(np.int64, copy=False)
    multi_idx_i64 = multi_idx.astype(np.int64, copy=False)

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

    order = np.argsort(rgs, kind="stable")
    sorted_rgs = rgs[order]
    unique_rgs, split_pts = np.unique(sorted_rgs, return_index=True)
    groups = np.split(order, split_pts[1:])

    use_numba_decode = (
        workers > 0
        and sys.byteorder == "little"
        and importlib.util.find_spec("numba") is not None
    )
    decode_kernel = _ensure_decode_kernel() if use_numba_decode else None
    tls = threading.local()

    def proc_group(g_idx: int) -> None:
        f = pf
        if workers > 1 and len(unique_rgs) > 1:
            f = getattr(tls, "pf", None)
            if f is None:
                f = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
                tls.pf = f

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

        rt = f.read_row_group(rg, columns=_PAYLOAD_COLUMNS, use_threads=False)

        if rg_no > 0:
            u8_po = _extract_chunk_val_u8(rt.column("place_ones"))
            if decode_kernel is not None:
                decode_kernel(
                    u8_po,
                    rg_base_o,
                    ones_idx_i64,
                    out_po,
                    p_firsts,
                    p_lasts,
                    dst_ones[span_ids],
                    is_delta,
                    shuffle,
                )
            else:
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
            if decode_kernel is not None:
                decode_kernel(
                    u8_pm,
                    rg_base_m,
                    multi_idx_i64,
                    out_pm,
                    p_firsts,
                    p_lasts,
                    dst_multi[span_ids],
                    is_delta,
                    shuffle,
                )
                decode_kernel(
                    u8_cm,
                    rg_base_m,
                    multi_idx_i64,
                    out_cm,
                    p_firsts,
                    p_lasts,
                    dst_multi[span_ids],
                    False,
                    shuffle,
                )
            else:
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
        with ThreadPoolExecutor(max_workers=min(workers, n_groups)) as pool:
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
) -> npt.NDArray[Any]:
    """Read and decode a single full event column from a Parquet v2 file."""
    del payload_codec
    total_elems = int(offsets[-1]) if offsets.size > 0 else 0
    if num_data == 0 or total_elems == 0:
        return np.empty(0, dtype=dtype)

    _, pq = _require_pyarrow()
    workers = env_workers("EMCFILE_PARQUET_FULL_SCAN_WORKERS", 4, allow_zero=True)
    use_numba_decode = (
        workers > 0
        and sys.byteorder == "little"
        and importlib.util.find_spec("numba") is not None
    )
    decode_kernel = _ensure_decode_kernel() if use_numba_decode else None

    offsets_i64 = offsets.astype(np.int64, copy=False)
    out_u32 = np.empty(total_elems, dtype=np.uint32)
    n_rgs = (num_data + rg_pats - 1) // rg_pats
    tls = threading.local()
    pf_main = (
        pq.ParquetFile(path, memory_map=True, pre_buffer=False)
        if not (workers > 1 and n_rgs > 1)
        else None
    )

    def proc_rg(rg: int) -> None:
        f = pf_main
        if f is None:
            f = getattr(tls, "pf", None)
            if f is None:
                f = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
                tls.pf = f
        ps = rg * rg_pats
        pe = min(num_data, (rg + 1) * rg_pats)
        rg_base = int(offsets_i64[ps])
        rg_n = int(offsets_i64[pe]) - rg_base
        if rg_n <= 0:
            return
        rt = f.read_row_group(rg, columns=[column], use_threads=False)
        u8 = _extract_chunk_val_u8(rt.column(column))
        p_first = np.array([ps], dtype=np.int64)
        p_last = np.array([pe], dtype=np.int64)
        dst_off = np.array([rg_base], dtype=np.int64)
        if decode_kernel is not None:
            decode_kernel(
                u8,
                rg_base,
                offsets_i64,
                out_u32,
                p_first,
                p_last,
                dst_off,
                is_delta,
                shuffle,
            )
        else:
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
        with ThreadPoolExecutor(max_workers=min(workers, n_rgs)) as pool:
            list(pool.map(proc_rg, range(n_rgs)))
    else:
        for rg in range(n_rgs):
            proc_rg(rg)

    return out_u32.view(dtype)
