"""Parquet 1-row-per-pattern reader and parallel multi-page shuffle+Zstd fast paths."""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import mmap
import os
import queue
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
_zstd_c_kernels: tuple[Any, Any, Any, Any, Any] | None = None
_libzstd_funcs: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None
_libzstd_checked: bool = False

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
    """Whether optional Numba + Zstandard acceleration is available."""
    return sys.byteorder == "little" and all(
        importlib.util.find_spec(name) is not None
        for name in ("pyarrow", "numba", "zstandard")
    )


def _get_libzstd() -> tuple[Any, Any, Any, Any, Any, Any, Any] | None:
    global _libzstd_funcs, _libzstd_checked
    if not _libzstd_checked:
        with _kernels_lock:
            if not _libzstd_checked:
                try:
                    import zstandard as _zstd_mod  # noqa: F401
                except ImportError:
                    pass
                candidates = [
                    ctypes.util.find_library("zstd"),
                    "libzstd.so.1",
                    "libzstd.so",
                    "libzstd.1.dylib",
                    "libzstd.dylib",
                ]
                for cand in candidates:
                    if not cand:
                        continue
                    try:
                        lib = ctypes.CDLL(cand)
                        create_cctx = lib.ZSTD_createCCtx
                        create_cctx.argtypes = []
                        create_cctx.restype = ctypes.c_void_p

                        free_cctx = lib.ZSTD_freeCCtx
                        free_cctx.argtypes = [ctypes.c_void_p]
                        free_cctx.restype = ctypes.c_size_t

                        compress_cctx = lib.ZSTD_compressCCtx
                        compress_cctx.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_size_t,
                            ctypes.c_void_p,
                            ctypes.c_size_t,
                            ctypes.c_int,
                        ]
                        compress_cctx.restype = ctypes.c_size_t

                        compress_bound = lib.ZSTD_compressBound
                        compress_bound.argtypes = [ctypes.c_size_t]
                        compress_bound.restype = ctypes.c_size_t

                        create_dctx = lib.ZSTD_createDCtx
                        create_dctx.argtypes = []
                        create_dctx.restype = ctypes.c_void_p

                        free_dctx = lib.ZSTD_freeDCtx
                        free_dctx.argtypes = [ctypes.c_void_p]
                        free_dctx.restype = ctypes.c_size_t

                        decompress_dctx = lib.ZSTD_decompressDCtx
                        decompress_dctx.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_size_t,
                            ctypes.c_void_p,
                            ctypes.c_size_t,
                        ]
                        decompress_dctx.restype = ctypes.c_size_t

                        _libzstd_funcs = (
                            create_cctx,
                            free_cctx,
                            compress_cctx,
                            compress_bound,
                            create_dctx,
                            free_dctx,
                            decompress_dctx,
                        )
                        break
                    except (OSError, AttributeError):
                        continue
                _libzstd_checked = True
    return _libzstd_funcs


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


def _ensure_zstd_c_kernels() -> tuple[Any, Any, Any, Any, Any] | None:
    global _zstd_c_kernels
    if _zstd_c_kernels is None:
        zstd_funcs = _get_libzstd()
        if zstd_funcs is None:
            return None
        page_kernel, _ = _ensure_encode_kernels()
        with _kernels_lock:
            if _zstd_c_kernels is None:
                from numba import njit

                (
                    _,
                    _,
                    ZSTD_compressCCtx,
                    ZSTD_compressBound,
                    _,
                    _,
                    ZSTD_decompressDCtx,
                ) = zstd_funcs

                @njit(nogil=True)
                def _write_varint(buf: np.ndarray, pos: int, n: int) -> int:
                    u = np.uint64(n)
                    while u >= np.uint64(0x80):
                        buf[pos] = np.uint8((u & np.uint64(0x7F)) | np.uint64(0x80))
                        pos += 1
                        u >>= np.uint64(7)
                    buf[pos] = np.uint8(u)
                    return pos + 1

                @njit(nogil=True)
                def _write_zigzag(buf: np.ndarray, pos: int, n: int) -> int:
                    s = np.int64(n)
                    z = (s << np.int64(1)) ^ (s >> np.int64(63))
                    return _write_varint(buf, pos, z)

                @njit(nogil=True)
                def _write_page_header(
                    buf: np.ndarray,
                    pos: int,
                    uncomp_len: int,
                    comp_len: int,
                    num_values: int,
                ) -> int:
                    buf[pos] = 0x15
                    buf[pos + 1] = 0x00
                    buf[pos + 2] = 0x15
                    pos = _write_zigzag(buf, pos + 3, uncomp_len)
                    buf[pos] = 0x15
                    pos = _write_zigzag(buf, pos + 1, comp_len)
                    buf[pos] = 0x2C
                    buf[pos + 1] = 0x15
                    pos = _write_zigzag(buf, pos + 2, num_values)
                    buf[pos] = 0x15
                    buf[pos + 1] = 0x00
                    buf[pos + 2] = 0x15
                    buf[pos + 3] = 0x06
                    buf[pos + 4] = 0x15
                    buf[pos + 5] = 0x06
                    buf[pos + 6] = 0x1C
                    buf[pos + 7] = 0x00
                    buf[pos + 8] = 0x00
                    buf[pos + 9] = 0x00
                    return pos + 10

                @njit(nogil=True)
                def _encode_col_pages(
                    cctx: int,
                    zstd_level: int,
                    src_i32: np.ndarray,
                    pat_offsets: np.ndarray,
                    ps: int,
                    pe: int,
                    page_pats: int,
                    scratch_u8: np.ndarray,
                    comp_scratch_u8: np.ndarray,
                    out_u8: np.ndarray,
                    out_pos: int,
                    is_delta: bool,
                    is_shuffled: bool,
                    check_sorted: bool,
                ) -> tuple[int, int, int, int]:
                    tu_col = np.int64(0)
                    tc_col = np.int64(0)
                    cur = ps
                    while cur < pe:
                        nxt = cur + page_pats
                        if nxt > pe:
                            nxt = pe
                        nr = nxt - cur
                        ne = pat_offsets[nxt] - pat_offsets[cur]
                        u_len = (ne + nr) * 4
                        rc = page_kernel(
                            src_i32,
                            pat_offsets,
                            cur,
                            nxt,
                            scratch_u8[:u_len],
                            is_delta,
                            is_shuffled,
                            check_sorted,
                        )
                        if rc != 0:
                            return rc, out_pos, tu_col, tc_col
                        bound = ZSTD_compressBound(u_len)
                        c_len = ZSTD_compressCCtx(
                            cctx,
                            comp_scratch_u8.ctypes.data,
                            bound,
                            scratch_u8.ctypes.data,
                            u_len,
                            zstd_level,
                        )
                        hdr_end = _write_page_header(out_u8, out_pos, u_len, c_len, nr)
                        hdr_len = hdr_end - out_pos
                        out_u8[hdr_end : hdr_end + c_len] = comp_scratch_u8[:c_len]
                        out_pos = hdr_end + c_len
                        tu_col += hdr_len + u_len
                        tc_col += hdr_len + c_len
                        cur = nxt
                    return 0, out_pos, tu_col, tc_col

                @njit(nogil=True)
                def encode_rg_all_cols(
                    cctx: int,
                    zstd_level: int,
                    ones_u32: np.ndarray,
                    multi_u32: np.ndarray,
                    po_i32: np.ndarray,
                    pm_i32: np.ndarray,
                    cm_i32: np.ndarray,
                    ones_idx: np.ndarray,
                    multi_idx: np.ndarray,
                    ps: int,
                    pe: int,
                    page_pats: int,
                    scratch_u8: np.ndarray,
                    comp_scratch_u8: np.ndarray,
                    out_u8: np.ndarray,
                    meta_out: np.ndarray,
                    is_delta: bool,
                    is_shuffled: bool,
                    check_sorted: bool,
                ) -> tuple[int, int]:
                    nr = pe - ps
                    u_cnt = nr * 4
                    pos = np.int64(0)
                    bound = ZSTD_compressBound(u_cnt)
                    c_len = ZSTD_compressCCtx(
                        cctx,
                        comp_scratch_u8.ctypes.data,
                        bound,
                        ones_u32[ps:pe].ctypes.data,
                        u_cnt,
                        zstd_level,
                    )
                    hdr_end = _write_page_header(out_u8, pos, u_cnt, c_len, nr)
                    hdr_len = hdr_end - pos
                    out_u8[hdr_end : hdr_end + c_len] = comp_scratch_u8[:c_len]
                    pos = hdr_end + c_len
                    meta_out[0, 0] = hdr_len + u_cnt
                    meta_out[0, 1] = hdr_len + c_len

                    c_len = ZSTD_compressCCtx(
                        cctx,
                        comp_scratch_u8.ctypes.data,
                        bound,
                        multi_u32[ps:pe].ctypes.data,
                        u_cnt,
                        zstd_level,
                    )
                    hdr_end = _write_page_header(out_u8, pos, u_cnt, c_len, nr)
                    hdr_len = hdr_end - pos
                    out_u8[hdr_end : hdr_end + c_len] = comp_scratch_u8[:c_len]
                    pos = hdr_end + c_len
                    meta_out[1, 0] = hdr_len + u_cnt
                    meta_out[1, 1] = hdr_len + c_len

                    rc, pos, tu, tc = _encode_col_pages(
                        cctx,
                        zstd_level,
                        po_i32,
                        ones_idx,
                        ps,
                        pe,
                        page_pats,
                        scratch_u8,
                        comp_scratch_u8,
                        out_u8,
                        pos,
                        is_delta,
                        is_shuffled,
                        check_sorted,
                    )
                    if rc != 0:
                        return rc, pos
                    meta_out[2, 0] = tu
                    meta_out[2, 1] = tc

                    rc, pos, tu, tc = _encode_col_pages(
                        cctx,
                        zstd_level,
                        pm_i32,
                        multi_idx,
                        ps,
                        pe,
                        page_pats,
                        scratch_u8,
                        comp_scratch_u8,
                        out_u8,
                        pos,
                        is_delta,
                        is_shuffled,
                        check_sorted,
                    )
                    if rc != 0:
                        return rc, pos
                    meta_out[3, 0] = tu
                    meta_out[3, 1] = tc

                    rc, pos, tu, tc = _encode_col_pages(
                        cctx,
                        zstd_level,
                        cm_i32,
                        multi_idx,
                        ps,
                        pe,
                        page_pats,
                        scratch_u8,
                        comp_scratch_u8,
                        out_u8,
                        pos,
                        False,
                        is_shuffled,
                        False,
                    )
                    meta_out[4, 0] = tu
                    meta_out[4, 1] = tc
                    return 0, pos

                @njit(nogil=True)
                def _dec_varint_nb(mm_u8: np.ndarray, pos: int) -> tuple[int, int]:
                    shift = 0
                    n = np.int64(0)
                    while True:
                        b = np.int64(mm_u8[pos])
                        pos += 1
                        n |= (b & 0x7F) << shift
                        if (b & 0x80) == 0:
                            break
                        shift += 7
                    return n, pos

                @njit(nogil=True)
                def _dec_zigzag_nb(mm_u8: np.ndarray, pos: int) -> tuple[int, int]:
                    n, pos = _dec_varint_nb(mm_u8, pos)
                    return (n >> 1) ^ -(n & 1), pos

                @njit(nogil=True)
                def parse_our_footer_nb(
                    footer_u8: np.ndarray,
                    schema_end_pos: int,
                    out_col_info: np.ndarray,
                ) -> tuple[int, int, int]:
                    pos = schema_end_pos
                    n_len = len(footer_u8)
                    if pos >= n_len or footer_u8[pos] != 0x16:
                        return -1, 0, 0
                    _, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                    if pos + 1 >= n_len or footer_u8[pos] != 0x19:
                        return -1, 0, 0
                    hdr_b = footer_u8[pos + 1]
                    if hdr_b == 0xFC:
                        n_rgs, pos = _dec_varint_nb(footer_u8, pos + 2)
                    else:
                        n_rgs = np.int64(hdr_b >> 4)
                        pos += 2
                    if n_rgs < 0:
                        return -1, 0, 0
                    if n_rgs > len(out_col_info):
                        return -2, n_rgs, 0

                    col_name_lens = np.array([4, 5, 10, 11, 11], dtype=np.int64)
                    for rg in range(n_rgs):
                        if (
                            pos + 1 >= n_len
                            or footer_u8[pos] != 0x19
                            or footer_u8[pos + 1] != 0x5C
                        ):
                            return -1, 0, 0
                        pos += 2
                        for c in range(5):
                            if (
                                pos + 4 >= n_len
                                or footer_u8[pos] != 0x26
                                or footer_u8[pos + 3] != 0x15
                            ):
                                return -1, 0, 0
                            _, pos = _dec_zigzag_nb(footer_u8, pos + 4)
                            pos += 6 + 1 + col_name_lens[c]
                            if pos >= n_len or footer_u8[pos] != 0x15:
                                return -1, 0, 0
                            codec, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                            if codec != 6:
                                return -1, 0, 0
                            if pos >= n_len or footer_u8[pos] != 0x16:
                                return -1, 0, 0
                            _, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                            if pos >= n_len or footer_u8[pos] != 0x16:
                                return -1, 0, 0
                            _, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                            if pos >= n_len or footer_u8[pos] != 0x16:
                                return -1, 0, 0
                            tc, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                            if pos >= n_len or footer_u8[pos] != 0x26:
                                return -1, 0, 0
                            poff, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                            out_col_info[rg, c, 0] = poff
                            out_col_info[rg, c, 1] = tc
                            pos += 11
                        for _ in range(4):
                            if pos >= n_len:
                                return -1, 0, 0
                            _, pos = _dec_zigzag_nb(footer_u8, pos + 1)
                        if pos >= n_len or footer_u8[pos] != 0x00:
                            return -1, 0, 0
                        pos += 1
                    return 0, n_rgs, pos

                @njit(nogil=True)
                def read_ones_multi_nb(
                    dctx: int,
                    mm_u8: np.ndarray,
                    col_info: np.ndarray,
                    n_rgs: int,
                    rg_pats: int,
                    num_data: int,
                    out_ones: np.ndarray,
                    out_multi: np.ndarray,
                ) -> int:
                    for rg in range(n_rgs):
                        ps = rg * rg_pats
                        pe = (
                            num_data
                            if (rg + 1) * rg_pats > num_data
                            else (rg + 1) * rg_pats
                        )
                        u_expect = (pe - ps) * 4

                        off0 = col_info[rg, 0, 0]
                        tc0 = col_info[rg, 0, 1]
                        u0, p1 = _dec_zigzag_nb(mm_u8, off0 + 3)
                        c0, _ = _dec_zigzag_nb(mm_u8, p1 + 1)
                        hdr0 = tc0 - c0
                        res0 = ZSTD_decompressDCtx(
                            dctx,
                            out_ones[ps:pe].ctypes.data,
                            u_expect,
                            mm_u8.ctypes.data + off0 + hdr0,
                            c0,
                        )
                        if res0 != u_expect or u0 != u_expect:
                            return -1

                        off1 = col_info[rg, 1, 0]
                        tc1 = col_info[rg, 1, 1]
                        u1, p1 = _dec_zigzag_nb(mm_u8, off1 + 3)
                        c1, _ = _dec_zigzag_nb(mm_u8, p1 + 1)
                        hdr1 = tc1 - c1
                        res1 = ZSTD_decompressDCtx(
                            dctx,
                            out_multi[ps:pe].ctypes.data,
                            u_expect,
                            mm_u8.ctypes.data + off1 + hdr1,
                            c1,
                        )
                        if res1 != u_expect or u1 != u_expect:
                            return -1
                    return 0

                @njit(nogil=True)
                def decode_col_pages(
                    dctx: int,
                    mm_u8: np.ndarray,
                    col_off: int,
                    col_tc: int,
                    buf_u8: np.ndarray,
                    rg_ps: int,
                    rg_pe: int,
                    pat_offsets: np.ndarray,
                    out_u32: np.ndarray,
                    pat_firsts: np.ndarray,
                    pat_lasts: np.ndarray,
                    dst_offsets: np.ndarray,
                    is_delta: bool,
                    is_shuffled: bool,
                ) -> int:
                    n_spans = len(pat_firsts)
                    if n_spans == 0:
                        return 0
                    last_req_pat = pat_lasts[n_spans - 1]
                    pos = col_off
                    col_end = col_off + col_tc
                    page_ps = rg_ps
                    s_idx = 0

                    while page_ps < rg_pe and pos < col_end:
                        if page_ps >= last_req_pat:
                            break
                        u_len, p1 = _dec_zigzag_nb(mm_u8, pos + 3)
                        c_len, p2 = _dec_zigzag_nb(mm_u8, p1 + 1)
                        page_nr, p3 = _dec_zigzag_nb(mm_u8, p2 + 2)
                        hdr_len = (p3 + 10) - pos
                        page_pe = page_ps + page_nr

                        while s_idx < n_spans and pat_lasts[s_idx] <= page_ps:
                            s_idx += 1
                        if s_idx >= n_spans:
                            break

                        if pat_firsts[s_idx] < page_pe:
                            if u_len > len(buf_u8):
                                buf_u8 = np.empty(u_len, dtype=np.uint8)
                            res = ZSTD_decompressDCtx(
                                dctx,
                                buf_u8.ctypes.data,
                                u_len,
                                mm_u8.ctypes.data + pos + hdr_len,
                                c_len,
                            )
                            if res != u_len:
                                return -1
                            page_base_elem = pat_offsets[page_ps]
                            cur_s = s_idx
                            while cur_s < n_spans and pat_firsts[cur_s] < page_pe:
                                sp_start = pat_firsts[cur_s]
                                sp_end = pat_lasts[cur_s]
                                ov_start = sp_start if sp_start > page_ps else page_ps
                                ov_end = sp_end if sp_end < page_pe else page_pe
                                out_pos = dst_offsets[cur_s] + (
                                    pat_offsets[ov_start] - pat_offsets[sp_start]
                                )
                                for p_idx in range(ov_start, ov_end):
                                    s = pat_offsets[p_idx]
                                    e = pat_offsets[p_idx + 1]
                                    k = e - s
                                    if k <= 0:
                                        continue
                                    b_s = (
                                        s - page_base_elem + (p_idx - page_ps) + 1
                                    ) * 4
                                    sub_out = out_u32[out_pos : out_pos + k]
                                    if is_shuffled:
                                        b0 = buf_u8[b_s : b_s + k]
                                        b1 = buf_u8[b_s + k : b_s + 2 * k]
                                        b2 = buf_u8[b_s + 2 * k : b_s + 3 * k]
                                        b3 = buf_u8[b_s + 3 * k : b_s + 4 * k]
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
                                    else:
                                        src_u32 = buf_u8[b_s : b_s + 4 * k].view(
                                            np.uint32
                                        )
                                        if is_delta:
                                            acc = np.uint32(0)
                                            for i in range(k):
                                                acc = np.uint32(acc + src_u32[i])
                                                sub_out[i] = acc
                                        else:
                                            for i in range(k):
                                                sub_out[i] = src_u32[i]
                                    out_pos += k
                                if sp_end <= page_pe:
                                    cur_s += 1
                                    s_idx = cur_s
                                else:
                                    break

                        pos += hdr_len + c_len
                        page_ps = page_pe
                    return 0

                @njit(nogil=True)
                def decode_rg_batch(
                    dctx: int,
                    mm_u8: np.ndarray,
                    col_offs: np.ndarray,
                    unique_rgs: np.ndarray,
                    group_starts: np.ndarray,
                    group_stops: np.ndarray,
                    group_span_ids: np.ndarray,
                    g_start: int,
                    g_stop: int,
                    rg_pats: int,
                    num_data: int,
                    ones_idx: np.ndarray,
                    multi_idx: np.ndarray,
                    firsts: np.ndarray,
                    lasts: np.ndarray,
                    dst_ones: np.ndarray,
                    dst_multi: np.ndarray,
                    out_po: np.ndarray,
                    out_pm: np.ndarray,
                    out_cm: np.ndarray,
                    buf_u8: np.ndarray,
                    is_delta: bool,
                    is_shuffled: bool,
                ) -> int:
                    for g_idx in range(g_start, g_stop):
                        rg = unique_rgs[g_idx]
                        ps = rg * rg_pats
                        pe = (
                            num_data
                            if (rg + 1) * rg_pats > num_data
                            else (rg + 1) * rg_pats
                        )
                        gs0 = group_starts[g_idx]
                        gs1 = group_stops[g_idx]
                        n_sp = gs1 - gs0
                        p_firsts = np.empty(n_sp, dtype=np.int64)
                        p_lasts = np.empty(n_sp, dtype=np.int64)
                        d_ones = np.empty(n_sp, dtype=np.int64)
                        d_multi = np.empty(n_sp, dtype=np.int64)
                        need_o = False
                        need_m = False
                        for i in range(n_sp):
                            sid = group_span_ids[gs0 + i]
                            f = firsts[sid]
                            last_p = lasts[sid]
                            p_firsts[i] = f
                            p_lasts[i] = last_p
                            d_ones[i] = dst_ones[sid]
                            d_multi[i] = dst_multi[sid]
                            if ones_idx[last_p] > ones_idx[f]:
                                need_o = True
                            if multi_idx[last_p] > multi_idx[f]:
                                need_m = True
                        if need_o:
                            rc = decode_col_pages(
                                dctx,
                                mm_u8,
                                col_offs[rg, 2, 0],
                                col_offs[rg, 2, 1],
                                buf_u8,
                                ps,
                                pe,
                                ones_idx,
                                out_po,
                                p_firsts,
                                p_lasts,
                                d_ones,
                                is_delta,
                                is_shuffled,
                            )
                            if rc != 0:
                                return rc
                        if need_m:
                            rc = decode_col_pages(
                                dctx,
                                mm_u8,
                                col_offs[rg, 3, 0],
                                col_offs[rg, 3, 1],
                                buf_u8,
                                ps,
                                pe,
                                multi_idx,
                                out_pm,
                                p_firsts,
                                p_lasts,
                                d_multi,
                                is_delta,
                                is_shuffled,
                            )
                            if rc != 0:
                                return rc
                            rc = decode_col_pages(
                                dctx,
                                mm_u8,
                                col_offs[rg, 4, 0],
                                col_offs[rg, 4, 1],
                                buf_u8,
                                ps,
                                pe,
                                multi_idx,
                                out_cm,
                                p_firsts,
                                p_lasts,
                                d_multi,
                                False,
                                is_shuffled,
                            )
                            if rc != 0:
                                return rc
                    return 0

                _zstd_c_kernels = (
                    encode_rg_all_cols,
                    parse_our_footer_nb,
                    read_ones_multi_nb,
                    decode_col_pages,
                    decode_rg_batch,
                )
    return _zstd_c_kernels


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


def _dec_uvarint_py(mv: memoryview | bytes, pos: int) -> tuple[int, int]:
    b = mv[pos]
    if b < 0x80:
        return b, pos + 1
    n = b & 0x7F
    b = mv[pos + 1]
    if b < 0x80:
        return n | (b << 7), pos + 2
    n |= (b & 0x7F) << 7
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
    file_meta = pq.read_metadata(resolved)
    meta = file_meta.metadata or {}
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
        zstd_c = _ensure_zstd_c_kernels()
        zstd_funcs = _get_libzstd()
        zstd_level = 1 if pq_level is None else int(pq_level)

        rg_bytes = max(1, rg_pats * bytes_per_pat)
        max_rgs_per_batch = max(1, buffer_size // rg_bytes)
        if len(datas) == 1 and isinstance(datas[0], PatternsSOne):
            batches_iter: Iterator[PatternsSOne] = iter([datas[0]])
        else:
            batches_iter = _iter_rg_batches(datas, rg_pats, max_rgs_per_batch)

        schema_prefix, created_by_suffix = _get_zstd_footer_template()
        kv_bytes = _encode_kv_metadata(meta_kv)
        suffix_bytes = kv_bytes + b"\x18" + created_by_suffix[1:]

        if zstd_c is not None and zstd_funcs is not None:
            encode_rg_all_cols = zstd_c[0]
            create_cctx, free_cctx, _, compress_bound, _, _, _ = zstd_funcs
            c_names = (b"ones", b"multi", b"place_ones", b"place_multi", b"count_multi")
            c_types = (1, 1, 6, 6, 6)

            n_slots = max(2, workers * 2)
            ctx_q: queue.SimpleQueue[
                tuple[int, npt.NDArray[np.uint8], npt.NDArray[np.uint8]]
            ] = queue.SimpleQueue()
            out_q: queue.SimpleQueue[npt.NDArray[np.uint8]] = queue.SimpleQueue()
            created_cctxs: list[int] = []
            for _ in range(max(1, workers)):
                cctx_ptr = int(create_cctx())
                created_cctxs.append(cctx_ptr)
                ctx_q.put(
                    (
                        cctx_ptr,
                        np.empty(256 * 1024, dtype=np.uint8),
                        np.empty(300 * 1024, dtype=np.uint8),
                    )
                )
            for _ in range(n_slots):
                out_q.put(np.empty(max(512 * 1024, rg_bytes // 2), dtype=np.uint8))

            col_metas_all: list[bytes] = []
            total_rgs = 0
            file_pos = 4
            pool = _get_worker_pool(workers) if workers > 1 else None
            try:
                with path.open("wb") as fp:
                    fp.write(_PARQUET_MAGIC)
                    for batch in batches_iter:
                        b_n = int(batch.num_data)
                        b_rgs = (b_n + rg_pats - 1) // rg_pats
                        ones_arr = np.ascontiguousarray(batch.ones, dtype=np.uint32)
                        multi_arr = np.ascontiguousarray(batch.multi, dtype=np.uint32)
                        ones_idx = batch.ones_idx.view(np.int64)
                        multi_idx = batch.multi_idx.view(np.int64)
                        po_src = np.ascontiguousarray(batch.place_ones).view(np.int32)
                        pm_src = np.ascontiguousarray(batch.place_multi).view(np.int32)
                        cm_src = np.ascontiguousarray(batch.count_multi).view(np.int32)
                        meta_batch = np.empty((b_rgs, 5, 2), dtype=np.int64)

                        def proc_rg_c(
                            rg: int,
                        ) -> tuple[int, int, int, npt.NDArray[np.uint8]]:
                            cctx_ptr, sc, csc = ctx_q.get()
                            out_buf = out_q.get()
                            returned_out = False
                            try:
                                ps = rg * rg_pats
                                pe = min(b_n, (rg + 1) * rg_pats)
                                nr = pe - ps
                                no = int(ones_idx[pe] - ones_idx[ps])
                                nm = int(multi_idx[pe] - multi_idx[ps])
                                need_sc = max((no + nr) * 4, (nm + nr) * 4, nr * 4, 64)
                                if len(sc) < need_sc:
                                    sc = np.empty(need_sc, dtype=np.uint8)
                                need_csc = int(compress_bound(need_sc)) + 64
                                if len(csc) < need_csc:
                                    csc = np.empty(need_csc, dtype=np.uint8)
                                n_pages = (nr + page_pats - 1) // page_pats
                                need_out = (
                                    int(compress_bound(nr * 4)) * 2
                                    + int(compress_bound((no + nr) * 4))
                                    + int(compress_bound((nm + nr) * 4)) * 2
                                    + (n_pages * 3 + 2) * 64
                                )
                                if len(out_buf) < need_out:
                                    out_buf = np.empty(need_out, dtype=np.uint8)
                                rc, pos = encode_rg_all_cols(
                                    cctx_ptr,
                                    zstd_level,
                                    ones_arr,
                                    multi_arr,
                                    po_src,
                                    pm_src,
                                    cm_src,
                                    ones_idx,
                                    multi_idx,
                                    ps,
                                    pe,
                                    page_pats,
                                    sc,
                                    csc,
                                    out_buf,
                                    meta_batch[rg],
                                    is_delta,
                                    shuffle,
                                    check_sorted,
                                )
                                returned_out = True
                                return int(rc), nr, int(pos), out_buf
                            finally:
                                ctx_q.put((cctx_ptr, sc, csc))
                                if not returned_out:
                                    out_q.put(out_buf)

                        rg_iter = (
                            pool.map(proc_rg_c, range(b_rgs))
                            if (pool is not None and b_rgs > 1)
                            else (proc_rg_c(rg) for rg in range(b_rgs))
                        )
                        for rg, (rc, nr, pos, out_buf) in enumerate(rg_iter):
                            try:
                                if rc == -1:
                                    raise ValueError(
                                        "position values must fit in uint32"
                                    )
                                if rc == -2:
                                    raise ValueError(
                                        "position values must be sorted within every pattern"
                                    )
                                total_rgs += 1
                                rg_start_pos = file_pos
                                fp.write(memoryview(out_buf)[:pos])
                            finally:
                                out_q.put(out_buf)
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
            finally:
                for cctx_ptr in created_cctxs:
                    free_cctx(cctx_ptr)
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
            pool = _get_worker_pool(min(workers, b_rgs))
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
            data_page_size=max(4096, page_bytes),
        )
        return

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
    if (
        col_info is not None
        and sys.byteorder == "little"
        and importlib.util.find_spec("zstandard") is not None
    ):
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

            if importlib.util.find_spec("numba") is not None:
                zstd_c = _ensure_zstd_c_kernels()
                zstd_funcs = _get_libzstd()
                if zstd_c is not None and zstd_funcs is not None:
                    read_ones_multi_nb = zstd_c[2]
                    _, _, _, _, create_dctx, free_dctx, _ = zstd_funcs
                    mm_u8 = np.frombuffer(mm_obj, dtype=np.uint8)
                    dctx = int(create_dctx())
                    try:
                        rc = int(
                            read_ones_multi_nb(
                                dctx,
                                mm_u8,
                                col_info,
                                n_rgs,
                                rg_pats,
                                num_data,
                                out_ones,
                                out_multi,
                            )
                        )
                    finally:
                        free_dctx(dctx)
                    del mm_u8
                    if rc == 0:
                        return out_ones, out_multi

            import zstandard

            dctx_py = zstandard.ZstdDecompressor()
            mv = memoryview(mm_obj)
            try:
                for rg in range(n_rgs):
                    ps = rg * rg_pats
                    pe = min(num_data, (rg + 1) * rg_pats)
                    u_expect = (pe - ps) * 4

                    off0 = int(col_info[rg, 0, 0])
                    tc0 = int(col_info[rg, 0, 1])
                    _, p1 = _dec_zigzag_py(mv, off0 + 3)
                    c0, _ = _dec_zigzag_py(mv, p1 + 1)
                    hdr0 = tc0 - c0
                    raw0 = dctx_py.decompress(
                        mv[off0 + hdr0 : off0 + tc0], max_output_size=u_expect
                    )
                    out_ones[ps:pe] = np.frombuffer(raw0, dtype=np.uint32)

                    off1 = int(col_info[rg, 1, 0])
                    tc1 = int(col_info[rg, 1, 1])
                    _, p1 = _dec_zigzag_py(mv, off1 + 3)
                    c1, _ = _dec_zigzag_py(mv, p1 + 1)
                    hdr1 = tc1 - c1
                    raw1 = dctx_py.decompress(
                        mv[off1 + hdr1 : off1 + tc1], max_output_size=u_expect
                    )
                    out_multi[ps:pe] = np.frombuffer(raw1, dtype=np.uint32)
            finally:
                del mv
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


def _extract_single_page_col_py(
    mv: memoryview,
    dctx: Any,
    col_off: int,
    col_tc: int,
    rg: int,
    col_idx: int,
    rg_ps: int,
    rg_pe: int,
    frame: int,
    pat_offsets: npt.NDArray[np.uint64],
    is_delta: bool,
    is_shuffled: bool,
    dtype: str,
    page_lru: dict[tuple[int, int, int], bytes] | None,
) -> npt.NDArray[Any]:
    s = int(pat_offsets[frame])
    e = int(pat_offsets[frame + 1])
    k = e - s
    if k <= 0:
        return np.empty(0, dtype=dtype)
    pos = col_off
    col_end = col_off + col_tc
    page_ps = rg_ps
    while page_ps < rg_pe and pos < col_end:
        u_len, p1 = _dec_zigzag_py(mv, pos + 3)
        c_len, p2 = _dec_zigzag_py(mv, p1 + 1)
        page_nr, p3 = _dec_zigzag_py(mv, p2 + 2)
        hdr_len = (p3 + 10) - pos
        page_pe = page_ps + page_nr
        if frame < page_pe:
            key = (rg, col_idx, page_ps)
            raw = None
            if page_lru is not None and key in page_lru:
                raw = page_lru.pop(key)
                page_lru[key] = raw
            if raw is None:
                raw = dctx.decompress(
                    mv[pos + hdr_len : pos + hdr_len + c_len], max_output_size=u_len
                )
                if page_lru is not None:
                    page_lru[key] = raw
                    if len(page_lru) > 6:
                        page_lru.pop(next(iter(page_lru)))
            u8 = np.frombuffer(raw, dtype=np.uint8)
            b_s = (s - int(pat_offsets[page_ps]) + (frame - page_ps) + 1) * 4
            if is_shuffled:
                arr = (
                    u8[b_s : b_s + 4 * k].reshape(4, k).T.copy().view(dtype).reshape(-1)
                )
            else:
                arr = u8[b_s : b_s + 4 * k].view(dtype).copy()
            if is_delta:
                np.cumsum(arr, dtype=np.uint32, out=arr)
            return arr
        pos += hdr_len + c_len
        page_ps = page_pe
    raise RuntimeError("Frame not found in Parquet column chunk")


class _ZstdDCtxHolder:
    def __init__(self, ptr: int, free_fn: Any) -> None:
        self.ptr = ptr
        self._free_fn = free_fn

    def __del__(self) -> None:
        if self.ptr:
            self._free_fn(self.ptr)
            self.ptr = 0


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
    """Fast single-frame read using page-level decompression and pure NumPy."""
    del payload_codec
    rg = frame // rg_pats
    ps = rg * rg_pats
    pe = min(num_data, (rg + 1) * rg_pats)

    if (
        col_info is not None
        and sys.byteorder == "little"
        and importlib.util.find_spec("zstandard") is not None
    ):
        mm_obj = cache.get("mm") if cache is not None else None
        fp_local = None
        mm_local = None
        try:
            if mm_obj is None:
                assert path is not None
                fp_local = path.open("rb")
                mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                mm_obj = mm_local

            if importlib.util.find_spec("numba") is not None:
                zstd_c = _ensure_zstd_c_kernels()
                zstd_funcs = _get_libzstd()
                if zstd_c is not None and zstd_funcs is not None:
                    decode_col_pages = zstd_c[3]
                    _, _, _, _, create_dctx, free_dctx, _ = zstd_funcs
                    k_o = int(ones_idx[frame + 1]) - int(ones_idx[frame])
                    k_m = int(multi_idx[frame + 1]) - int(multi_idx[frame])
                    po = np.empty(k_o, dtype=np.uint32)
                    pm = np.empty(k_m, dtype=np.uint32)
                    cm = np.empty(k_m, dtype=np.uint32)
                    if k_o == 0 and k_m == 0:
                        return po, pm, cm.view(np.int32)

                    holder = cache.get("c_dctx") if cache is not None else None
                    if holder is None:
                        holder = _ZstdDCtxHolder(int(create_dctx()), free_dctx)
                        if cache is not None:
                            cache["c_dctx"] = holder
                    s_buf = cache.get("s_buf") if cache is not None else None
                    need_buf = max(256 * 1024, (max(k_o, k_m) + 16) * 4)
                    if s_buf is None or len(s_buf) < need_buf:
                        s_buf = np.empty(need_buf, dtype=np.uint8)
                        if cache is not None:
                            cache["s_buf"] = s_buf

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
                    pf_arr = np.array([frame], dtype=np.int64)
                    pl_arr = np.array([frame + 1], dtype=np.int64)
                    dst_arr = np.array([0], dtype=np.int64)
                    mm_u8 = np.frombuffer(mm_obj, dtype=np.uint8)
                    try:
                        if k_o > 0:
                            decode_col_pages(
                                holder.ptr,
                                mm_u8,
                                int(col_info[rg, 2, 0]),
                                int(col_info[rg, 2, 1]),
                                s_buf,
                                ps,
                                pe,
                                ones_i64,
                                po,
                                pf_arr,
                                pl_arr,
                                dst_arr,
                                is_delta,
                                is_shuffled,
                            )
                        if k_m > 0:
                            decode_col_pages(
                                holder.ptr,
                                mm_u8,
                                int(col_info[rg, 3, 0]),
                                int(col_info[rg, 3, 1]),
                                s_buf,
                                ps,
                                pe,
                                multi_i64,
                                pm,
                                pf_arr,
                                pl_arr,
                                dst_arr,
                                is_delta,
                                is_shuffled,
                            )
                            decode_col_pages(
                                holder.ptr,
                                mm_u8,
                                int(col_info[rg, 4, 0]),
                                int(col_info[rg, 4, 1]),
                                s_buf,
                                ps,
                                pe,
                                multi_i64,
                                cm,
                                pf_arr,
                                pl_arr,
                                dst_arr,
                                False,
                                is_shuffled,
                            )
                        return po, pm, cm.view(np.int32)
                    finally:
                        del mm_u8

            import zstandard

            dctx = cache.get("dctx") if cache is not None else None
            if dctx is None:
                dctx = zstandard.ZstdDecompressor()
                if cache is not None:
                    cache["dctx"] = dctx
            page_lru = cache.setdefault("page_lru", {}) if cache is not None else None
            mv = memoryview(mm_obj)
            try:
                po = _extract_single_page_col_py(
                    mv,
                    dctx,
                    int(col_info[rg, 2, 0]),
                    int(col_info[rg, 2, 1]),
                    rg,
                    2,
                    ps,
                    pe,
                    frame,
                    ones_idx,
                    is_delta,
                    is_shuffled,
                    "<u4",
                    page_lru,
                )
                pm = _extract_single_page_col_py(
                    mv,
                    dctx,
                    int(col_info[rg, 3, 0]),
                    int(col_info[rg, 3, 1]),
                    rg,
                    3,
                    ps,
                    pe,
                    frame,
                    multi_idx,
                    is_delta,
                    is_shuffled,
                    "<u4",
                    page_lru,
                )
                cm = _extract_single_page_col_py(
                    mv,
                    dctx,
                    int(col_info[rg, 4, 0]),
                    int(col_info[rg, 4, 1]),
                    rg,
                    4,
                    ps,
                    pe,
                    frame,
                    multi_idx,
                    False,
                    is_shuffled,
                    "<i4",
                    page_lru,
                )
                return po, pm, cm
            finally:
                del mv
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

    use_numba_decode = (
        workers > 0
        and sys.byteorder == "little"
        and importlib.util.find_spec("numba") is not None
    )
    has_zstd = importlib.util.find_spec("zstandard") is not None

    if col_info is not None and use_numba_decode and has_zstd:
        zstd_c = _ensure_zstd_c_kernels()
        zstd_funcs = _get_libzstd()
        if zstd_c is not None and zstd_funcs is not None:
            decode_rg_batch = zstd_c[4]
            _, _, _, _, create_dctx, free_dctx, _ = zstd_funcs
            group_starts = split_pts.astype(np.int64)
            group_stops = np.empty_like(group_starts)
            group_stops[:-1] = group_starts[1:]
            group_stops[-1] = len(order)
            group_span_ids = order.astype(np.int64)
            n_groups = len(unique_rgs)
            n_workers = max(1, min(workers, n_groups))

            mm_obj = cache.get("mm") if cache is not None else None
            fp_local = None
            mm_local = None
            dctxs = [int(create_dctx()) for _ in range(n_workers)]
            bufs = [np.empty(256 * 1024, dtype=np.uint8) for _ in range(n_workers)]
            try:
                if mm_obj is None:
                    fp_local = path.open("rb")
                    mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                    mm_obj = mm_local
                mm_u8 = np.frombuffer(mm_obj, dtype=np.uint8)
                try:
                    if n_workers > 1:
                        bounds = np.linspace(0, n_groups, n_workers + 1, dtype=np.int64)
                        pool = _get_worker_pool(n_workers)
                        futs = [
                            pool.submit(
                                decode_rg_batch,
                                dctxs[w],
                                mm_u8,
                                col_info,
                                unique_rgs,
                                group_starts,
                                group_stops,
                                group_span_ids,
                                int(bounds[w]),
                                int(bounds[w + 1]),
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
                                bufs[w],
                                is_delta,
                                shuffle,
                            )
                            for w in range(n_workers)
                        ]
                        for fut in futs:
                            if int(fut.result()) != 0:
                                raise ValueError("Corrupted Zstd page in Parquet file")
                    else:
                        rc = int(
                            decode_rg_batch(
                                dctxs[0],
                                mm_u8,
                                col_info,
                                unique_rgs,
                                group_starts,
                                group_stops,
                                group_span_ids,
                                0,
                                n_groups,
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
                                bufs[0],
                                is_delta,
                                shuffle,
                            )
                        )
                        if rc != 0:
                            raise ValueError("Corrupted Zstd page in Parquet file")
                finally:
                    del mm_u8
            finally:
                for dctx in dctxs:
                    free_dctx(dctx)
                if mm_local is not None:
                    mm_local.close()
                if fp_local is not None:
                    fp_local.close()
            return out_po, out_pm, out_cm.view(np.int32)

    _, pq = _require_pyarrow()
    if pf is None:
        pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)

    groups = np.split(order, split_pts[1:])
    decode_kernel = _ensure_decode_kernel() if use_numba_decode else None

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
    use_numba_decode = (
        workers > 0
        and sys.byteorder == "little"
        and importlib.util.find_spec("numba") is not None
    )
    has_zstd = importlib.util.find_spec("zstandard") is not None
    _, col_info = _get_file_info(path, cache=cache)

    offsets_i64 = (
        offsets.view(np.int64)
        if offsets.dtype == np.uint64
        else offsets.astype(np.int64, copy=False)
    )
    out_u32 = np.empty(total_elems, dtype=np.uint32)
    n_rgs = (num_data + rg_pats - 1) // rg_pats

    col_idx_map = {"place_ones": 2, "place_multi": 3, "count_multi": 4}
    if col_info is not None and column in col_idx_map and use_numba_decode and has_zstd:
        zstd_c = _ensure_zstd_c_kernels()
        zstd_funcs = _get_libzstd()
        if zstd_c is not None and zstd_funcs is not None:
            decode_col_pages = zstd_c[3]
            _, _, _, _, create_dctx, free_dctx, _ = zstd_funcs
            c_idx = col_idx_map[column]
            n_workers = max(1, min(workers, n_rgs))
            dctxs = [int(create_dctx()) for _ in range(n_workers)]
            bufs = [np.empty(256 * 1024, dtype=np.uint8) for _ in range(n_workers)]

            mm_obj = cache.get("mm") if cache is not None else None
            fp_local = None
            mm_local = None
            try:
                if mm_obj is None:
                    fp_local = path.open("rb")
                    mm_local = mmap.mmap(fp_local.fileno(), 0, access=mmap.ACCESS_READ)
                    mm_obj = mm_local
                mm_u8 = np.frombuffer(mm_obj, dtype=np.uint8)
                try:

                    def proc_col_range(w: int, r0: int, r1: int) -> None:
                        dctx = dctxs[w]
                        buf = bufs[w]
                        for rg in range(r0, r1):
                            ps = rg * rg_pats
                            pe = min(num_data, (rg + 1) * rg_pats)
                            rg_base = int(offsets_i64[ps])
                            if int(offsets_i64[pe]) <= rg_base:
                                continue
                            pf_arr = np.array([ps], dtype=np.int64)
                            pl_arr = np.array([pe], dtype=np.int64)
                            dst_arr = np.array([rg_base], dtype=np.int64)
                            rc = int(
                                decode_col_pages(
                                    dctx,
                                    mm_u8,
                                    int(col_info[rg, c_idx, 0]),
                                    int(col_info[rg, c_idx, 1]),
                                    buf,
                                    ps,
                                    pe,
                                    offsets_i64,
                                    out_u32,
                                    pf_arr,
                                    pl_arr,
                                    dst_arr,
                                    is_delta,
                                    shuffle,
                                )
                            )
                            if rc != 0:
                                raise ValueError("Corrupted Zstd page in Parquet file")

                    if n_workers > 1:
                        bounds = np.linspace(0, n_rgs, n_workers + 1, dtype=np.int64)
                        pool = _get_worker_pool(n_workers)
                        futs = [
                            pool.submit(
                                proc_col_range, w, int(bounds[w]), int(bounds[w + 1])
                            )
                            for w in range(n_workers)
                        ]
                        for fut in futs:
                            fut.result()
                    else:
                        proc_col_range(0, 0, n_rgs)
                finally:
                    del mm_u8
            finally:
                for dctx in dctxs:
                    free_dctx(dctx)
                if mm_local is not None:
                    mm_local.close()
                if fp_local is not None:
                    fp_local.close()
            return out_u32.view(dtype)

    _, pq = _require_pyarrow()
    decode_kernel = _ensure_decode_kernel() if use_numba_decode else None
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
        pool = _get_worker_pool(min(workers, n_rgs))
        list(pool.map(proc_rg, range(n_rgs)))
    else:
        for rg in range(n_rgs):
            proc_rg(rg)

    return out_u32.view(dtype)
