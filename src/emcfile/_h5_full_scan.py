"""Direct-chunk full scans for the standard HDF5 shuffle+Zstd layout."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import local
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from ._h5_workers import env_workers

MIN_FULL_SCAN_BYTES = 8 * 1024**2
_unshuffle_kernel: Any | None = None
_fused_delta_kernel: Any | None = None
_delta_only_kernel: Any | None = None
_fixup_kernel: Any | None = None
_THREAD_STATE = local()


def _unshuffle_u32(src: Any, dst: Any, n: int, src_offset: int = 0) -> None:
    """Use Numba only once the eligible direct path is actually selected."""
    global _unshuffle_kernel
    if _unshuffle_kernel is None:
        from numba import njit

        @njit(nogil=True)
        def kernel(source: Any, output: Any, size: int, offset: int) -> None:
            valid = output.size
            b0 = source[offset : offset + valid]
            b1 = source[size + offset : size + offset + valid]
            b2 = source[2 * size + offset : 2 * size + offset + valid]
            b3 = source[3 * size + offset : 3 * size + offset + valid]
            for index in range(valid):
                output[index] = (
                    np.uint32(b0[index])
                    | (np.uint32(b1[index]) << np.uint32(8))
                    | (np.uint32(b2[index]) << np.uint32(16))
                    | (np.uint32(b3[index]) << np.uint32(24))
                )

        _unshuffle_kernel = kernel
    _unshuffle_kernel(src, dst, n, src_offset)


def _ensure_fused_kernels() -> tuple[Any, Any, Any]:
    global _fused_delta_kernel, _delta_only_kernel, _fixup_kernel
    if _fused_delta_kernel is None:
        from numba import njit

        @njit(nogil=True)
        def fused_l1(
            src_u8: Any,
            dst_u32: Any,
            chunk_n: int,
            offsets: Any,
            chunk_start: int,
            p_idx: int,
            acc: np.uint32,
        ) -> tuple[int, np.uint32]:
            valid_n = dst_u32.size
            num_p = offsets.size - 1
            block_size = 4096
            for blk in range(0, valid_n, block_size):
                blk_end = min(blk + block_size, valid_n)
                b0 = src_u8[blk:blk_end]
                b1 = src_u8[chunk_n + blk : chunk_n + blk_end]
                b2 = src_u8[2 * chunk_n + blk : 2 * chunk_n + blk_end]
                b3 = src_u8[3 * chunk_n + blk : 3 * chunk_n + blk_end]
                out_blk = dst_u32[blk:blk_end]
                for i in range(blk_end - blk):
                    out_blk[i] = (
                        np.uint32(b0[i])
                        | (np.uint32(b1[i]) << np.uint32(8))
                        | (np.uint32(b2[i]) << np.uint32(16))
                        | (np.uint32(b3[i]) << np.uint32(24))
                    )
                g_pos = chunk_start + blk
                g_end = chunk_start + blk_end
                while g_pos < g_end:
                    while p_idx < num_p and np.int64(offsets[p_idx + 1]) <= g_pos:
                        p_idx += 1
                    if p_idx >= num_p:
                        break
                    p_start = np.int64(offsets[p_idx])
                    p_stop = np.int64(offsets[p_idx + 1])
                    if g_pos == p_start:
                        acc = np.uint32(0)
                    seg_end = p_stop if p_stop < g_end else g_end
                    l_start = g_pos - chunk_start
                    l_stop = seg_end - chunk_start
                    for j in range(l_start, l_stop):
                        acc = np.uint32(acc + dst_u32[j])
                        dst_u32[j] = acc
                    g_pos = seg_end
            return p_idx, acc

        @njit(nogil=True)
        def delta_only(
            dst_u32: Any,
            offsets: Any,
            chunk_start: int,
            p_idx: int,
            acc: np.uint32,
        ) -> tuple[int, np.uint32]:
            valid_n = dst_u32.size
            num_p = offsets.size - 1
            g_pos = chunk_start
            g_end = chunk_start + valid_n
            while g_pos < g_end:
                while p_idx < num_p and np.int64(offsets[p_idx + 1]) <= g_pos:
                    p_idx += 1
                if p_idx >= num_p:
                    break
                p_start = np.int64(offsets[p_idx])
                p_stop = np.int64(offsets[p_idx + 1])
                if g_pos == p_start:
                    acc = np.uint32(0)
                seg_end = p_stop if p_stop < g_end else g_end
                l_start = g_pos - chunk_start
                l_stop = seg_end - chunk_start
                for j in range(l_start, l_stop):
                    acc = np.uint32(acc + dst_u32[j])
                    dst_u32[j] = acc
                g_pos = seg_end
            return p_idx, acc

        @njit(nogil=True)
        def fixup_boundaries(out: Any, offsets: Any, batch_starts: Any) -> None:
            num_b = batch_starts.size - 1
            num_p = offsets.size - 1
            for b in range(1, num_b):
                b_start = batch_starts[b]
                b_stop = batch_starts[b + 1]
                if b_start <= 0 or b_start >= out.size:
                    continue
                lo = 0
                hi = num_p
                while lo < hi:
                    mid = (lo + hi) // 2
                    if np.int64(offsets[mid + 1]) <= b_start:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < num_p:
                    p_start = np.int64(offsets[lo])
                    p_stop = np.int64(offsets[lo + 1])
                    if p_start < b_start and p_stop > b_start:
                        end = p_stop if p_stop < b_stop else b_stop
                        base = out[b_start - 1]
                        for j in range(b_start, end):
                            out[j] = np.uint32(out[j] + base)

        _fused_delta_kernel = fused_l1
        _delta_only_kernel = delta_only
        _fixup_kernel = fixup_boundaries
    return _fused_delta_kernel, _delta_only_kernel, _fixup_kernel


def _eligible_dataset(dataset: h5py.Dataset) -> bool:
    if (
        sys.byteorder != "little"
        or dataset.is_virtual
        or dataset.ndim != 1
        or dataset.chunks is None
    ):
        return False
    if dataset.dtype.str not in ("<u4", "<i4"):
        return False
    plist = dataset.id.get_create_plist()
    if plist.get_nfilters() != 2:
        return False
    shuffle, zstd = plist.get_filter(0), plist.get_filter(1)
    n = dataset.chunks[0]
    return shuffle[0] == 2 and shuffle[2] == (4,) and zstd[0] == 32015 and n > 0


def _decode_chunk(
    mask: int,
    payload: bytes,
    n: int,
    *,
    decoder: Any = None,
    output: Any = None,
    selection: slice | None = None,
    spans: Any = None,
) -> npt.NDArray[np.uint32]:
    """Decode the exact two-filter layout, respecting per-chunk skip bits."""
    import zstandard

    if mask & ~3:
        raise ValueError("Unexpected HDF5 v2 chunk filter mask")
    if decoder is None:
        decoder = getattr(_THREAD_STATE, "decoder", None)
        if decoder is None:
            decoder = zstandard.ZstdDecompressor()
            _THREAD_STATE.decoder = decoder
    raw = payload if mask & 2 else decoder.decompress(payload, max_output_size=n * 4)
    if len(raw) != n * 4:
        raise ValueError("Unexpected HDF5 v2 decoded chunk length")
    if spans is not None:
        assert output is not None
        if mask & 1:
            values = np.frombuffer(raw, "<u4")
            out_u32 = output.view(np.uint32)
            for destination, source, span in spans:
                out_u32[destination : destination + span] = values[
                    source : source + span
                ]
            return output
        raw_u8 = np.frombuffer(raw, "u1")
        out_u32 = output.view(np.uint32)
        for destination, source, span in spans:
            _unshuffle_u32(raw_u8, out_u32[destination : destination + span], n, source)
        return output
    if selection is not None:
        if mask & 1:
            selected = np.frombuffer(raw, "<u4")[selection]
            if output is not None:
                output.view("<u4")[:] = selected
                return output
            return selected.copy()
        # Only transpose the requested byte-plane span, not the entire chunk.
        u = np.frombuffer(raw, "u1").reshape(4, n)[:, selection]
        if output is None:
            output = np.empty(u.shape[1], "<u4")
        dst_2d = output.view("u1").reshape(-1, 4)
        dst_2d[:, 0] = u[0]
        dst_2d[:, 1] = u[1]
        dst_2d[:, 2] = u[2]
        dst_2d[:, 3] = u[3]
        return output
    if mask & 1:
        values = np.frombuffer(raw, "<u4")
        if output is None:
            return values
        output[:] = values[: output.size]
        return output
    if output is None:
        output = np.empty(n, "u4")
    _unshuffle_u32(np.frombuffer(raw, "u1"), output, n)
    return output


def eligible(group: Any, *, datasets: list[h5py.Dataset] | None = None) -> bool:
    """Whether *group* uses the exact portable physical layout we own."""
    names = ("place_ones", "place_multi", "count_multi")
    if (
        str(group.attrs.get("version", "")) != "2"
        or str(group.attrs.get("position_encoding", "absolute")) != "delta"
        or any(name not in group for name in names)
    ):
        return False
    if datasets is None:
        datasets = [group[name] for name in names]
    return [dataset.dtype.str for dataset in datasets] == ["<u4", "<u4", "<i4"] and all(
        _eligible_dataset(dataset) for dataset in datasets
    )


def _read(
    dataset: h5py.Dataset,
    pool: ThreadPoolExecutor,
    workers: int,
    offsets: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[Any]:
    n = dataset.chunks[0]
    out = np.empty(dataset.size, dtype=np.uint32)
    if out.size == 0:
        return out.view(dataset.dtype)

    if offsets is None:

        def work_plain(batch: list[tuple[int, int, bytes]]) -> None:
            import zstandard

            decoder = getattr(_THREAD_STATE, "decoder", None)
            if decoder is None:
                decoder = zstandard.ZstdDecompressor()
                _THREAD_STATE.decoder = decoder
            for start, mask, payload in batch:
                dst = out[start : min(start + n, out.size)]
                _decode_chunk(mask, payload, n, decoder=decoder, output=dst)

        pending, batch = [], []
        batch_size = max(1, min(64, 8 * 1024**2 // max(1, n * 4)))
        for start in range(0, out.size, n):
            mask, payload = dataset.id.read_direct_chunk((start,))
            batch.append((start, mask, payload))
            if len(batch) == batch_size:
                pending.append(pool.submit(work_plain, batch))
                batch = []
                if len(pending) >= workers * 3:
                    pending.pop(0).result()
        if batch:
            pending.append(pool.submit(work_plain, batch))
        for future in pending:
            future.result()
        return out.view(dataset.dtype)

    fused_l1, delta_only, fixup_boundaries = _ensure_fused_kernels()

    def work_delta(batch: list[tuple[int, int, bytes]]) -> None:
        import zstandard

        decoder = getattr(_THREAD_STATE, "decoder", None)
        if decoder is None:
            decoder = zstandard.ZstdDecompressor()
            _THREAD_STATE.decoder = decoder
        b_start = batch[0][0]
        p_idx = max(0, int(np.searchsorted(offsets, b_start, side="right")) - 1)
        acc = np.uint32(0)
        for start, mask, payload in batch:
            if mask & ~3:
                raise ValueError("Unexpected HDF5 v2 chunk filter mask")
            raw = (
                payload
                if mask & 2
                else decoder.decompress(payload, max_output_size=n * 4)
            )
            if len(raw) != n * 4:
                raise ValueError("Unexpected HDF5 v2 decoded chunk length")
            dst = out[start : min(start + n, out.size)]
            if mask & 1:
                dst[:] = np.frombuffer(raw, "<u4")[: dst.size]
                p_idx, acc = delta_only(dst, offsets, start, p_idx, acc)
            else:
                p_idx, acc = fused_l1(
                    np.frombuffer(raw, "u1"), dst, n, offsets, start, p_idx, acc
                )

    pending, batch = [], []
    batch_starts = [0]
    batch_size = max(1, min(64, 8 * 1024**2 // max(1, n * 4)))
    for start in range(0, out.size, n):
        mask, payload = dataset.id.read_direct_chunk((start,))
        batch.append((start, mask, payload))
        if len(batch) == batch_size:
            batch_starts.append(min(out.size, start + n))
            pending.append(pool.submit(work_delta, batch))
            batch = []
            if len(pending) >= workers * 3:
                pending.pop(0).result()
    if batch:
        batch_starts.append(out.size)
        pending.append(pool.submit(work_delta, batch))
    for future in pending:
        future.result()
    if len(batch_starts) > 2:
        fixup_boundaries(out, offsets, np.asarray(batch_starts, dtype=np.int64))
    return out.view(dataset.dtype)


def full_scan(
    group: Any, ones_offsets: Any, multi_offsets: Any
) -> (
    tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]] | None
):
    """Read an eligible complete v2 payload, otherwise return ``None``."""
    workers = env_workers("EMCFILE_H5_FULL_SCAN_WORKERS", 4, allow_zero=True)
    if workers <= 0 or not eligible(group):
        return None
    if any(importlib.util.find_spec(name) is None for name in ("numba", "zstandard")):
        return None
    datasets = [group[name] for name in ("place_ones", "place_multi", "count_multi")]
    if (
        sum(dataset.size * dataset.dtype.itemsize for dataset in datasets)
        < MIN_FULL_SCAN_BYTES
    ):
        return None
    for dataset, offsets in zip(datasets, (ones_offsets, multi_offsets, multi_offsets)):
        if (
            offsets.ndim != 1
            or offsets.size == 0
            or offsets[0] != 0
            or int(offsets[-1]) != dataset.size
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("Pattern counts do not match HDF5 v2 payload size")
    dummy_raw = np.frombuffer(bytes(4), "u1")
    dummy_out = np.empty(1, "u4")
    dummy_offsets = np.array([0, 1], dtype=np.uint64)
    _unshuffle_u32(dummy_raw, dummy_out, 1)
    fused_l1, delta_only, fixup_boundaries = _ensure_fused_kernels()
    fused_l1(dummy_raw, dummy_out, 1, dummy_offsets, 0, 0, np.uint32(0))
    delta_only(dummy_out, dummy_offsets, 0, 0, np.uint32(0))
    fixup_boundaries(dummy_out, dummy_offsets, np.array([0, 1], dtype=np.int64))
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dataset, offsets in zip(datasets, (ones_offsets, multi_offsets, None)):
            outputs.append(_read(dataset, pool, workers, offsets))
    return outputs[0], outputs[1], outputs[2]
