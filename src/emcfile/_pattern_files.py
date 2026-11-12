from __future__ import annotations

import glob
import io
import logging
import os
from collections.abc import Sequence
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Any, cast, overload

import h5py
import numpy as np
import numpy.typing as npt

from ._delta import decode_pattern_local_delta
from ._emc_accel import (
    EmcAccelEntry,
    create_private_emc_entry,
    emc_workers,
    get_emc_cache_entry,
    get_emc_rs,
)
from ._emc_patterns import (
    INT_INDEX,
    SPARSE_PATTERN,
    TRANGE,
    PatternsSOne,
    PatternsSOneBase,
    _count_offsets,
    _index_1d_pattern_columns,
    _is_int_index,
    write_patterns,
)
from ._formatting import pretty_size
from ._h5_accel import (
    H5AccelEntry,
    LazyH5Group,
    create_private_entry,
    get_cache_entry,
    get_h5_rs,
)
from ._h5_filters import ensure_group_filters
from ._hdf5 import PATH_TYPE, H5Path, h5path, make_path
from ._html_display import html_card
from ._indexing import contiguous_ranges

__all__ = [
    "EMCBinaryPatternFile",
    "EMCPatternCollection",
    "FileBackedEMCPatterns",
    "HDF5PatternFile",
    "LegacyHDF5PatternFile",
    "ParquetPatternFile",
    "PatternsSOneEMC",
    "PatternsSOneH5",
    "PatternsSOneList",
    "PatternsSOneParquet",
    "file_patterns",
    "open_patterns",
]
_log = logging.getLogger(__name__)

I4 = np.dtype("i4").itemsize

INDEX_ARRAY = npt.NDArray[np.integer[Any]]


def _materialize_dense_pattern(
    num_pix: int,
    place_ones: npt.NDArray[np.uint32],
    place_multi: npt.NDArray[np.uint32],
    count_multi: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    rs = get_h5_rs()
    if (
        rs is not None
        and place_ones.dtype == np.uint32
        and place_multi.dtype == np.uint32
        and count_multi.dtype == np.int32
        and place_ones.flags.c_contiguous
        and place_multi.flags.c_contiguous
        and count_multi.flags.c_contiguous
    ):
        return cast(
            npt.NDArray[np.int32],
            rs.scatter_dense_pattern(num_pix, place_ones, place_multi, count_multi),
        )
    ans = np.zeros(num_pix, np.int32)
    ans[place_ones] = 1
    ans[place_multi] = count_multi
    return ans


def _dispatch_2d_file_getitem(
    source: PatternsSOneBase, axes: tuple[Any, Any]
) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
    ax0, ax1 = axes
    if isinstance(ax0, (bool, np.bool_)) or isinstance(ax1, (bool, np.bool_)):
        raise IndexError("Boolean scalar indices are not supported")
    if _is_int_index(ax0):
        return _index_1d_pattern_columns(source[int(ax0)], ax1)
    if _is_int_index(ax1):
        col = int(ax1)
        if col < 0:
            col += source.num_pix
        if col < 0 or col >= source.num_pix:
            raise IndexError(
                f"column index {ax1} is out of bounds for axis 1 "
                f"with size {source.num_pix}"
            )
        return cast(PatternsSOne, source[cast(TRANGE, ax0)])[:, col]
    return cast(PatternsSOne, source[cast(TRANGE, ax0)])[:, cast(TRANGE, ax1)]


def _normalize_row_index_ranges(
    index: int | np.integer[Any] | TRANGE, num_data: int
) -> tuple[bool, INDEX_ARRAY]:
    if isinstance(index, (bool, np.bool_)):
        raise IndexError("Boolean scalar indices are not supported")
    match index:
        case int() | np.integer():
            idx = int(index)
            if idx < 0:
                idx += num_data
            if idx < 0 or idx >= num_data:
                raise IndexError("Pattern index out of range")
            return True, np.array([[idx, idx + 1]], dtype=np.int64)
        case np.ndarray() if np.issubdtype(index.dtype, bool):
            if index.ndim != 1 or index.size != num_data:
                raise IndexError("Boolean row mask must match the number of patterns")
            return False, contiguous_ranges(np.where(index)[0])
        case np.ndarray() if np.issubdtype(index.dtype, np.integer):
            if index.ndim != 1:
                raise IndexError("row indices must be one-dimensional")
            if index.size == 0:
                return False, np.empty((0, 2), dtype=np.uint64)
            if np.any(index >= num_data) or (
                np.issubdtype(index.dtype, np.signedinteger)
                and np.any(index < -num_data)
            ):
                raise IndexError("row index out of range")
            ids = index.astype(np.intp, copy=True)
            ids[ids < 0] += num_data
            return False, contiguous_ranges(ids)
        case slice():
            start, stop, step = index.indices(num_data)
            if step == 1:
                return (
                    False,
                    np.array([(start, stop)], dtype=np.int64)
                    if stop > start
                    else np.empty((0, 2), dtype=np.int64),
                )
            return (
                False,
                np.array(
                    [(i, i + 1) for i in range(start, stop, step)], dtype=np.int64
                ).reshape(-1, 2),
            )
        case _:
            raise IndexError(f"Unsupported pattern index: {type(index).__name__}")


def read_indexed_array(
    file_obj: BufferedReader | BytesIO,
    index_ranges: INDEX_ARRAY,
    offsets: INDEX_ARRAY,
    current_offset: int,
) -> tuple[npt.NDArray[np.int32], int]:
    if len(index_ranges) == 1:
        start, stop = index_ranges[0]
        stop = offsets[stop]
        start = offsets[start]
        count = int(stop - start)
        file_obj.seek(I4 * (int(start) - current_offset), os.SEEK_CUR)
        raw = file_obj.read(count * I4)
        if len(raw) != count * I4:
            raise ValueError("Truncated EMC pattern file")
        return np.frombuffer(raw, count=count, dtype="<i4").astype(
            np.int32, copy=False
        ), int(stop) - int(offsets[-1])

    arrays = []
    for start, stop in index_ranges:
        stop = offsets[stop]
        start = offsets[start]
        count = int(stop - start)
        file_obj.seek(I4 * (int(start) - current_offset), os.SEEK_CUR)
        raw = file_obj.read(count * I4)
        if len(raw) != count * I4:
            raise ValueError("Truncated EMC pattern file")
        arrays.append(
            np.frombuffer(raw, count=count, dtype="<i4").astype(np.int32, copy=False)
        )
        current_offset = int(stop)
    return (
        np.concatenate(arrays) if arrays else np.array([], np.int32),
        int(current_offset) - int(offsets[-1]),
    )


def read_patterns(
    file_obj: BufferedReader | BytesIO,
    index_ranges: INDEX_ARRAY,
    ones_idx: INDEX_ARRAY,
    multi_idx: INDEX_ARRAY,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    seek_start = PatternsSOneEMC.HEADER_BYTES + I4 * (len(ones_idx) - 1) * 2
    file_obj.seek(seek_start)
    place_ones, current_offset = read_indexed_array(file_obj, index_ranges, ones_idx, 0)
    place_multi, current_offset = read_indexed_array(
        file_obj, index_ranges, multi_idx, current_offset
    )
    count_multi, current_offset = read_indexed_array(
        file_obj, index_ranges, multi_idx, current_offset
    )
    file_obj.seek(I4 * (-current_offset), os.SEEK_CUR)
    if file_obj.read(1):
        total = seek_start + place_ones.nbytes + place_multi.nbytes + count_multi.nbytes
        _log.error(
            "START: %d, place_ones: %d, place_multi: %d, "
            "count_multi: %d, total=%d; offset: %d",
            seek_start,
            place_ones.nbytes,
            place_multi.nbytes,
            count_multi.nbytes,
            total,
            current_offset,
        )
        raise ValueError("Error when parsing")
    return place_ones.view("u4"), place_multi.view("u4"), count_multi


class PatternsSOneFile:
    """Base interface for file-backed sparse photon patterns.

    ``FileBackedEMCPatterns`` is the descriptive public alias. Use
    ``open_patterns`` to select a concrete reader by file format.
    Metadata is available without loading all events. Per-pattern counts
    and cumulative offsets are initialized when needed.

    Integer indexing returns a dense pattern; slicing or NumPy row index
    arrays materializes only the selected sparse events. Reading an event
    property such as ``place_ones`` loads its entire event stream.
    """

    ndim: int = 2
    num_data: int
    num_pix: int
    _init_idx: bool

    @property
    def num_patterns(self) -> int:
        """Number of patterns exposed by this source."""
        return self.num_data

    @property
    def num_pixels(self) -> int:
        """Number of pixels in each pattern."""
        return self.num_pix

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_data, self.num_pix)

    @property
    def ones(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        return self._ones

    @property
    def multi(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        return self._multi

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        raise NotImplementedError()

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        raise NotImplementedError()

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        raise NotImplementedError()

    @property
    def nbytes(self) -> int:
        self.init_idx()
        return int(
            self.ones.nbytes
            + self.multi.nbytes
            + (self.ones.sum() + self.multi.sum() * 2) * I4
        )

    def sparsity(self) -> float:
        if self.num_data == 0 or self.num_pix == 0:
            return 0.0
        return float(self.nbytes / (4 * self.num_data * self.num_pix))

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        raise NotImplementedError()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        raise NotImplementedError()

    def init_idx(self) -> None:
        if self._init_idx:
            return
        self._ones, self._multi = self._read_ones_multi()
        self.ones_idx = _count_offsets(self._ones)
        self.multi_idx = _count_offsets(self._multi)
        self._init_idx = True

    def _ensure_offsets_loaded(self) -> None:
        self.init_idx()

    @property
    def ones_offsets(self) -> npt.NDArray[np.uint64]:
        self._ensure_offsets_loaded()
        return self.ones_idx

    @property
    def multi_offsets(self) -> npt.NDArray[np.uint64]:
        self._ensure_offsets_loaded()
        return self.multi_idx

    def __repr__(self) -> str:
        return f"""PatternFile(1-sparse) <{hex(id(self))}>
  Number of patterns: {self.num_data}
  Number of pixels: {self.num_pix}
  Size: {pretty_size(self.nbytes)}
  Sparsity: {self.sparsity() * 100:.2f} %
"""

    def _repr_html_(self) -> str:
        summary = {
            "patterns": self.num_data,
            "pixels": self.num_pix,
            "size": pretty_size(self.nbytes),
        }
        return html_card(
            "Pattern file",
            summary,
            details={
                "type": self.__class__.__name__,
                "source": str(getattr(self, "_fn", "<memory>")),
            },
            bars=(("sparsity", self.sparsity() * 100, "#0f766e"),),
        )

    def __len__(self) -> int:
        return int(self.num_data)

    @overload
    def __getitem__(self, index: tuple[INT_INDEX, INT_INDEX]) -> np.int32: ...

    @overload
    def __getitem__(
        self,
        index: INT_INDEX | tuple[INT_INDEX, TRANGE] | tuple[TRANGE, INT_INDEX],
    ) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE | tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: INT_INDEX
        | TRANGE
        | tuple[INT_INDEX, INT_INDEX]
        | tuple[INT_INDEX, TRANGE]
        | tuple[TRANGE, INT_INDEX]
        | tuple[TRANGE, TRANGE],
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
        if isinstance(index, tuple) and len(index) == 2:
            return _dispatch_2d_file_getitem(self, index)

        is_scalar, index_ranges = _normalize_row_index_ranges(
            cast(int | np.integer[Any] | TRANGE, index), self.num_data
        )
        place_ones, place_multi, count_multi = self._read_patterns(index_ranges)
        if is_scalar:
            return _materialize_dense_pattern(
                self.num_pix, place_ones, place_multi, count_multi
            )
        return PatternsSOne(
            self.num_pix,
            self.ones[index],
            self.multi[index],
            place_ones,
            place_multi,
            count_multi,
        )

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        """Read one pattern as sparse event arrays with absolute positions.

        ``index`` is a zero-based index in ``[0, num_patterns)``. Returns a
        ``SparsePattern`` without allocating a dense array of pixel counts.
        """
        idx = int(index)
        if idx < 0:
            idx += self.num_data
        if idx < 0 or idx >= self.num_data:
            raise IndexError(f"{index}")
        place_ones, place_multi, count_multi = self._read_patterns(
            np.array([[idx, idx + 1]])
        )
        return SPARSE_PATTERN(self.num_pix, place_ones, place_multi, count_multi)

    def sum(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        dtype: npt.DTypeLike | None = None,
        chunk_size: int | None = None,
    ) -> int | float | npt.NDArray[Any]:
        if axis is not None:
            if isinstance(axis, (bool, np.bool_)) or not isinstance(
                axis, (int, np.integer)
            ):
                raise TypeError(
                    f"axis must be an integer or None, not {type(axis).__name__}"
                )
            axis = int(axis)
            if axis < 0:
                axis += 2
            if axis not in (0, 1):
                raise ValueError(f"Do not support axis={axis}.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.num_data == 0:
            empty = PatternsSOne(
                self.num_pix,
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.int32),
            )
            return cast(
                int | float | npt.NDArray[Any],
                empty.sum(axis=axis, keepdims=keepdims, dtype=dtype),
            )
        if chunk_size is None:
            bytes_per_pat = max(1, self.nbytes // max(1, self.num_data))
            chunk_size = max(8, 100_000_000 // bytes_per_pat)
        acc_dtype = np.empty(0, dtype=np.int32).sum(dtype=dtype).dtype
        sums = [
            self[i : min(i + chunk_size, self.num_data)].sum(axis=axis, dtype=acc_dtype)
            for i in range(0, self.num_data, chunk_size)
        ]
        if axis == 1:
            ans_1 = np.concatenate(sums)
            return ans_1.reshape(-1, 1) if keepdims else ans_1
        if axis == 0:
            ans = np.sum(cast(list[npt.NDArray[Any]], sums), axis=0, dtype=acc_dtype)
            return cast(npt.NDArray[Any], ans.reshape(1, -1) if keepdims else ans)
        total = np.sum(np.asarray(sums, dtype=acc_dtype), dtype=acc_dtype)
        if keepdims:
            return np.asarray(total, dtype=acc_dtype).reshape((1, 1))
        return cast(float | int, total)


def _emc_scalar_sum_to_dtype(
    raw_total: int, acc_dtype: np.dtype[Any], keepdims: bool
) -> int | float | npt.NDArray[Any]:
    if np.issubdtype(acc_dtype, np.integer):
        info = np.iinfo(acc_dtype)
        if info.min <= raw_total <= info.max:
            scalar_total: Any = acc_dtype.type(raw_total)
        else:
            scalar_total = np.array(
                raw_total & ((1 << 64) - 1), dtype=np.uint64
            ).astype(acc_dtype, copy=False)[()]
    else:
        scalar_total = acc_dtype.type(raw_total)
    if keepdims:
        return np.asarray(scalar_total, dtype=acc_dtype).reshape((1, 1))
    return cast(int | float, scalar_total)


def _emc_resolve_col_row_ids(
    ax0: Any, num_data: int
) -> tuple[bool, npt.NDArray[np.int64] | None]:
    if isinstance(ax0, slice):
        start, stop, step = ax0.indices(num_data)
        if start == 0 and stop == num_data and step == 1:
            return True, None
        return True, np.arange(start, stop, step, dtype=np.int64)
    if isinstance(ax0, np.ndarray) and np.issubdtype(ax0.dtype, np.bool_):
        if ax0.ndim != 1 or ax0.size != num_data:
            raise IndexError("Boolean row mask must match the number of patterns")
        return True, np.flatnonzero(ax0).astype(np.int64, copy=False)
    if isinstance(ax0, np.ndarray) and np.issubdtype(ax0.dtype, np.integer):
        if ax0.ndim != 1:
            raise IndexError("row indices must be one-dimensional")
        if ax0.dtype == np.uint64 and ax0.size > 0 and int(ax0.max()) >= num_data:
            raise IndexError("row index out of range")
        return True, np.ascontiguousarray(ax0, dtype=np.int64)
    raise IndexError(f"Unsupported pattern index: {type(ax0).__name__}")


def _emc_buffer_getitem(
    source: PatternsSOneFile,
    rs: Any,
    buf: Any,
    index: Any,
) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
    if isinstance(index, (bool, np.bool_)):
        raise IndexError("Boolean scalar indices are not supported")
    if isinstance(index, tuple) and len(index) == 2:
        ax0, ax1 = index
        if isinstance(ax0, (bool, np.bool_)) or isinstance(ax1, (bool, np.bool_)):
            raise IndexError("Boolean scalar indices are not supported")
        if _is_int_index(ax0):
            return _index_1d_pattern_columns(source[int(ax0)], ax1)
        if _is_int_index(ax1):
            col = int(ax1)
            if col < 0:
                col += source.num_pix
            if col < 0 or col >= source.num_pix:
                raise IndexError(
                    f"column index {ax1} is out of bounds for axis 1 "
                    f"with size {source.num_pix}"
                )
            _, row_ids = _emc_resolve_col_row_ids(ax0, source.num_data)
            source.init_idx()
            return cast(
                npt.NDArray[np.int32],
                rs.emc_extract_column(
                    buf,
                    source.num_data,
                    source.num_pix,
                    source.ones_idx,
                    source.multi_idx,
                    row_ids,
                    col,
                    emc_workers(),
                ),
            )
        return _dispatch_2d_file_getitem(source, index)

    if _is_int_index(index):
        idx = int(index)
        if idx < 0:
            idx += source.num_data
        if idx < 0 or idx >= source.num_data:
            raise IndexError("Pattern index out of range")
        source.init_idx()
        return cast(
            npt.NDArray[np.int32],
            rs.emc_read_dense_frame(
                buf,
                source.num_data,
                source.num_pix,
                source.ones_idx,
                source.multi_idx,
                idx,
            ),
        )

    if isinstance(index, slice):
        start, stop, step = index.indices(source.num_data)
        source.init_idx()
        workers = emc_workers()
        if step == 1:
            end = max(start, stop)
            po, pm, cm = rs.emc_read_slice(
                buf,
                source.num_data,
                source.ones_idx,
                source.multi_idx,
                start,
                end,
                workers,
            )
            ones_sub = source._ones[start:end].copy()
            multi_sub = source._multi[start:end].copy()
            oidx_sub = (
                source.ones_idx[: end + 1]
                if start == 0
                else source.ones_idx[start : end + 1] - source.ones_idx[start]
            )
            midx_sub = (
                source.multi_idx[: end + 1]
                if start == 0
                else source.multi_idx[start : end + 1] - source.multi_idx[start]
            )
            return PatternsSOne._from_raw_parts(
                source.num_pix,
                ones_sub,
                multi_sub,
                oidx_sub,
                midx_sub,
                po,
                pm,
                cm,
            )
        parts = rs.emc_read_strided(
            buf,
            source.num_data,
            source._ones,
            source._multi,
            source.ones_idx,
            source.multi_idx,
            start,
            stop,
            step,
            workers,
        )
        return PatternsSOne._from_raw_parts(source.num_pix, *parts)

    if isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.bool_):
        if index.ndim != 1 or index.size != source.num_data:
            raise IndexError("Boolean row mask must match the number of patterns")
        source.init_idx()
        mask_c = np.ascontiguousarray(index, dtype=np.bool_)
        parts = rs.emc_read_mask(
            buf,
            source.num_data,
            source._ones,
            source._multi,
            source.ones_idx,
            source.multi_idx,
            mask_c,
            emc_workers(),
        )
        return PatternsSOne._from_raw_parts(source.num_pix, *parts)

    if isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.integer):
        if index.ndim != 1:
            raise IndexError("row indices must be one-dimensional")
        if (
            index.dtype == np.uint64
            and index.size > 0
            and int(index.max()) >= source.num_data
        ):
            raise IndexError("row index out of range")
        source.init_idx()
        ids_i64 = np.ascontiguousarray(index, dtype=np.int64)
        parts = rs.emc_read_indexed(
            buf,
            source.num_data,
            source._ones,
            source._multi,
            source.ones_idx,
            source.multi_idx,
            ids_i64,
            emc_workers(),
        )
        return PatternsSOne._from_raw_parts(source.num_pix, *parts)

    raise IndexError(f"Unsupported pattern index: {type(index).__name__}")


def _emc_buffer_sum(
    source: PatternsSOneFile,
    rs: Any,
    buf: Any,
    axis: int | None,
    keepdims: bool,
    dtype: npt.DTypeLike | None,
) -> int | float | npt.NDArray[Any]:
    source.init_idx()
    total_ones = int(source.ones_idx[-1])
    total_multi = int(source.multi_idx[-1])
    acc_dtype = np.empty(0, dtype=np.int32).sum(dtype=dtype).dtype
    workers = emc_workers()
    if axis is None:
        raw_total = rs.emc_sum_total(
            buf, source.num_data, total_ones, total_multi, workers
        )
        return _emc_scalar_sum_to_dtype(int(raw_total), acc_dtype, keepdims)
    if axis == 1:
        row_i64 = rs.emc_sum_axis1(
            buf,
            source.num_data,
            source._ones,
            source.multi_idx,
            total_ones,
            workers,
        )
        row_ans = (
            row_i64
            if acc_dtype == np.dtype(np.int64)
            else row_i64.astype(acc_dtype, copy=False)
        )
        return cast(npt.NDArray[Any], row_ans[:, None] if keepdims else row_ans)
    col_i64 = rs.emc_sum_axis0(
        buf,
        source.num_data,
        source.num_pix,
        total_ones,
        total_multi,
        workers,
    )
    col_ans = (
        col_i64
        if acc_dtype == np.dtype(np.int64)
        else col_i64.astype(acc_dtype, copy=False)
    )
    return cast(npt.NDArray[Any], col_ans[None, :] if keepdims else col_ans)


class PatternsSOneEMC(PatternsSOneFile):
    """
    Represents a collection of patterns stored in an EMC-formatted binary file.

    This class provides an interface for reading patterns from `.emc` or `.bin`
    files, which are custom binary formats for storing sparse pattern data.
    It supports lazy loading, meaning that data is only read from the file
    when it is actually needed.

    Parameters
    ----------
    fn
        The path to the EMC file.

    Attributes
    ----------
    num_data : int
        The number of patterns in the file.
    num_pix : int
        The number of pixels in each pattern.
    """

    HEADER_BYTES = 1024

    def __init__(
        self, fn: str | Path, num_data: None | int = None, num_pix: None | int = None
    ):
        self._fn = Path(fn).resolve()
        if num_data is not None and num_pix is not None:
            self.num_data = int(num_data)
            self.num_pix = int(num_pix)
        else:
            entry = get_emc_cache_entry(self._fn, _resolved=True)
            if (
                entry is not None
                and entry.num_data is not None
                and entry.num_pix is not None
            ):
                self.num_data = entry.num_data
                self.num_pix = entry.num_pix
            else:
                with open(self._fn, "rb") as fin:
                    header_raw = fin.read(self.HEADER_BYTES)
                if len(header_raw) < self.HEADER_BYTES:
                    raise ValueError(f"Truncated EMC header in {self._fn}")
                header_vals = np.frombuffer(header_raw, dtype="<i4", count=2)
                self.num_data = int(header_vals[0])
                self.num_pix = int(header_vals[1])
                if entry is not None and self.num_data >= 0 and self.num_pix >= 0:
                    entry.num_data = self.num_data
                    entry.num_pix = self.num_pix
        if self.num_data < 0 or self.num_pix < 0:
            raise ValueError(
                f"Invalid EMC header dimensions ({self.num_data}, {self.num_pix})"
            )
        self.ndim = 2
        self._init_idx = False

    def _get_emc_entry(self) -> EmcAccelEntry | None:
        return get_emc_cache_entry(self._fn, _resolved=True)

    def init_idx(self) -> None:
        if self._init_idx:
            return
        entry = self._get_emc_entry()
        if (
            entry is not None
            and entry.ones is not None
            and entry.multi is not None
            and entry.ones_idx is not None
            and entry.multi_idx is not None
            and len(entry.ones) == self.num_data
        ):
            self._ones = entry.ones
            self._multi = entry.multi
            self.ones_idx = entry.ones_idx
            self.multi_idx = entry.multi_idx
            self._init_idx = True
            return

        rs = get_emc_rs()
        if rs is not None and entry is not None:
            with entry.borrow_mmap() as mm:
                if mm is not None:
                    ones, multi, ones_idx, multi_idx = rs.emc_read_counts_and_offsets(
                        mm, self.num_data
                    )
                    ones_idx.flags.writeable = False
                    multi_idx.flags.writeable = False
                    self._ones = ones
                    self._multi = multi
                    self.ones_idx = ones_idx
                    self.multi_idx = multi_idx
                    if entry.num_data is None or entry.num_data == self.num_data:
                        entry.ones = ones
                        entry.multi = multi
                        entry.ones_idx = ones_idx
                        entry.multi_idx = multi_idx
                    self._init_idx = True
                    return

        self._ones, self._multi = self._read_ones_multi()
        self.ones_idx = _count_offsets(self._ones)
        self.multi_idx = _count_offsets(self._multi)
        if entry is not None and (
            entry.num_data is None or entry.num_data == self.num_data
        ):
            entry.ones = self._ones
            entry.multi = self._multi
            entry.ones_idx = self.ones_idx
            entry.multi_idx = self.multi_idx
        self._init_idx = True

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            ones = np.fromfile(fin, dtype="<u4", count=self.num_data).astype(
                np.uint32, copy=False
            )
            multi = np.fromfile(fin, dtype="<u4", count=self.num_data).astype(
                np.uint32, copy=False
            )
        if len(ones) != self.num_data or len(multi) != self.num_data:
            raise ValueError(f"Truncated EMC pattern count arrays in {self._fn}")
        return ones, multi

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        with self._fn.open("rb") as fin:
            return read_patterns(fin, index_ranges, self.ones_idx, self.multi_idx)

    @overload
    def __getitem__(self, index: tuple[INT_INDEX, INT_INDEX]) -> np.int32: ...

    @overload
    def __getitem__(
        self,
        index: INT_INDEX | tuple[INT_INDEX, TRANGE] | tuple[TRANGE, INT_INDEX],
    ) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE | tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: INT_INDEX
        | TRANGE
        | tuple[INT_INDEX, INT_INDEX]
        | tuple[INT_INDEX, TRANGE]
        | tuple[TRANGE, INT_INDEX]
        | tuple[TRANGE, TRANGE],
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
        rs = get_emc_rs()
        if rs is not None:
            entry = self._get_emc_entry()
            if entry is not None:
                with entry.borrow_mmap() as mm:
                    if mm is not None:
                        return _emc_buffer_getitem(self, rs, mm, index)
        return super().__getitem__(index)

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        rs = get_emc_rs()
        if rs is not None:
            idx = int(index)
            if idx < 0:
                idx += self.num_data
            if idx < 0 or idx >= self.num_data:
                raise IndexError(f"{index}")
            entry = self._get_emc_entry()
            if entry is not None:
                self.init_idx()
                with entry.borrow_mmap() as mm:
                    if mm is not None:
                        po, pm, cm = rs.emc_read_sparse_frame(
                            mm, self.num_data, self.ones_idx, self.multi_idx, idx
                        )
                        return SPARSE_PATTERN(self.num_pix, po, pm, cm)
        return super().sparse_pattern(index)

    def sum(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        dtype: npt.DTypeLike | None = None,
        chunk_size: int | None = None,
    ) -> int | float | npt.NDArray[Any]:
        norm_axis: int | None = None
        if axis is not None:
            if isinstance(axis, (bool, np.bool_)) or not isinstance(
                axis, (int, np.integer)
            ):
                raise TypeError(
                    f"axis must be an integer or None, not {type(axis).__name__}"
                )
            norm_axis = int(axis)
            if norm_axis < 0:
                norm_axis += 2
            if norm_axis not in (0, 1):
                raise ValueError(f"Do not support axis={norm_axis}.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.num_data == 0:
            return super().sum(
                axis=axis, keepdims=keepdims, dtype=dtype, chunk_size=chunk_size
            )
        rs = get_emc_rs()
        if rs is not None:
            entry = self._get_emc_entry()
            if entry is not None:
                with entry.borrow_mmap() as mm:
                    if mm is not None:
                        return _emc_buffer_sum(self, rs, mm, norm_axis, keepdims, dtype)
        return super().sum(
            axis=axis, keepdims=keepdims, dtype=dtype, chunk_size=chunk_size
        )

    def open(self) -> PatternsSOneEMCReadBuffer:
        """Open a persistent reader with its own file handle.

        Use ``with source.open() as reader:`` to reuse the handle across
        accesses and close it on exit. Without a context manager, call
        ``reader.close()`` explicitly when finished.
        """
        return PatternsSOneEMCReadBuffer(self._fn)

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        expected = int(self.ones_idx[-1])
        entry = self._get_emc_entry()
        if get_emc_rs() is not None and entry is not None:
            with entry.borrow_mmap() as mm:
                if mm is not None:
                    byte_off = 1024 + I4 * self.num_data * 2
                    if len(mm) < byte_off + I4 * expected:
                        raise ValueError(
                            f"Truncated EMC place_ones array in {self._fn}"
                        )
                    return np.frombuffer(
                        mm, dtype="<u4", count=expected, offset=byte_off
                    ).astype(np.uint32, copy=True)
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            fin.seek(I4 * self.num_data * 2, os.SEEK_CUR)
            arr = np.fromfile(fin, dtype="<u4", count=expected).astype(
                np.uint32, copy=False
            )
        if len(arr) != expected:
            raise ValueError(f"Truncated EMC place_ones array in {self._fn}")
        return arr

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        expected = int(self.multi_idx[-1])
        entry = self._get_emc_entry()
        if get_emc_rs() is not None and entry is not None:
            with entry.borrow_mmap() as mm:
                if mm is not None:
                    byte_off = 1024 + I4 * (self.num_data * 2 + int(self.ones_idx[-1]))
                    if len(mm) < byte_off + I4 * expected:
                        raise ValueError(
                            f"Truncated EMC place_multi array in {self._fn}"
                        )
                    return np.frombuffer(
                        mm, dtype="<u4", count=expected, offset=byte_off
                    ).astype(np.uint32, copy=True)
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            fin.seek(I4 * (self.num_data * 2 + int(self.ones_idx[-1])), os.SEEK_CUR)
            arr = np.fromfile(fin, dtype="<u4", count=expected).astype(
                np.uint32, copy=False
            )
        if len(arr) != expected:
            raise ValueError(f"Truncated EMC place_multi array in {self._fn}")
        return arr

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        self.init_idx()
        expected = int(self.multi_idx[-1])
        entry = self._get_emc_entry()
        if get_emc_rs() is not None and entry is not None:
            with entry.borrow_mmap() as mm:
                if mm is not None:
                    byte_off = 1024 + I4 * (
                        self.num_data * 2
                        + int(self.ones_idx[-1])
                        + int(self.multi_idx[-1])
                    )
                    if len(mm) < byte_off + I4 * expected:
                        raise ValueError(
                            f"Truncated EMC count_multi array in {self._fn}"
                        )
                    return np.frombuffer(
                        mm, dtype="<i4", count=expected, offset=byte_off
                    ).astype(np.int32, copy=True)
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            fin.seek(
                I4
                * (
                    self.num_data * 2 + int(self.ones_idx[-1]) + int(self.multi_idx[-1])
                ),
                os.SEEK_CUR,
            )
            arr = np.fromfile(fin, dtype="<i4", count=expected).astype(
                np.int32, copy=False
            )
        if len(arr) != expected:
            raise ValueError(f"Truncated EMC count_multi array in {self._fn}")
        return arr


class _PatternsSOneBytes(PatternsSOneFile):
    HEADER_BYTES = 1024

    def __init__(self, fn: BytesIO):
        self._fn = fn
        self._fn.seek(0)
        header_raw = self._fn.read(self.HEADER_BYTES)
        if len(header_raw) < self.HEADER_BYTES:
            raise ValueError("Truncated EMC header in byte stream")
        header_vals = np.frombuffer(header_raw, dtype="<i4", count=2)
        self.num_data = int(header_vals[0])
        self.num_pix = int(header_vals[1])
        if self.num_data < 0 or self.num_pix < 0:
            raise ValueError(
                f"Invalid EMC header dimensions ({self.num_data}, {self.num_pix})"
            )
        self.ndim = 2
        self._init_idx = False

    def init_idx(self) -> None:
        if self._init_idx:
            return
        rs = get_emc_rs()
        if rs is not None:
            with self._fn.getbuffer() as raw_mv:
                ro_mv = memoryview(raw_mv).toreadonly()
                ones, multi, ones_idx, multi_idx = rs.emc_read_counts_and_offsets(
                    ro_mv, self.num_data
                )
            ones_idx.flags.writeable = False
            multi_idx.flags.writeable = False
            self._ones = ones
            self._multi = multi
            self.ones_idx = ones_idx
            self.multi_idx = multi_idx
            self._init_idx = True
            return
        super().init_idx()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        self._fn.seek(1024)
        expected_bytes = I4 * self.num_data
        raw_ones = self._fn.read(expected_bytes)
        raw_multi = self._fn.read(expected_bytes)
        if len(raw_ones) != expected_bytes or len(raw_multi) != expected_bytes:
            raise ValueError("Truncated EMC pattern count arrays in byte stream")
        return (
            np.frombuffer(raw_ones, dtype="<u4", count=self.num_data).astype(
                np.uint32, copy=False
            ),
            np.frombuffer(raw_multi, dtype="<u4", count=self.num_data).astype(
                np.uint32, copy=False
            ),
        )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        self._fn.seek(0)
        return read_patterns(self._fn, index_ranges, self.ones_idx, self.multi_idx)

    @overload
    def __getitem__(self, index: tuple[INT_INDEX, INT_INDEX]) -> np.int32: ...

    @overload
    def __getitem__(
        self,
        index: INT_INDEX | tuple[INT_INDEX, TRANGE] | tuple[TRANGE, INT_INDEX],
    ) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE | tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: INT_INDEX
        | TRANGE
        | tuple[INT_INDEX, INT_INDEX]
        | tuple[INT_INDEX, TRANGE]
        | tuple[TRANGE, INT_INDEX]
        | tuple[TRANGE, TRANGE],
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
        rs = get_emc_rs()
        if rs is not None:
            with self._fn.getbuffer() as raw_mv:
                ro_mv = memoryview(raw_mv).toreadonly()
                return _emc_buffer_getitem(self, rs, ro_mv, index)
        return super().__getitem__(index)

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        rs = get_emc_rs()
        if rs is not None:
            idx = int(index)
            if idx < 0:
                idx += self.num_data
            if idx < 0 or idx >= self.num_data:
                raise IndexError(f"{index}")
            self.init_idx()
            with self._fn.getbuffer() as raw_mv:
                ro_mv = memoryview(raw_mv).toreadonly()
                po, pm, cm = rs.emc_read_sparse_frame(
                    ro_mv, self.num_data, self.ones_idx, self.multi_idx, idx
                )
            return SPARSE_PATTERN(self.num_pix, po, pm, cm)
        return super().sparse_pattern(index)

    def sum(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        dtype: npt.DTypeLike | None = None,
        chunk_size: int | None = None,
    ) -> int | float | npt.NDArray[Any]:
        norm_axis: int | None = None
        if axis is not None:
            if isinstance(axis, (bool, np.bool_)) or not isinstance(
                axis, (int, np.integer)
            ):
                raise TypeError(
                    f"axis must be an integer or None, not {type(axis).__name__}"
                )
            norm_axis = int(axis)
            if norm_axis < 0:
                norm_axis += 2
            if norm_axis not in (0, 1):
                raise ValueError(f"Do not support axis={norm_axis}.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.num_data == 0:
            return super().sum(
                axis=axis, keepdims=keepdims, dtype=dtype, chunk_size=chunk_size
            )
        rs = get_emc_rs()
        if rs is not None:
            with self._fn.getbuffer() as raw_mv:
                ro_mv = memoryview(raw_mv).toreadonly()
                return _emc_buffer_sum(self, rs, ro_mv, norm_axis, keepdims, dtype)
        return super().sum(
            axis=axis, keepdims=keepdims, dtype=dtype, chunk_size=chunk_size
        )

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        expected = int(self.ones_idx[-1])
        expected_bytes = I4 * expected
        self._fn.seek(1024 + I4 * self.num_data * 2)
        raw = self._fn.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError("Truncated EMC place_ones array in byte stream")
        return np.frombuffer(raw, dtype="<u4", count=expected).astype(
            np.uint32, copy=True
        )

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        expected = int(self.multi_idx[-1])
        expected_bytes = I4 * expected
        self._fn.seek(1024 + I4 * (self.num_data * 2 + int(self.ones_idx[-1])))
        raw = self._fn.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError("Truncated EMC place_multi array in byte stream")
        return np.frombuffer(raw, dtype="<u4", count=expected).astype(
            np.uint32, copy=True
        )

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        self.init_idx()
        expected = int(self.multi_idx[-1])
        expected_bytes = I4 * expected
        self._fn.seek(
            1024
            + I4
            * (self.num_data * 2 + int(self.ones_idx[-1]) + int(self.multi_idx[-1]))
        )
        raw = self._fn.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError("Truncated EMC count_multi array in byte stream")
        return np.frombuffer(raw, dtype="<i4", count=expected).astype(
            np.int32, copy=True
        )


class PatternsSOneEMCReadBuffer(PatternsSOneEMC):
    def __init__(self, fn: str | Path):
        self._file_handle: BufferedReader | None = None
        self._private_entry: EmcAccelEntry | None = None
        self._closed = False
        try:
            super().__init__(fn)
            self._file_handle = self._fn.open("rb")
            self._private_entry = create_private_emc_entry(self._fn)
            self._private_entry.num_data = self.num_data
            self._private_entry.num_pix = self.num_pix
        except BaseException:
            self.close()
            raise

    def _ensure_file_handle(self) -> BufferedReader:
        if self._closed or self._file_handle is None:
            raise ValueError("I/O operation on closed EMC pattern reader")
        return self._file_handle

    def _get_emc_entry(self) -> EmcAccelEntry | None:
        if self._closed:
            raise ValueError("I/O operation on closed EMC pattern reader")
        return self._private_entry

    def __enter__(self) -> PatternsSOneEMCReadBuffer:
        return self

    def close(self) -> None:
        self._closed = True
        if self._private_entry is not None:
            self._private_entry.close()
            self._private_entry = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.close()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        fin = self._ensure_file_handle()
        fin.seek(1024)
        ones = np.fromfile(fin, dtype="<u4", count=self.num_data).astype(
            np.uint32, copy=False
        )
        multi = np.fromfile(fin, dtype="<u4", count=self.num_data).astype(
            np.uint32, copy=False
        )
        if len(ones) != self.num_data or len(multi) != self.num_data:
            raise ValueError(f"Truncated EMC pattern count arrays in {self._fn}")
        return ones, multi

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        fin = self._ensure_file_handle()
        self.init_idx()
        return read_patterns(fin, index_ranges, self.ones_idx, self.multi_idx)

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        self._ensure_file_handle()
        return super().place_ones

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        self._ensure_file_handle()
        return super().place_multi

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        self._ensure_file_handle()
        return super().count_multi

    def sum(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        dtype: npt.DTypeLike | None = None,
        chunk_size: int | None = None,
    ) -> int | float | npt.NDArray[Any]:
        self._ensure_file_handle()
        return super().sum(
            axis=axis, keepdims=keepdims, dtype=dtype, chunk_size=chunk_size
        )


def read_indexed_array_h5(
    dataset: h5py.Dataset,
    index_ranges: INDEX_ARRAY,
    offsets: INDEX_ARRAY,
) -> npt.NDArray[np.int32]:
    if len(index_ranges) == 1:
        start, stop = index_ranges[0]
        stop = offsets[stop]
        start = offsets[start]
        return cast(npt.NDArray[np.int32], dataset[start:stop])

    arrays = []
    for start, stop in index_ranges:
        stop = offsets[stop]
        start = offsets[start]
        arrays.append(dataset[start:stop])
    return (
        cast(npt.NDArray[np.int32], np.concatenate(arrays))
        if arrays
        else np.array([], np.int32)
    )


def _selected_counts(
    counts: npt.NDArray[np.uint32], index_ranges: INDEX_ARRAY
) -> npt.NDArray[np.uint32]:
    selected = [counts[start:stop] for start, stop in index_ranges]
    return np.concatenate(selected) if selected else np.array([], dtype=np.uint32)


def _as_nonneg_u32(values: np.ndarray) -> npt.NDArray[np.uint32]:
    if values.dtype == np.uint32:
        return values
    if np.issubdtype(values.dtype, np.signedinteger) and values.size > 0:
        if int(values.min()) < 0:
            raise ValueError("Negative unsigned count or position in HDF5 dataset")
    if values.dtype == np.int32:
        return values.view(np.uint32)
    return values.astype(np.uint32)


class PatternsSOneH5(PatternsSOneFile):
    """
    Represents a collection of patterns stored in an HDF5 file.

    This class provides an interface for reading patterns from HDF5 files that
    adhere to the `emcfile` storage format. It supports lazy loading of data
    to efficiently handle large datasets.

    Parameters
    ----------
    fn
        The path to the HDF5 file, which can be a string, `pathlib.Path`, or
        `H5Path` object.

    Attributes
    ----------
    num_data : int
        The number of patterns in the file.
    num_pix : int
        The number of pixels in each pattern.
    """

    def __init__(self, fn: str | Path | H5Path):
        self._fn = h5path(fn).resolve()
        entry = get_cache_entry(self._fn.fn, self._fn.gn)
        if (
            entry is not None
            and entry.version is not None
            and entry.num_data is not None
            and entry.num_pix is not None
            and entry.position_encoding is not None
        ):
            self.num_data = entry.num_data
            self.num_pix = entry.num_pix
            self.h5version = entry.version
            self.position_encoding = entry.position_encoding
        else:
            with self._fn.open_group() as (_, gp):
                self.num_data = int(cast(int, gp.attrs["num_data"]))
                self.num_pix = int(cast(int, gp.attrs["num_pix"]))
                self.h5version = str(gp.attrs.get("version", "2"))
                self.position_encoding = str(
                    gp.attrs.get("position_encoding", "absolute")
                )
            if entry is not None:
                entry.version = self.h5version
                entry.num_data = self.num_data
                entry.num_pix = self.num_pix
                entry.position_encoding = self.position_encoding
        if self.h5version != "2":
            raise ValueError(f"Unsupported HDF5 EMC format version {self.h5version!r}")
        if self.position_encoding not in {"absolute", "delta"}:
            raise ValueError(
                f"Unsupported HDF5 position encoding {self.position_encoding!r}"
            )
        self.ndim = 2
        self._init_idx = False

    def _get_accel_entry(self) -> H5AccelEntry | None:
        return get_cache_entry(self._fn.fn, self._fn.gn)

    def init_idx(self) -> None:
        if self._init_idx:
            return
        entry = self._get_accel_entry()
        if (
            entry is not None
            and entry.ones is not None
            and entry.multi is not None
            and entry.ones_idx is not None
            and entry.multi_idx is not None
        ):
            self._ones = entry.ones
            self._multi = entry.multi
            self.ones_idx = entry.ones_idx
            self.multi_idx = entry.multi_idx
            self._init_idx = True
            return
        self._ones, self._multi = self._read_ones_multi()
        self.ones_idx = _count_offsets(self._ones)
        self.multi_idx = _count_offsets(self._multi)
        if entry is not None:
            entry.ones = self._ones
            entry.multi = self._multi
            entry.ones_idx = self.ones_idx
            entry.multi_idx = self.multi_idx
        self._init_idx = True

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        idx = int(index)
        if idx < 0:
            idx += self.num_data
        if idx < 0 or idx >= self.num_data:
            raise IndexError(f"{index}")
        place_ones, place_multi, count_multi = self._read_patterns(
            np.array([[idx, idx + 1]])
        )
        return SPARSE_PATTERN(self.num_pix, place_ones, place_multi, count_multi)

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        with self._fn.open_group() as (fp, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            ensure_group_filters(gp)
            return (
                _as_nonneg_u32(cast(h5py.Dataset, gp["ones"])[...]),
                _as_nonneg_u32(cast(h5py.Dataset, gp["multi"])[...]),
            )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        entry = self._get_accel_entry()
        with LazyH5Group(self._fn, entry) as lazy_gp:
            if (
                self.position_encoding == "delta"
                and index_ranges.shape == (1, 2)
                and int(index_ranges[0, 0]) == 0
                and int(index_ranges[0, 1]) == self.num_data
            ):
                from ._h5_full_scan import full_scan

                result = full_scan(lazy_gp, self.ones_idx, self.multi_idx)
                if result is not None:
                    return result
            elif self.position_encoding == "delta":
                from ._h5_indexed import indexed_read

                result = indexed_read(
                    lazy_gp, index_ranges, self.ones_idx, self.multi_idx
                )
                if result is not None:
                    return result
            gp = lazy_gp.real_group()
            assert isinstance(gp, (h5py.Group, h5py.File))
            ensure_group_filters(gp)
            place_ones = read_indexed_array_h5(
                cast(h5py.Dataset, gp["place_ones"]), index_ranges, self.ones_idx
            )
            place_multi = read_indexed_array_h5(
                cast(h5py.Dataset, gp["place_multi"]), index_ranges, self.multi_idx
            )
            count_multi = read_indexed_array_h5(
                cast(h5py.Dataset, gp["count_multi"]), index_ranges, self.multi_idx
            )
            if self.position_encoding == "delta":
                return (
                    decode_pattern_local_delta(
                        place_ones, _selected_counts(self.ones, index_ranges)
                    ),
                    decode_pattern_local_delta(
                        place_multi, _selected_counts(self.multi, index_ranges)
                    ),
                    count_multi,
                )
            return place_ones.view("u4"), place_multi.view("u4"), count_multi

    def open(self) -> PatternsSOneH5ReadBuffer:
        """Open a persistent reader with its own file handle.

        Use ``with source.open() as reader:`` to reuse the handle across
        accesses and close it on exit. Without a context manager, call
        ``reader.close()`` explicitly when finished.
        """
        return PatternsSOneH5ReadBuffer(self._fn)

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            ensure_group_filters(gp)
            values = cast(h5py.Dataset, gp["place_ones"])[...]
            return (
                decode_pattern_local_delta(values, self.ones)
                if self.position_encoding == "delta"
                else _as_nonneg_u32(values)
            )

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            ensure_group_filters(gp)
            values = cast(h5py.Dataset, gp["place_multi"])[...]
            return (
                decode_pattern_local_delta(values, self.multi)
                if self.position_encoding == "delta"
                else _as_nonneg_u32(values)
            )

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            ensure_group_filters(gp)
            return cast(h5py.Dataset, gp["count_multi"])[...]


class PatternsSOneH5ReadBuffer(PatternsSOneH5):
    def __init__(self, fn: str | Path | H5Path):
        self._fn = h5path(fn).resolve()
        self._file_handle: h5py.File | None = None
        self._private_entry: H5AccelEntry | None = None
        try:
            self._file_handle = h5py.File(self._fn.fn, "r")
            self._private_entry = create_private_entry(self._fn.fn, self._fn.gn)
            gp = self._file_handle[self._fn.gn]
            assert isinstance(gp, (h5py.Group, h5py.File))
            self.num_data = int(cast(int, gp.attrs["num_data"]))
            self.num_pix = int(cast(int, gp.attrs["num_pix"]))
            self.h5version = str(gp.attrs.get("version", "2"))
            self.position_encoding = str(gp.attrs.get("position_encoding", "absolute"))
            if self.h5version != "2":
                raise ValueError(
                    f"Unsupported HDF5 EMC format version {self.h5version!r}"
                )
            if self.position_encoding not in {"absolute", "delta"}:
                raise ValueError(
                    f"Unsupported HDF5 position encoding {self.position_encoding!r}"
                )
            self._private_entry.version = self.h5version
            self._private_entry.num_data = self.num_data
            self._private_entry.num_pix = self.num_pix
            self._private_entry.position_encoding = self.position_encoding
            self.ndim = 2
            self._init_idx = False
        except BaseException:
            self.close()
            raise

    def _get_accel_entry(self) -> H5AccelEntry | None:
        return self._private_entry

    def _ensure_file_handle(self) -> h5py.File:
        if self._file_handle is None:
            raise ValueError("I/O operation on closed HDF5 pattern reader")
        return self._file_handle

    def __enter__(self) -> PatternsSOneH5ReadBuffer:
        return self

    def close(self) -> None:
        if self._private_entry is not None:
            self._private_entry.close()
            self._private_entry = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.close()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        fp = self._ensure_file_handle()
        gp = fp[self._fn.gn]
        assert isinstance(gp, (h5py.Group, h5py.File))
        ensure_group_filters(gp)
        return (
            _as_nonneg_u32(cast(h5py.Dataset, gp["ones"])[...]),
            _as_nonneg_u32(cast(h5py.Dataset, gp["multi"])[...]),
        )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        entry = self._get_accel_entry()
        lazy_gp = LazyH5Group(self._fn, entry, reader=self)
        if (
            self.position_encoding == "delta"
            and index_ranges.shape == (1, 2)
            and int(index_ranges[0, 0]) == 0
            and int(index_ranges[0, 1]) == self.num_data
        ):
            from ._h5_full_scan import full_scan

            result = full_scan(lazy_gp, self.ones_idx, self.multi_idx)
            if result is not None:
                return result
        elif self.position_encoding == "delta":
            from ._h5_indexed import indexed_read

            result = indexed_read(lazy_gp, index_ranges, self.ones_idx, self.multi_idx)
            if result is not None:
                return result
        gp = lazy_gp.real_group()
        assert isinstance(gp, (h5py.Group, h5py.File))
        ensure_group_filters(gp)
        place_ones = read_indexed_array_h5(
            cast(h5py.Dataset, gp["place_ones"]), index_ranges, self.ones_idx
        )
        place_multi = read_indexed_array_h5(
            cast(h5py.Dataset, gp["place_multi"]), index_ranges, self.multi_idx
        )
        count_multi = read_indexed_array_h5(
            cast(h5py.Dataset, gp["count_multi"]), index_ranges, self.multi_idx
        )
        if self.position_encoding == "delta":
            return (
                decode_pattern_local_delta(
                    place_ones, _selected_counts(self.ones, index_ranges)
                ),
                decode_pattern_local_delta(
                    place_multi, _selected_counts(self.multi, index_ranges)
                ),
                count_multi,
            )
        return place_ones.view("u4"), place_multi.view("u4"), count_multi

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        fp = self._ensure_file_handle()
        gp = fp[self._fn.gn]
        assert isinstance(gp, (h5py.Group, h5py.File))
        ensure_group_filters(gp)
        values = cast(h5py.Dataset, gp["place_ones"])[...]
        return (
            decode_pattern_local_delta(values, self.ones)
            if self.position_encoding == "delta"
            else _as_nonneg_u32(values)
        )

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        fp = self._ensure_file_handle()
        gp = fp[self._fn.gn]
        assert isinstance(gp, (h5py.Group, h5py.File))
        ensure_group_filters(gp)
        values = cast(h5py.Dataset, gp["place_multi"])[...]
        return (
            decode_pattern_local_delta(values, self.multi)
            if self.position_encoding == "delta"
            else _as_nonneg_u32(values)
        )

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        fp = self._ensure_file_handle()
        gp = fp[self._fn.gn]
        assert isinstance(gp, (h5py.Group, h5py.File))
        ensure_group_filters(gp)
        return cast(h5py.Dataset, gp["count_multi"])[...]


class PatternsSOneH5V1(PatternsSOneFile):
    def __init__(self, fn: str | Path | H5Path):
        _log.warning(
            "This format has performance issue. `PatternsSOneH5` is recommended"
        )
        self._fn = h5path(fn)
        with self._fn.open_group() as (_, gp):
            self.num_data = len(cast(h5py.Dataset, gp["place_ones"]))
            self.num_pix = int(cast(h5py.Dataset, gp["num_pix"])[:][0])
        self.ndim = 2
        self._init_idx = False

    def init_idx(self) -> None:
        if self._init_idx:
            return
        with self._fn.open_group() as (_, gp):
            raw_ones = cast(h5py.Dataset, gp["place_ones"])[...]
            raw_multi = cast(h5py.Dataset, gp["place_multi"])[...]
            raw_count = cast(h5py.Dataset, gp["count_multi"])[...]
            if len(raw_ones) == 0:
                ones = np.empty(0, dtype=np.uint32)
                multi = np.empty(0, dtype=np.uint32)
                place_ones = np.empty(0, dtype=np.uint32)
                place_multi = np.empty(0, dtype=np.uint32)
                count_multi = np.empty(0, dtype=np.int32)
            else:
                ones = np.array([len(i) for i in raw_ones], np.uint32)
                place_ones = np.concatenate(raw_ones).astype(np.uint32, copy=False)
                multi = np.array([len(i) for i in raw_multi], np.uint32)
                place_multi = np.concatenate(raw_multi).astype(np.uint32, copy=False)
                count_multi = np.concatenate(raw_count).astype(np.int32, copy=False)
        self._patterns = PatternsSOne(
            self.num_pix, ones, multi, place_ones, place_multi, count_multi
        )
        self._init_idx = True

    @property
    def ones(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        return self._patterns.ones

    @property
    def multi(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        return self._patterns.multi

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        return self._patterns.place_ones

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        return self._patterns.place_multi

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        self.init_idx()
        return self._patterns.count_multi

    @overload
    def __getitem__(self, index: tuple[INT_INDEX, INT_INDEX]) -> np.int32: ...

    @overload
    def __getitem__(
        self,
        index: INT_INDEX | tuple[INT_INDEX, TRANGE] | tuple[TRANGE, INT_INDEX],
    ) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE | tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: INT_INDEX
        | TRANGE
        | tuple[INT_INDEX, INT_INDEX]
        | tuple[INT_INDEX, TRANGE]
        | tuple[TRANGE, INT_INDEX]
        | tuple[TRANGE, TRANGE],
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
        self.init_idx()
        return self._patterns[index]

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        self.init_idx()
        return self._patterns.sparse_pattern(index)


class PatternsSOneList(PatternsSOneFile):
    """
    Represents a collection of patterns from multiple source files.

    This class provides a unified interface for accessing patterns that are
    distributed across multiple files. It can handle a mixture of file formats
    (e.g., EMC and HDF5) and `PatternsSOne` objects, treating them as a single,
    large dataset.

    Parameters
    ----------
    pattern_list
        A sequence of paths to pattern files or `PatternsSOneBase` objects.

    Attributes
    ----------
    pattern_list : list[PatternsSOneBase]
        The list of pattern sources.
    num_data : int
        The total number of patterns in the collection.
    num_pix : int
        The number of pixels in each pattern.

    Raises
    ------
    ValueError
        If the pattern list is empty or if the number of pixels is
        inconsistent across the source files.
    """

    def __init__(
        self,
        pattern_list: Sequence[PATH_TYPE | PatternsSOneBase],
    ):
        self.pattern_list: list[PatternsSOneBase] = []
        indptr = [0]
        num_pix = None
        for f in pattern_list:
            # patn = f if isinstance(f, PatternsSOneFile) else file_patterns(f)
            match f:
                case PatternsSOneFile() | PatternsSOneBase():
                    patn = f
                case _:
                    patn = file_patterns(f)
            if num_pix is None:
                num_pix = patn.num_pix
            elif num_pix != patn.num_pix:
                raise ValueError("Inconsistent number of pixels")
            self.pattern_list.append(patn)
            indptr.append(len(patn))
        if num_pix is None:
            raise ValueError("Empty pattern list is not allowed")
        self._indptr = np.cumsum(indptr)
        self.num_pix = int(num_pix)
        self.num_data = int(self._indptr[-1])
        self._init_idx = False

    @property
    def pattern_sources(self) -> list[PatternsSOneBase]:
        """Underlying pattern sources in collection order."""
        return self.pattern_list

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        return (
            np.concatenate([p.ones for p in self.pattern_list]).astype(
                np.uint32, copy=False
            ),
            np.concatenate([p.multi for p in self.pattern_list]).astype(
                np.uint32, copy=False
            ),
        )

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        arrays = [p.place_ones for p in self.pattern_list]
        return (
            np.concatenate(arrays).astype(np.uint32, copy=False)
            if arrays
            else np.empty(0, dtype=np.uint32)
        )

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        arrays = [p.place_multi for p in self.pattern_list]
        return (
            np.concatenate(arrays).astype(np.uint32, copy=False)
            if arrays
            else np.empty(0, dtype=np.uint32)
        )

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        arrays = [p.count_multi for p in self.pattern_list]
        return (
            np.concatenate(arrays).astype(np.int32, copy=False)
            if arrays
            else np.empty(0, dtype=np.int32)
        )

    @overload
    def __getitem__(self, index: tuple[INT_INDEX, INT_INDEX]) -> np.int32: ...

    @overload
    def __getitem__(
        self,
        index: INT_INDEX | tuple[INT_INDEX, TRANGE] | tuple[TRANGE, INT_INDEX],
    ) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE | tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: INT_INDEX
        | TRANGE
        | tuple[INT_INDEX, INT_INDEX]
        | tuple[INT_INDEX, TRANGE]
        | tuple[TRANGE, INT_INDEX]
        | tuple[TRANGE, TRANGE],
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
        if isinstance(index, (bool, np.bool_)):
            raise IndexError("Boolean scalar indices are not supported")
        match index:
            case tuple() as axes if len(axes) == 2:
                return _dispatch_2d_file_getitem(self, axes)
            case int() | np.integer():
                pidx = int(index)
                if pidx < 0:
                    pidx += self.num_data
                if pidx < 0 or pidx >= self.num_data:
                    raise IndexError("index out of range")
                lidx = np.digitize(pidx, self._indptr[1:])
                return self.pattern_list[lidx][int(pidx - self._indptr[lidx])]
            case _:
                selected = cast(
                    npt.NDArray[np.int32 | np.int64],
                    np.arange(self.num_data)[index],
                )
                if len(selected) == 0:
                    return PatternsSOne(
                        self.num_pix,
                        np.array([], dtype=np.uint32),
                        np.array([], dtype=np.uint32),
                        np.array([], dtype=np.uint32),
                        np.array([], dtype=np.uint32),
                        np.array([], dtype=np.int32),
                    )
                lids = np.digitize(selected, self._indptr[1:])
                source_order = np.argsort(lids, kind="stable")
                grouped_lids = lids[source_order]
                grouped_indices = selected[source_order]

                patns: list[PatternsSOne] = []
                starts = np.r_[
                    0, np.flatnonzero(grouped_lids[1:] != grouped_lids[:-1]) + 1
                ]
                ends = np.r_[starts[1:], len(grouped_indices)]
                for start, end in zip(starts, ends):
                    i = grouped_lids[start]
                    patns.append(
                        self.pattern_list[i][
                            grouped_indices[start:end] - self._indptr[i]
                        ]
                    )
                grouped = cast(PatternsSOne, np.concatenate(patns))
                if np.all(source_order == np.arange(len(source_order))):
                    return grouped
                return grouped[np.argsort(source_order)]

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        pidx = int(index)
        if pidx < 0:
            pidx += self.num_data
        if pidx < 0 or pidx >= self.num_data:
            raise IndexError(f"{index}")
        lidx = int(np.digitize(pidx, self._indptr[1:]))
        local_idx = int(pidx - self._indptr[lidx])
        source = self.pattern_list[lidx]
        if hasattr(source, "sparse_pattern"):
            return cast(SPARSE_PATTERN, cast(Any, source).sparse_pattern(local_idx))
        return cast(PatternsSOne, source[local_idx : local_idx + 1]).sparse_pattern(0)

    def write(
        self,
        path: PATH_TYPE | io.BytesIO,
        *,
        h5version: str = "2",
        overwrite: bool = False,
        compression: None | int | str = None,
        compression_opts: Any = None,
        shuffle: bool = False,
        position_encoding: str = "absolute",
        check_sorted: bool = False,
        hdf5_version: str | None = None,
    ) -> None:
        return write_patterns(
            self.pattern_list,
            path,
            h5version=h5version,
            overwrite=overwrite,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            position_encoding=position_encoding,
            check_sorted=check_sorted,
            hdf5_version=hdf5_version,
        )


class PatternsSOneParquet(PatternsSOneFile):
    """Represents a collection of patterns stored in an Apache Parquet file.

    ``ParquetPatternFile`` is the descriptive public alias. Supports lazy
    loading of metadata and per-pattern counts, direct single-frame reads,
    and parallel row-group decoding for full-scan and indexed reads.
    """

    def __init__(self, fn: str | Path):
        from ._parquet import read_parquet_metadata

        self._fn = Path(fn).resolve()
        self._cache: dict[str, Any] = {}
        meta = read_parquet_metadata(self._fn, cache=self._cache)
        self.num_data = int(meta["num_data"])
        self.num_pix = int(meta["num_pix"])
        self.pqversion = str(meta["version"])
        self.position_encoding = str(meta["position_encoding"])
        self.rg_pats = int(meta["rg_pats"])
        self.shuffle = bool(meta["shuffle"])
        self.payload_codec = str(meta["payload_codec"])
        self._num_ones: int | None = meta.get("num_ones")
        self._num_multi: int | None = meta.get("num_multi")
        self._file_handle: Any = None
        self.ndim = 2
        self._init_idx = False

    @property
    def nbytes(self) -> int:
        if self._num_ones is not None and self._num_multi is not None:
            return int((self.num_data * 2 + self._num_ones + self._num_multi * 2) * I4)
        return super().nbytes

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        from ._parquet import read_parquet_ones_multi

        return read_parquet_ones_multi(
            self._fn, self.num_data, pf=self._file_handle, cache=self._cache
        )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        from ._parquet import read_parquet_patterns

        return read_parquet_patterns(
            self._fn,
            index_ranges,
            self.ones_idx if self._init_idx else None,
            self.multi_idx if self._init_idx else None,
            num_data=self.num_data,
            rg_pats=self.rg_pats,
            position_encoding=self.position_encoding,
            shuffle=self.shuffle,
            payload_codec=self.payload_codec,
            pf=self._file_handle,
            cache=self._cache,
        )

    @overload
    def __getitem__(self, index: tuple[INT_INDEX, INT_INDEX]) -> np.int32: ...

    @overload
    def __getitem__(
        self,
        index: INT_INDEX | tuple[INT_INDEX, TRANGE] | tuple[TRANGE, INT_INDEX],
    ) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE | tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: INT_INDEX
        | TRANGE
        | tuple[INT_INDEX, INT_INDEX]
        | tuple[INT_INDEX, TRANGE]
        | tuple[TRANGE, INT_INDEX]
        | tuple[TRANGE, TRANGE],
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne:
        from ._parquet import read_parquet_patterns_with_counts

        if isinstance(index, tuple) and len(index) == 2:
            return _dispatch_2d_file_getitem(self, index)

        is_scalar, index_ranges = _normalize_row_index_ranges(
            cast(int | np.integer[Any] | TRANGE, index), self.num_data
        )
        if is_scalar:
            place_ones, place_multi, count_multi = self._read_patterns(index_ranges)
            return _materialize_dense_pattern(
                self.num_pix, place_ones, place_multi, count_multi
            )

        sel_ones, sel_multi, place_ones, place_multi, count_multi = (
            read_parquet_patterns_with_counts(
                self._fn,
                index_ranges,
                self.ones_idx if self._init_idx else None,
                self.multi_idx if self._init_idx else None,
                num_data=self.num_data,
                rg_pats=self.rg_pats,
                position_encoding=self.position_encoding,
                shuffle=self.shuffle,
                payload_codec=self.payload_codec,
                pf=self._file_handle,
                cache=self._cache,
            )
        )
        return PatternsSOne(
            self.num_pix,
            sel_ones,
            sel_multi,
            place_ones,
            place_multi,
            count_multi,
        )

    def open(self) -> PatternsSOneParquetReadBuffer:
        """Open a persistent reader with its own Parquet file handle."""
        return PatternsSOneParquetReadBuffer(self._fn)

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        from ._parquet import read_parquet_column

        return read_parquet_column(
            self._fn,
            "place_ones",
            num_data=self.num_data,
            rg_pats=self.rg_pats,
            is_delta=(self.position_encoding == "delta"),
            shuffle=self.shuffle,
            payload_codec=self.payload_codec,
            dtype="u4",
            cache=self._cache,
        )

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        from ._parquet import read_parquet_column

        return read_parquet_column(
            self._fn,
            "place_multi",
            num_data=self.num_data,
            rg_pats=self.rg_pats,
            is_delta=(self.position_encoding == "delta"),
            shuffle=self.shuffle,
            payload_codec=self.payload_codec,
            dtype="u4",
            cache=self._cache,
        )

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        from ._parquet import read_parquet_column

        return read_parquet_column(
            self._fn,
            "count_multi",
            num_data=self.num_data,
            rg_pats=self.rg_pats,
            is_delta=False,
            shuffle=self.shuffle,
            payload_codec=self.payload_codec,
            dtype="i4",
            cache=self._cache,
        )


class PatternsSOneParquetReadBuffer(PatternsSOneParquet):
    def __init__(self, fn: str | Path):
        import mmap

        super().__init__(fn)
        self._fp: Any = None
        self._mm: mmap.mmap | None = None
        try:
            if self._fn.stat().st_size > 0:
                with self._fn.open("rb") as fp:
                    self._mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                self._cache["mm"] = self._mm
        except (OSError, ValueError):
            pass

    def __enter__(self) -> PatternsSOneParquetReadBuffer:
        return self

    def close(self) -> None:
        self._cache.clear()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.close()


def file_patterns(fn: Sequence[PATH_TYPE] | PATH_TYPE) -> PatternsSOneFile:
    if (
        isinstance(fn, (str, Path))
        and "::" not in str(fn)
        and glob.has_magic(str(fn))
        and not Path(fn).exists()
    ):
        matched = sorted(glob.glob(str(fn)))
        if not matched:
            raise FileNotFoundError(f"No pattern files matched {fn!r}")
        return PatternsSOneList(matched)
    if isinstance(fn, (tuple, list)):
        return PatternsSOneList(fn)
    assert isinstance(fn, (str, Path, H5Path))
    p = make_path(fn)
    if not isinstance(p, H5Path):
        if h5py.is_hdf5(p):
            p = h5path(p, "/")
    if not isinstance(p, H5Path):
        from ._parquet import is_parquet_file

        if is_parquet_file(p):
            return PatternsSOneParquet(p)
        return PatternsSOneEMC(p)
    entry = get_cache_entry(p.fn, p.gn)
    if entry is not None and entry.version is not None:
        version = entry.version
    else:
        with p.open_group() as (_, gp):
            version = str(gp.attrs.get("version", "1"))
            if entry is not None:
                entry.version = version
                if version == "2" and "num_data" in gp.attrs and "num_pix" in gp.attrs:
                    entry.num_data = int(cast(int, gp.attrs["num_data"]))
                    entry.num_pix = int(cast(int, gp.attrs["num_pix"]))
                    entry.position_encoding = str(
                        gp.attrs.get("position_encoding", "absolute")
                    )
    if version == "1":
        return PatternsSOneH5V1(p)
    if version == "2":
        return PatternsSOneH5(p)
    raise ValueError(f"Unsupported HDF5 EMC format version {version!r}")


def open_patterns(path: Sequence[PATH_TYPE] | PATH_TYPE) -> PatternsSOneFile:
    """Open one or more pattern files without loading all event arrays.

    Parameters
    ----------
    path : path-like or sequence of paths
        EMC binary, HDF5, or Parquet file, including ``"file.h5::/group"`` paths.
        A list or tuple creates an ``EMCPatternCollection`` in input order.
        A glob expression creates a collection in sorted filename order.
        Collection members must have the same number of pixels.

    Returns
    -------
    FileBackedEMCPatterns
        Format-specific reader or a collection of readers. Integer row
        indexing returns a dense NumPy array; slices and NumPy row index
        arrays return in-memory ``EMCPatternArray`` objects.

    Notes
    -----
    Counts and offsets are initialized when needed. Ordinary readers
    reopen files during access. EMC binary, HDF5 v2, and Parquet readers
    provide ``open()`` for persistent access using a context manager.
    HDF5 v1 readers load all events on first access.

    Examples
    --------
    Keep a file open for repeated reads::

        import emcfile as ef

        source = ef.open_patterns("patterns.h5")
        with source.open() as reader:
            frame = reader.sparse_pattern(0)
            batch = reader[:10]

    See Also
    --------
    patterns : Load or convert pattern data into memory.
    EMCPatternCollection : Treat several sources as one sequence.
    """
    return file_patterns(path)


# Descriptive alternatives. These remain aliases, rather than wrapper
# subclasses, to preserve exact ``isinstance`` behavior in downstream code.
FileBackedEMCPatterns = PatternsSOneFile
EMCBinaryPatternFile = PatternsSOneEMC
HDF5PatternFile = PatternsSOneH5
LegacyHDF5PatternFile = PatternsSOneH5V1
ParquetPatternFile = PatternsSOneParquet
EMCPatternCollection = PatternsSOneList
