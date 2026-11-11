from __future__ import annotations

import io
import logging
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import h5py
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, hstack
from typing_extensions import deprecated

from ._delta import encode_pattern_local_delta_parallel
from ._formatting import pretty_size
from ._h5_accel import invalidate_cache_for_path
from ._h5_direct_write import PrefilteredDatasetWriter
from ._h5_direct_write import available as direct_zstd_available
from ._h5_filters import hdf5plugin
from ._h5_workers import env_workers
from ._hdf5 import (
    PATH_TYPE,
    H5Path,
    atomic_write_path,
    check_remove_groups,
    make_path,
)
from ._html_display import html_card
from ._indexing import contiguous_ranges

_log = logging.getLogger(__name__)


class SPARSE_PATTERN(NamedTuple):
    """Sparse event arrays for one pattern.

    ``SparsePattern`` is the descriptive public alias for this named tuple.
    ``num_pix`` is the total number of detector pixels. ``place_ones`` stores
    indices with count one; ``place_multi`` and ``count_multi`` store paired
    indices and counts greater than one. All positions are absolute pixel
    indices, including when the source file uses delta encoding.

    The tuple is immutable, but its NumPy arrays can be writable views.
    """

    num_pix: int
    place_ones: npt.NDArray[np.uint32]
    place_multi: npt.NDArray[np.uint32]
    count_multi: npt.NDArray[np.int32]

    @property
    def num_pixels(self) -> int:
        """Number of pixels in this pattern."""
        return self.num_pix


# Canonical public spelling. ``SPARSE_PATTERN`` remains an exact alias so
# tuple identity, pattern matching, and pickled data stay compatible.
SparsePattern = SPARSE_PATTERN


HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


TRANGE = slice | npt.NDArray[np.bool_ | np.int32 | np.int64 | np.uint32 | np.uint64]
INT_INDEX = int | np.integer[Any]

_U32_MAX = int(np.iinfo(np.uint32).max)
_I32_MIN = int(np.iinfo(np.int32).min)
_I32_MAX = int(np.iinfo(np.int32).max)


def _is_int_index(val: object) -> TypeGuard[INT_INDEX]:
    return isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_))


def _index_1d_pattern_columns(
    row_arr: npt.NDArray[np.int32],
    columns: object,
) -> np.int32 | npt.NDArray[np.int32]:
    if isinstance(columns, (bool, np.bool_)):
        raise IndexError("Boolean scalar indices are not supported")
    if _is_int_index(columns):
        return cast(np.int32, row_arr[int(columns)])
    if isinstance(columns, slice):
        return row_arr[columns]
    if isinstance(columns, np.ndarray):
        if columns.ndim != 1:
            raise IndexError("column indices must be one-dimensional")
        if np.issubdtype(columns.dtype, np.bool_):
            if columns.size != row_arr.size:
                raise IndexError(
                    "Boolean column mask must match the number of pixels"
                )
            return row_arr[columns]
        if np.issubdtype(columns.dtype, np.integer):
            if columns.size > 0 and (
                np.any(columns >= row_arr.size)
                or (
                    np.issubdtype(columns.dtype, np.signedinteger)
                    and np.any(columns < -row_arr.size)
                )
            ):
                raise IndexError("column index out of range")
            return row_arr[columns.astype(np.intp, copy=False)]
        raise IndexError("column indices must be integers or Boolean")
    raise IndexError(f"Unsupported column index: {type(columns).__name__}")


def _ensure_32bit_unsigned(arr: Any, name: str) -> npt.NDArray[np.uint32]:
    if (
        isinstance(arr, np.ndarray)
        and arr.ndim == 1
        and arr.dtype in (np.dtype(np.uint32), np.dtype(np.int32))
    ):
        return cast(npt.NDArray[np.uint32], arr)
    if isinstance(arr, (list, tuple)) and len(arr) == 0:
        return np.empty(0, dtype=np.uint32)
    a = np.asarray(arr)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a 1D integer array")
    if a.size > 0:
        if np.issubdtype(a.dtype, np.signedinteger) and int(a.min()) < 0:
            raise ValueError(f"{name} values must be non-negative")
        if a.dtype.itemsize > 4 and int(a.max()) > _U32_MAX:
            raise ValueError(f"{name} values exceed uint32 maximum")
    return a.astype(np.uint32, copy=False)


def _ensure_32bit_signed(arr: Any, name: str) -> npt.NDArray[np.int32]:
    if (
        isinstance(arr, np.ndarray)
        and arr.ndim == 1
        and arr.dtype == np.dtype(np.int32)
    ):
        return cast(npt.NDArray[np.int32], arr)
    if isinstance(arr, (list, tuple)) and len(arr) == 0:
        return np.empty(0, dtype=np.int32)
    a = np.asarray(arr)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a 1D integer array")
    if a.size > 0:
        if np.issubdtype(a.dtype, np.signedinteger) and int(a.min()) < _I32_MIN:
            raise ValueError(f"{name} values are below int32 minimum")
        if (
            a.dtype.itemsize > 4 or np.issubdtype(a.dtype, np.unsignedinteger)
        ) and int(a.max()) > _I32_MAX:
            raise ValueError(f"{name} values exceed int32 maximum")
    return a.astype(np.int32, copy=False)


def _count_offsets(
    counts: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint64]:
    """Return cumulative event offsets for per-pattern counts."""
    arr = np.asarray(counts)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("counts must be a 1D integer array")
    if arr.size > 0 and np.issubdtype(arr.dtype, np.signedinteger) and np.any(arr < 0):
        raise ValueError("counts must be non-negative")
    offsets = np.empty(len(arr) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(arr, dtype=np.uint64, out=offsets[1:])
    offsets.flags.writeable = False
    return offsets


@runtime_checkable
class PatternsSOneBase(Protocol):
    @property
    def num_pix(self) -> int: ...

    @property
    def num_data(self) -> int: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def ones(self) -> npt.NDArray[np.uint32]: ...

    @property
    def multi(self) -> npt.NDArray[np.uint32]: ...

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]: ...

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]: ...

    @property
    def count_multi(self) -> npt.NDArray[np.int32]: ...

    def __len__(self) -> int: ...

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
    ) -> np.int32 | npt.NDArray[np.int32] | PatternsSOne: ...


EMCPatternSource = PatternsSOneBase


def _patterns_preview(
    patterns: PatternsSOneBase,
    preview_rows: int = 8,
    preview_cols: int = 16,
) -> npt.NDArray[np.int32]:
    rows = min(patterns.num_data, preview_rows)
    cols = min(patterns.num_pix, preview_cols)
    if rows == 0:
        return np.zeros((0, cols), dtype=np.int32)
    preview = cast(PatternsSOne, patterns[:rows]).todense()
    return cast(npt.NDArray[np.int32], np.atleast_2d(preview)[:, :cols])


class PatternsSOne:
    """
    Represents a collection of diffraction patterns in a sparse format.

    `EMCPatternArray` is the preferred descriptive public name. The historical
    `PatternsSOne` name remains an exact alias for compatibility.

    This class is optimized for storing and manipulating large sets of diffraction
    patterns where the data is sparse (i.e., most pixel values are zero). It
    achieves this by separately storing the locations of single-photon pixels
    (value = 1) and multi-photon pixels (value > 1), which significantly
    reduces memory usage compared to a dense NumPy array.

    Attributes
    ----------
    num_pix : int
        The number of pixels in each pattern.
    ones : numpy.ndarray
        A (num_data,) array storing the number of single-photon pixels for each
        pattern.
    multi : numpy.ndarray
        A (num_data,) array storing the number of multi-photon pixels for each
        pattern.
    place_ones : numpy.ndarray
        A 1D array storing the pixel indices of all single-photon events.
    place_multi : numpy.ndarray
        A 1D array storing the pixel indices of all multi-photon events.
    count_multi : numpy.ndarray
        A 1D array storing the photon counts for all multi-photon events.
    """

    ATTRS = ["ones", "multi", "place_ones", "place_multi", "count_multi"]

    def __init__(
        self,
        num_pix: int,
        ones: npt.NDArray[np.uint32],
        multi: npt.NDArray[np.uint32],
        place_ones: npt.NDArray[np.uint32],
        place_multi: npt.NDArray[np.uint32],
        count_multi: npt.NDArray[np.int32],
    ) -> None:
        self.ndim: int = 2
        self.num_pix = int(num_pix)
        self.ones = _ensure_32bit_unsigned(ones, "ones")
        self.multi = _ensure_32bit_unsigned(multi, "multi")
        self.place_ones = _ensure_32bit_unsigned(place_ones, "place_ones")
        self.place_multi = _ensure_32bit_unsigned(place_multi, "place_multi")
        self.count_multi = _ensure_32bit_signed(count_multi, "count_multi")
        self._update_offsets()

    def _update_offsets(self) -> None:
        self.ones_idx = _count_offsets(self.ones)
        self.multi_idx = _count_offsets(self.multi)

    @deprecated("Offsets are updated automatically; use _update_offsets() internally.")
    def update_idx(self) -> None:
        self._update_offsets()

    @property
    def num_pixels(self) -> int:
        """Number of pixels in each pattern."""
        return self.num_pix

    @property
    def num_patterns(self) -> int:
        """Number of patterns in the array."""
        return self.num_data

    @property
    def ones_offsets(self) -> npt.NDArray[np.uint64]:
        """Cumulative offsets into ``place_ones``."""
        return self.ones_idx

    @property
    def multi_offsets(self) -> npt.NDArray[np.uint64]:
        """Cumulative offsets into ``place_multi`` and ``count_multi``."""
        return self.multi_idx

    def check(self) -> bool:
        """Check that per-pattern counts match the event-array lengths.

        Returns ``True`` on success and raises ``ValueError`` on a length
        mismatch. Does not validate dtypes, pixel bounds, photon values,
        sorted positions, or duplicate positions.
        """
        if self.num_data != len(self.multi):
            raise ValueError(
                f"The `multi`{len(self.multi)} has different length with `ones`({self.num_data})"
            )
        ones_total = int(self.ones.sum(dtype=np.uint64))
        if ones_total != len(self.place_ones):
            raise ValueError(
                f"The expected length of `place_ones`({len(self.place_ones)}) should be {ones_total}."
            )

        multi_total = int(self.multi.sum(dtype=np.uint64))
        if multi_total != len(self.place_multi):
            raise ValueError(
                f"The expected length of `place_multi`({len(self.place_multi)}) should be {multi_total}."
            )

        if multi_total != len(self.count_multi):
            raise ValueError(
                f"The expected length of `count_multi`({len(self.count_multi)}) should be {multi_total}."
            )
        return True

    def __len__(self) -> int:
        return self.num_data

    def sparse_pattern(self, idx: int) -> SPARSE_PATTERN:
        """Return the sparse events for one pattern without densifying it.

        Parameters
        ----------
        idx : int
            Zero-based pattern index; negative indices count from the end.

        Returns
        -------
        SparsePattern
            Pixel count and views into this array's event buffers. Modifying
            those views modifies the original data.
        """
        orig_idx = int(idx)
        norm_idx = orig_idx + self.num_data if orig_idx < 0 else orig_idx
        if norm_idx < 0 or norm_idx >= self.num_data:
            raise IndexError(f"{orig_idx}")
        return SPARSE_PATTERN(
            self.num_pix,
            self.place_ones[self.ones_idx[norm_idx] : self.ones_idx[norm_idx + 1]],
            self.place_multi[self.multi_idx[norm_idx] : self.multi_idx[norm_idx + 1]],
            self.count_multi[self.multi_idx[norm_idx] : self.multi_idx[norm_idx + 1]],
        )

    @property
    def num_data(self) -> int:
        return len(self.ones)

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_data, self.num_pix

    def mean_photon_count(self) -> float:
        """Return total photons divided by the number of patterns.

        Returns ``0.0`` for an empty dataset. This is a per-pattern mean,
        not a per-pixel mean.
        """
        if self.num_data == 0:
            return 0.0
        return cast(int, self.sum()) / self.num_data

    @deprecated("Use mean_photon_count() instead.")
    def get_mean_count(self) -> float:
        return self.mean_photon_count()

    def __repr__(self) -> str:
        return f"""Pattern(1-sparse) <{hex(id(self))}>
  Number of patterns: {self.num_data}
  Number of pixels: {self.num_pix}
  Mean number of counts: {self.mean_photon_count():.3f}
  Size: {pretty_size(self.nbytes)}
  Sparsity: {self.sparsity() * 100:.2f} %
"""

    def _repr_html_(self) -> str:
        summary = {
            "patterns": self.num_data,
            "pixels": self.num_pix,
            "mean count": self.mean_photon_count() if self.num_data > 0 else 0.0,
            "size": pretty_size(self.nbytes),
        }
        return html_card(
            "Patterns",
            summary,
            details={"type": self.__class__.__name__},
            bars=(("sparsity", self.sparsity() * 100, "#2563eb"),),
        )

    @property
    def nbytes(self) -> int:
        return int(np.sum([getattr(self, i).nbytes for i in PatternsSOne.ATTRS]))

    def sparsity(self) -> float:
        if self.num_data == 0 or self.num_pix == 0:
            return 0.0
        return self.nbytes / (4 * self.num_data * self.num_pix)

    def __eq__(self, d: object) -> bool:
        if not isinstance(d, PatternsSOne):
            return NotImplemented
        if self.num_data != d.num_data:
            return False
        if self.num_pix != d.num_pix:
            return False
        for i in PatternsSOne.ATTRS + ["ones_idx", "multi_idx"]:
            if cast(bool, np.any(getattr(self, i) != getattr(d, i))):
                return False
        return True

    def _get_pattern(self, idx: int) -> npt.NDArray[np.int32]:
        orig_idx = idx
        if idx < 0:
            idx += self.num_data
        if idx >= self.num_data or idx < 0:
            raise IndexError(f"{orig_idx}")
        pattern = np.zeros(self.num_pix, "int32")
        pattern[self.place_ones[self.ones_idx[idx] : self.ones_idx[idx + 1]]] = 1
        r = slice(*self.multi_idx[idx : idx + 2])
        pattern[self.place_multi[r]] = self.count_multi[r]
        return pattern

    def _get_subdataset(self, idx: Any) -> PatternsSOne:
        so = self._get_sparse_ones().__getitem__(idx)
        sm = self._get_sparse_multi().__getitem__(idx)
        ones = so.indptr[1:] - so.indptr[:-1]
        multi = sm.indptr[1:] - sm.indptr[:-1]
        return PatternsSOne(
            so.shape[1],
            ones.astype(np.uint32),
            multi.astype(np.uint32),
            so.indices.astype(np.uint32),
            sm.indices.astype(np.uint32),
            sm.data,
        )

    def _get_chunked_column_slice(
        self, rows: slice, columns: TRANGE, *, event_budget: int = 1_000_000_000
    ) -> PatternsSOne:
        """Select columns with bounded int32 CSR chunks from contiguous rows."""
        row_start, row_stop, _ = rows.indices(self.num_data)
        row_stop = max(row_start, row_stop)
        if not 0 < event_budget <= np.iinfo(np.int32).max:
            raise ValueError("event_budget must fit signed int32")
        if isinstance(columns, slice):
            column_count = len(range(*columns.indices(self.num_pix)))
        else:
            if columns.ndim != 1:
                raise IndexError("column indices must be one-dimensional")
            if np.issubdtype(columns.dtype, np.bool_):
                if columns.size != self.num_pix:
                    raise IndexError(
                        "Boolean column mask must match the number of pixels"
                    )
                column_count = int(np.count_nonzero(columns))
            else:
                if not np.issubdtype(columns.dtype, np.integer):
                    raise IndexError("column indices must be integers or Boolean")
                if np.any(columns >= self.num_pix) or (
                    np.issubdtype(columns.dtype, np.signedinteger)
                    and np.any(columns < -self.num_pix)
                ):
                    raise IndexError("column index out of range")
                column_count = columns.size
        if row_start == row_stop or column_count == 0:
            return _zeros((row_stop - row_start, column_count))
        ones_counts: list[np.ndarray] = []
        multi_counts: list[np.ndarray] = []
        ones_places: list[np.ndarray] = []
        multi_places: list[np.ndarray] = []
        multi_values: list[np.ndarray] = []
        cursor = row_start
        while cursor < row_stop:
            stop = min(row_stop, cursor + np.iinfo(np.int32).max)
            for offsets in (self.ones_idx, self.multi_idx):
                limit = int(offsets[cursor]) + event_budget
                stop = min(stop, int(np.searchsorted(offsets, limit, side="right") - 1))
            if stop <= cursor:
                # A single row exceeds the budget; preserve the general path.
                source = self._get_contiguous_rows(rows)
                return source._get_subdataset((slice(None), columns))

            selected = []
            for place, offsets, data in (
                (self.place_ones, self.ones_idx, None),
                (self.place_multi, self.multi_idx, self.count_multi),
            ):
                event_start, event_stop = int(offsets[cursor]), int(offsets[stop])
                # SciPy copies a normal slice whose base is much larger than
                # the slice, even with copy=False. Give it a chunk-sized base.
                indices = np.frombuffer(
                    memoryview(place)[event_start:event_stop], dtype=np.int32
                )
                indptr = (offsets[cursor : stop + 1] - event_start).astype(np.int32)
                if data is None:
                    seed = np.ones(1, dtype=np.int32)
                    values = np.lib.stride_tricks.as_strided(
                        seed, shape=(event_stop - event_start,), strides=(0,)
                    )
                else:
                    values = (
                        np.frombuffer(
                            memoryview(data)[event_start:event_stop], dtype=data.dtype
                        )
                        if data.flags.c_contiguous
                        else data[event_start:event_stop]
                    )
                sparse = csr_array(
                    (values, indices, indptr),
                    shape=(stop - cursor, self.num_pix),
                    copy=False,
                )
                selected.append(sparse[:, columns])

            ones, multi = selected
            ones_counts.append(np.diff(ones.indptr).astype(np.uint32))
            multi_counts.append(np.diff(multi.indptr).astype(np.uint32))
            ones_places.append(ones.indices.astype(np.uint32))
            multi_places.append(multi.indices.astype(np.uint32))
            multi_values.append(multi.data)
            cursor = stop

        def join(parts: list[np.ndarray]) -> np.ndarray:
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

        return PatternsSOne(
            column_count,
            join(ones_counts),
            join(multi_counts),
            join(ones_places),
            join(multi_places),
            join(multi_values),
        )

    def _get_contiguous_rows(self, rows: slice) -> PatternsSOne:
        """Select rows using native EMC offsets without materializing CSR arrays."""
        start, stop, _ = rows.indices(self.num_data)
        # A forward slice with start > stop is empty, not a reversed event span.
        stop = max(start, stop)
        ones_start, ones_stop = int(self.ones_idx[start]), int(self.ones_idx[stop])
        multi_start, multi_stop = int(self.multi_idx[start]), int(self.multi_idx[stop])
        return PatternsSOne(
            self.num_pix,
            self.ones[start:stop],
            self.multi[start:stop],
            self.place_ones[ones_start:ones_stop],
            self.place_multi[multi_start:multi_stop],
            self.count_multi[multi_start:multi_stop],
        )

    def __pow__(self, n: int) -> PatternsSOne:
        if not isinstance(n, int):
            raise TypeError(f"n should be int, not {type(n)}")
        if n == 0:
            return _ones((self.num_data, self.num_pix))
        return PatternsSOne(
            self.num_pix,
            self.ones,
            self.multi,
            self.place_ones,
            self.place_multi,
            self.count_multi**n,
        )

    def sum(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.NDArray[Any] | np.int32 | np.int64 | np.float32 | np.float64 | int | float:
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
        acc_dtype = np.empty(0, dtype=np.int32).sum(dtype=dtype).dtype
        if axis is None:
            total = np.array(len(self.place_ones), dtype=acc_dtype) + np.sum(
                self.count_multi, dtype=acc_dtype
            )
            if keepdims:
                return np.asarray(total, dtype=acc_dtype).reshape((1, 1))
            return cast(
                np.int32 | np.int64 | np.float32 | np.float64 | int | float, total
            )
        elif axis == 1:
            ans: npt.NDArray[Any] = self.ones.astype(acc_dtype, copy=True)
            if self.place_multi.size > 0:
                ans += np.asarray(
                    self._get_sparse_multi().sum(axis=1, dtype=acc_dtype)
                ).reshape(-1)
            return ans[:, None] if keepdims else ans
        else:
            column_sum: npt.NDArray[Any] = np.zeros(self.num_pix, dtype=acc_dtype)
            np.add.at(column_sum, self.place_ones, 1)
            np.add.at(column_sum, self.place_multi, self.count_multi)
            return column_sum[None, :] if keepdims else column_sum

    def _get_subdataset0(self, i: npt.NDArray[np.integer[Any]]) -> PatternsSOne:
        if i.ndim != 1:
            raise IndexError("row indices must be one-dimensional")
        if i.size == 0:
            return _zeros((0, self.num_pix))
        if np.any(i >= self.num_data) or (
            np.issubdtype(i.dtype, np.signedinteger) and np.any(i < -self.num_data)
        ):
            raise IndexError("row index out of range")
        ids = i.astype(np.intp, copy=True)
        ids[ids < 0] += self.num_data
        ones = self.ones[ids]
        multi = self.multi[ids]
        num_ones = int(np.sum(ones, dtype=np.uint64))
        num_multi = int(np.sum(multi, dtype=np.uint64))
        result = PatternsSOne(
            self.num_pix,
            ones,
            multi,
            np.empty(num_ones, dtype=np.uint32),
            np.empty(num_multi, dtype=np.uint32),
            np.empty(num_multi, dtype=np.int32),
        )
        ranges = contiguous_ranges(ids)
        ones_ranges = self.ones_idx[ranges]
        multi_ranges = self.multi_idx[ranges]
        for source, spans, target in (
            (self.place_ones, ones_ranges, result.place_ones),
            (self.place_multi, multi_ranges, result.place_multi),
            (self.count_multi, multi_ranges, result.count_multi),
        ):
            # Legacy HDF5 stores positions as signed int32. Valid EMC positions
            # are nonnegative, so preserve the public uint32 output convention.
            np.concatenate(
                [source[start:stop] for start, stop in spans],
                out=target,
                casting="unsafe",
            )
        return result

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
        """Select patterns and optionally pixels.

        An integer returns a dense 1D ``int32`` NumPy array. A row slice or
        1D integer/Boolean NumPy array returns an ``EMCPatternArray``.
        Contiguous row slices share event buffers; strided slices and row
        index arrays copy events. Integer row indices preserve order and
        duplicates. Boolean masks must match the number of patterns.

        A two-item tuple selects rows and columns, reducing integer axes like
        NumPy: ``(int, int)`` returns an ``np.int32`` scalar, and ``(int,
        slice/array)`` or ``(slice/array, int)`` returns a 1D ``int32`` array.
        Selections containing two slices/arrays select rows first; two advanced
        selectors follow SciPy's paired-index semantics rather than forming a
        Cartesian product.
        """
        if isinstance(index, (bool, np.bool_)):
            raise IndexError("Boolean scalar indices are not supported")
        if isinstance(index, tuple) and len(index) == 2:
            rows, columns = index
            if isinstance(rows, (bool, np.bool_)) or isinstance(
                columns, (bool, np.bool_)
            ):
                raise IndexError("Boolean scalar indices are not supported")
            if _is_int_index(rows):
                return _index_1d_pattern_columns(self._get_pattern(int(rows)), columns)
            if _is_int_index(columns):
                col = int(columns)
                if col < 0:
                    col += self.num_pix
                if col < 0 or col >= self.num_pix:
                    raise IndexError(
                        f"column index {columns} is out of bounds for axis 1 "
                        f"with size {self.num_pix}"
                    )
                sub = cast(PatternsSOne, self[cast(TRANGE, rows), col : col + 1])
                col_out = sub.ones.astype(np.int32, copy=True)
                if sub.place_multi.size > 0:
                    row_ids = np.repeat(
                        np.arange(sub.num_data, dtype=np.intp),
                        sub.multi.astype(np.intp, copy=False),
                    )
                    np.add.at(col_out, row_ids, sub.count_multi)
                return col_out
            # Two fancy selectors have SciPy's paired-index semantics, rather
            # than a Cartesian product. Keep that existing general path.
            if (
                isinstance(rows, (slice, np.ndarray))
                and isinstance(columns, (slice, np.ndarray))
                and (isinstance(rows, slice) or isinstance(columns, slice))
            ):
                if isinstance(rows, slice) and rows.step in (None, 1):
                    source = self._get_contiguous_rows(rows)
                else:
                    source = cast(PatternsSOne, self[rows])
                if isinstance(columns, slice) and columns.indices(self.num_pix) == (
                    0,
                    self.num_pix,
                    1,
                ):
                    return source
                if (
                    source.num_pix <= np.iinfo(np.int32).max
                    and source.place_ones.dtype
                    in (np.dtype(np.uint32), np.dtype(np.int32))
                    and source.place_multi.dtype
                    in (np.dtype(np.uint32), np.dtype(np.int32))
                    and source.place_ones.flags.c_contiguous
                    and source.place_multi.flags.c_contiguous
                ):
                    return source._get_chunked_column_slice(slice(None), columns)
                return source._get_subdataset((slice(None), columns))
        match index:
            case int() | np.integer():
                return self._get_pattern(int(index))
            case np.ndarray() if np.issubdtype(index.dtype, bool):
                if index.ndim != 1 or index.size != self.num_data:
                    raise IndexError(
                        "Boolean row mask must match the number of patterns"
                    )
                return self._get_subdataset0(np.where(index)[0])
            case np.ndarray() if np.issubdtype(index.dtype, np.integer):
                return self._get_subdataset0(cast(npt.NDArray[np.integer[Any]], index))
            case slice() if index.step is None or index.step == 1:
                return self._get_contiguous_rows(index)
            case slice():
                return self._get_subdataset0(
                    np.arange(*index.indices(self.num_data), dtype=np.intp)
                )
            case _:
                return self._get_subdataset(index)

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
        """Write this array to an EMC binary or HDF5 pattern file.

        Parameters match ``write_patterns`` except that the source is this
        array and the write buffer size is not configurable here. HDF5 v2
        is the default, with absolute positions and no compression.

        See Also
        --------
        write_patterns : Output formats, compression, and encoding options.
        """
        return write_patterns(
            [self],
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

    def _get_sparse_ones(self) -> csr_array:
        _one = np.ones(1, "i4")
        _one = np.lib.stride_tricks.as_strided(
            _one, shape=(self.place_ones.shape[0],), strides=(0,)
        )
        indices, indptr = self._csr_index_buffers(self.place_ones, self.ones_idx)
        return csr_array((_one, indices, indptr), shape=self.shape, copy=False)

    def _csr_index_buffers(
        self, place: npt.NDArray[np.uint32], offsets: npt.NDArray[np.uint64]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use signed 32-bit CSR indices when valid EMC positions fit.

        Valid positions are smaller than ``num_pix``, so the shape bounds the
        unsigned-to-signed reinterpretation without scanning all events.
        """
        limit = np.iinfo(np.int32).max
        if (
            self.num_data <= limit
            and self.num_pix <= limit
            and int(offsets[-1]) <= limit
        ):
            if place.dtype == np.uint32:
                indices: np.ndarray = place.view(np.int32)
            elif place.dtype == np.int32:
                indices = place
            else:
                return place, offsets
            return indices, offsets.astype(np.int32)
        return place, offsets

    def _get_sparse_multi(self) -> csr_array:
        indices, indptr = self._csr_index_buffers(self.place_multi, self.multi_idx)
        return csr_array(
            (self.count_multi, indices, indptr), shape=self.shape, copy=False
        )

    def tocsr(self) -> csr_array:
        """Convert to a SciPy CSR array with shape ``self.shape``.

        Combines the one-photon and multi-photon streams without allocating
        a dense pattern array.
        """
        return self._get_sparse_ones() + self._get_sparse_multi()

    def todense(self) -> npt.NDArray[np.int32]:
        """Allocate a dense ``int32`` NumPy array of photon counts.

        The result is squeezed: a singleton pattern or pixel dimension is
        removed, and a one-pattern, one-pixel dataset becomes a 0D array.
        Memory before squeezing is ``num_patterns * num_pixels * 4`` bytes.
        """
        ans = np.zeros(self.shape, dtype=np.int32)
        ans += self._get_sparse_ones()
        ans += self._get_sparse_multi()
        return cast(npt.NDArray[np.int32], np.squeeze(ans))

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[Any]:
        ans = self.todense()
        if dtype is not None:
            ans = ans.astype(dtype, copy=False)
        if copy:
            ans = ans.copy()
        return ans

    def __matmul__(self, mtx: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return cast(
            npt.NDArray[Any],
            self._get_sparse_ones() @ mtx + self._get_sparse_multi() @ mtx,
        )

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type[object]],
        args: Iterable[object],
        kwargs: Mapping[str, object],
    ) -> object:
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override

        # __array_function__ to handle PatternsSOne objects.

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def has_sorted_indices(self) -> bool:
        """Return whether positions are strictly increasing within each stream.

        One-photon and multi-photon positions are checked separately for
        each pattern; duplicate positions within a stream are not sorted.
        """
        for values, offsets in (
            (self.place_multi, self.multi_idx),
            (self.place_ones, self.ones_idx),
        ):
            if values.size > 1:
                not_increasing = values[1:] <= values[:-1]
                starts = offsets[np.flatnonzero(np.diff(offsets))]
                valid_starts = starts[(starts > 0) & (starts < values.size)]
                not_increasing[valid_starts.astype(np.intp) - 1] = False
                if np.any(not_increasing):
                    return False
        return True

    @deprecated("Use has_sorted_indices() instead.")
    def check_indices_ordered(self) -> bool:
        return self.has_sorted_indices()

    def sort_indices(self) -> None:
        """Sort pixel positions within each pattern in place.

        Reorders multi-photon counts together with their positions. Does
        not merge duplicates. Shared event-buffer views observe the changes.
        """
        if self.has_sorted_indices():
            return
        for i in range(self.num_data):
            s, e = self.ones_idx[i], self.ones_idx[i + 1]
            t = np.argsort(self.place_ones[s:e])
            self.place_ones[s:e] = self.place_ones[s:e][t]
            s, e = self.multi_idx[i], self.multi_idx[i + 1]
            t = np.argsort(self.place_multi[s:e])
            self.place_multi[s:e] = self.place_multi[s:e][t]
            self.count_multi[s:e] = self.count_multi[s:e][t]

    @deprecated("Use sort_indices() instead.")
    def ensure_indices_ordered(self) -> None:
        self.sort_indices()


# Keep one runtime class object for full ``isinstance`` and subclass
# compatibility across downstream projects.
EMCPatternArray = PatternsSOne


FT = TypeVar("FT", bound=Callable[..., Any])


def implements(np_function: Callable[..., Any]) -> Callable[[FT], FT]:
    "Register an __array_function__ implementation for PatternsSOne objects."

    def decorator(func: FT) -> FT:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def _iter_buffered_arrays(
    pattern_sets: Sequence[PatternsSOneBase], buffer_size: int, attribute: str
) -> Iterable[npt.NDArray[np.int32] | npt.NDArray[np.uint32]]:
    buffer = []
    nbytes = 0
    for pattern_set in pattern_sets:
        ag = getattr(pattern_set, attribute)
        nbytes += ag.nbytes
        buffer.append(ag)
        if nbytes < buffer_size:
            continue
        if len(buffer) == 1:
            yield buffer[0]
        else:
            yield np.concatenate(buffer)
        buffer = []
        nbytes = 0
    if nbytes > 0:
        if len(buffer) == 1:
            yield buffer[0]
        else:
            yield np.concatenate(buffer)


@deprecated("Use _iter_buffered_arrays() internally.")
def iter_array_buffer(
    datas: Sequence[PatternsSOneBase], buffer_size: int, g: str
) -> Iterable[npt.NDArray[np.int32] | npt.NDArray[np.uint32]]:
    return _iter_buffered_arrays(datas, buffer_size, g)


def _as_bin_attr_le32(data: PatternsSOneBase, attr: str) -> np.ndarray:
    arr = getattr(data, attr)
    if attr == "count_multi":
        return _ensure_32bit_signed(arr, attr).astype("<i4", copy=False)
    return _ensure_32bit_unsigned(arr, attr).view(np.uint32).astype("<u4", copy=False)


def _validate_bin_sources(datas: Sequence[PatternsSOneBase]) -> tuple[int, int]:
    if not datas:
        raise ValueError("At least one pattern source is required")
    num_pix = int(datas[0].num_pix)
    if any(int(d.num_pix) != num_pix for d in datas):
        raise ValueError("Inconsistent number of pixels across pattern sources")
    num_data = sum(int(data.num_data) for data in datas)
    if num_data < 0 or num_data > _I32_MAX or num_pix < 0 or num_pix > _I32_MAX:
        raise ValueError("EMC binary header requires int32 num_data and num_pix")
    return num_data, num_pix


def _write_bin(datas: Sequence[PatternsSOneBase], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists")
    num_data, num_pix = _validate_bin_sources(datas)
    with atomic_write_path(path) as tmp_path, tmp_path.open("wb") as fptr:
        header = np.zeros((256), dtype="<i4")
        header[:2] = [num_data, num_pix]
        header.tofile(fptr)
        for g in PatternsSOne.ATTRS:
            for data in datas:
                _as_bin_attr_le32(data, g).tofile(fptr)


def _write_bytes(datas: Sequence[PatternsSOneBase], path: io.BytesIO) -> None:
    num_data, num_pix = _validate_bin_sources(datas)
    header = np.zeros((256), dtype="<i4")
    header[:2] = [num_data, num_pix]
    path.write(header.tobytes())
    for g in PatternsSOne.ATTRS:
        for data in datas:
            path.write(_as_bin_attr_le32(data, g).tobytes())


def _h5_filter_kwargs(
    compression: None | int | str, compression_opts: Any, shuffle: bool
) -> dict[str, Any]:
    if compression == "zstd":
        plugin = hdf5plugin()
        return {
            "shuffle": shuffle,
            **plugin.Zstd(clevel=1 if compression_opts is None else compression_opts),
        }
    result: dict[str, Any] = {"shuffle": shuffle}
    if compression is not None:
        result["compression"] = compression
        if compression_opts is not None:
            result["compression_opts"] = compression_opts
    return result


_ALL_H5_PATTERN_KEYS = frozenset(
    {"num_pix", "ones", "multi", "place_ones", "place_multi", "count_multi"}
)
_ALL_H5_PATTERN_ATTRS = frozenset(
    {"num_pix", "num_data", "version", "position_encoding"}
)


@contextmanager
def _open_h5_pattern_target(
    path: H5Path, names: Sequence[str], overwrite: bool
) -> Iterator[h5py.Group | h5py.File]:
    file_exists = path.fn.exists()
    is_valid_h5 = file_exists and h5py.is_hdf5(path.fn)
    if file_exists and not overwrite:
        if not is_valid_h5:
            raise FileExistsError(f"{path.fn} exists")
        with h5py.File(path.fn, "r") as rf:
            if path.gn in rf:
                grp = rf[path.gn]
                if not isinstance(grp, h5py.Group):
                    raise TypeError(f"{path} is not a group.")
                check_remove_groups(grp, names, overwrite=False)

    use_atomic = False
    if not file_exists or not is_valid_h5:
        use_atomic = True
    elif overwrite and path.gn == "/":
        with h5py.File(path.fn, "r") as rf:
            if set(rf.keys()).issubset(_ALL_H5_PATTERN_KEYS) and set(
                rf.attrs.keys()
            ).issubset(_ALL_H5_PATTERN_ATTRS):
                use_atomic = True

    if use_atomic:
        with atomic_write_path(path.fn) as tmp_fn:
            with h5py.File(tmp_fn, "w") as h5f:
                target = h5f if path.gn == "/" else h5f.create_group(path.gn)
                yield target
    else:
        with path.open_group("a", "a") as (_, fp):
            assert isinstance(fp, (h5py.Group, h5py.File))
            check_remove_groups(fp, names, overwrite)
            yield fp


def _write_h5_v2(
    datas: Sequence[PatternsSOneBase],
    path: H5Path,
    overwrite: bool,
    buffer_size: int,
    compression: None | int | str = None,
    compression_opts: Any = None,
    shuffle: bool = False,
    position_encoding: str = "absolute",
    check_sorted: bool = False,
) -> None:
    if position_encoding not in {"absolute", "delta"}:
        raise ValueError("position_encoding must be 'absolute' or 'delta'")
    if not datas:
        raise ValueError("at least one pattern source is required")
    num_ones = int(sum(int(d.ones.sum(dtype=np.uint64)) for d in datas))
    num_multi = int(sum(int(d.multi.sum(dtype=np.uint64)) for d in datas))
    num_data = int(sum(data.num_data for data in datas))
    num_pix = datas[0].num_pix
    if any(data.num_pix != num_pix for data in datas):
        raise ValueError("all pattern sources must have the same number of pixels")
    direct_zstd = (
        position_encoding == "delta"
        and compression == "zstd"
        and shuffle
        and all(size > 0 for size in (num_data, num_ones, num_multi))
        and direct_zstd_available()
    )
    env_chunk_bytes = os.environ.get("EMCFILE_H5_DIRECT_CHUNK_BYTES")
    chunk_bytes = int(env_chunk_bytes) if env_chunk_bytes is not None else (512 << 10)
    if chunk_bytes <= 0 or chunk_bytes % np.dtype("u4").itemsize:
        raise ValueError(
            "EMCFILE_H5_DIRECT_CHUNK_BYTES must be a positive multiple of 4"
        )
    base_chunk_elems = chunk_bytes // 4
    max_payload = max(1, num_ones, num_multi)
    workers = env_workers("EMCFILE_H5_WRITE_WORKERS", 4)
    kwargs = _h5_filter_kwargs(compression, compression_opts, shuffle)
    names = ["ones", "multi", "place_ones", "place_multi", "count_multi"]
    check_names = ["num_pix", *names]
    invalidate_cache_for_path(path.fn)
    with _open_h5_pattern_target(path, check_names, overwrite) as fp:
        shapes = {
            "ones": (num_data, "i4"),
            "multi": (num_data, "i4"),
            "place_ones": (num_ones, "u4"),
            "place_multi": (num_multi, "u4"),
            "count_multi": (num_multi, "i4"),
        }
        datasets = {}
        for name, (size, dtype) in shapes.items():
            dataset_kwargs = dict(kwargs)
            if (direct_zstd or (compression == "zstd" and shuffle)) and size > 0:
                if env_chunk_bytes is not None or name in ("ones", "multi"):
                    target_elems = base_chunk_elems
                else:
                    scaled = int(base_chunk_elems * size // max_payload)
                    target_elems = max(min(size, 16384), min(base_chunk_elems, scaled))
                dataset_kwargs["chunks"] = (max(1, min(size, target_elems)),)
            datasets[name] = fp.create_dataset(
                name, (size,), dtype=dtype, **dataset_kwargs
            )
        pool = ThreadPoolExecutor(max_workers=workers) if direct_zstd else None
        slots = threading.Semaphore(max(2, workers * 2)) if direct_zstd else None
        writers = (
            {
                name: PrefilteredDatasetWriter(
                    dataset,
                    1 if compression_opts is None else int(compression_opts),
                    workers,
                    pool=pool,
                    slots=slots,
                )
                for name, dataset in datasets.items()
            }
            if direct_zstd
            else None
        )
        offsets = {name: 0 for name in names}
        try:
            for data in datas:
                if (
                    writers is not None
                    and len(datas) == 1
                    and isinstance(data, PatternsSOne)
                ):
                    batches = [data]
                else:
                    patterns_per_batch = max(
                        1, buffer_size // max(1, data.nbytes // max(1, data.num_data))
                    )
                    batches = [
                        data[start : min(start + patterns_per_batch, data.num_data)]
                        for start in range(0, data.num_data, patterns_per_batch)
                    ]
                for batch in batches:
                    assert isinstance(batch, PatternsSOne)
                    if (
                        writers is not None
                        and position_encoding == "delta"
                        and hasattr(writers["place_ones"], "write_delta")
                    ):
                        for name in ("ones", "multi"):
                            writers[name].write(np.asarray(getattr(batch, name)))
                            writers[name].drain()
                        writers["place_ones"].write_delta(
                            batch.place_ones,
                            batch.ones_idx,
                            batch.ones,
                            check_sorted=check_sorted,
                        )
                        writers["place_ones"].drain()
                        writers["place_multi"].write_delta(
                            batch.place_multi,
                            batch.multi_idx,
                            batch.multi,
                            check_sorted=check_sorted,
                        )
                        writers["place_multi"].drain()
                        writers["count_multi"].write(np.asarray(batch.count_multi))
                        writers["count_multi"].drain()
                        continue
                    arrays = {
                        "ones": np.asarray(batch.ones),
                        "multi": np.asarray(batch.multi),
                        "place_ones": encode_pattern_local_delta_parallel(
                            batch.place_ones,
                            batch.ones,
                            workers,
                            check_sorted=check_sorted,
                            accelerated=direct_zstd,
                        )
                        if position_encoding == "delta"
                        else np.asarray(batch.place_ones),
                        "place_multi": encode_pattern_local_delta_parallel(
                            batch.place_multi,
                            batch.multi,
                            workers,
                            check_sorted=check_sorted,
                            accelerated=direct_zstd,
                        )
                        if position_encoding == "delta"
                        else np.asarray(batch.place_multi),
                        "count_multi": np.asarray(batch.count_multi),
                    }
                    for name, array in arrays.items():
                        if writers is None:
                            datasets[name][
                                offsets[name] : offsets[name] + array.size
                            ] = array
                            offsets[name] += array.size
                        else:
                            writers[name].write(array)
                            writers[name].drain()
            if writers is not None:
                for writer in writers.values():
                    writer.close()
            fp.attrs.update(
                num_pix=num_pix,
                num_data=num_data,
                version="2",
                position_encoding=position_encoding,
            )
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
    invalidate_cache_for_path(path.fn)


def write_patterns(
    datas: Sequence[PatternsSOneBase],
    path: PATH_TYPE | io.BytesIO,
    *,
    h5version: str = "2",
    overwrite: bool = False,
    buffer_size: int = 1073741824,  # 2 ** 30 bytes = 1 GB
    compression: None | int | str = None,
    compression_opts: Any = None,
    shuffle: bool = False,
    position_encoding: str = "absolute",
    check_sorted: bool = False,
    hdf5_version: str | None = None,
) -> None:
    """Write pattern sources in order to one EMC binary, HDF5, or Parquet dataset.

    Parameters
    ----------
    datas : sequence of pattern sources
        In-memory arrays or file-backed sources with equal pixel counts.
    path : str, pathlib.Path, H5Path, or io.BytesIO
        ``.emc`` or ``.bin`` for binary output; ``.h5`` or
        ``"file.h5::/group"`` for HDF5; ``.parquet`` or ``.pq`` for Parquet.
        A byte stream receives EMC binary.
    h5version : {"1", "2"}, default "2"
        HDF5 layout version. Version 1 accepts only one source.
    overwrite : bool, default False
        Allow replacing existing output pattern data. For a byte stream,
        data is written at its current position regardless of this flag.
    buffer_size : int, default 1073741824
        Approximate source batch size in bytes for HDF5 v2 writes. This
        is a batching target, not a cap on total memory usage.
    compression : str or int, optional
        HDF5 or Parquet compressor, such as ``"gzip"``, ``"lzf"``, or ``"zstd"``.
        Zstd requires ``hdf5plugin`` / ``emcfile-fast`` from the ``fast`` extra.
    compression_opts : optional
        Compressor-specific options. For Zstd, the compression level
        defaults to 1 when omitted.
    shuffle : bool, default False
        Enable the byte-shuffle filter.
    position_encoding : {"absolute", "delta"}, default "absolute"
        HDF5 v2 / Parquet pixel-position encoding. Delta encoding restarts within
        each pattern and requires sorted positions. Counts are unchanged.
    check_sorted : bool, default False
        Validate position ordering before writing delta-encoded data.
    hdf5_version : {"1", "2"}, optional
        Descriptive alias for ``h5version``. Do not supply both names.

    Raises
    ------
    FileExistsError
        If output pattern data exists and overwriting is disabled.
    ValueError
        If the output format, HDF5 version, encoding, or source pixel
        counts are incompatible, or requested ordering validation fails.
    NotImplementedError
        If writing multiple sources using HDF5 version 1.

    Notes
    -----
    Compression, shuffle, position encoding, ordering validation, and
    buffer size apply to HDF5 v2 and Parquet output. Binary and HDF5 v1 output do not
    use these options. Compatible delta/shuffle/Zstd writes can use the
    optional accelerated writer automatically.

    Examples
    --------
    >>> import numpy as np
    >>> import emcfile as ef
    >>> from io import BytesIO
    >>> data = ef.patterns(np.array([[0, 1, 2]], dtype=np.int32))
    >>> stream = BytesIO()
    >>> ef.write_patterns([data, data], stream)
    >>> _ = stream.seek(0)
    >>> ef.patterns(stream).shape
    (2, 3)
    """
    if hdf5_version is not None:
        if h5version != "2":
            raise TypeError("Use either 'hdf5_version' or 'h5version', not both")
        h5version = hdf5_version
    if isinstance(path, io.BytesIO):
        return _write_bytes(datas, path)

    f = make_path(path)
    if isinstance(f, Path):
        if f.suffix in [".emc", ".bin"]:
            return _write_bin(datas, f, overwrite)
        if f.suffix.lower() in [".parquet", ".pq"]:
            from ._parquet import write_parquet_patterns

            return write_parquet_patterns(
                datas,
                f,
                overwrite=overwrite,
                buffer_size=buffer_size,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                position_encoding=position_encoding,
                check_sorted=check_sorted,
            )
    elif isinstance(f, H5Path):
        if h5version == "1":
            if len(datas) > 1:
                raise NotImplementedError()
            _log.warning(
                'This format has performance issue. h5version = "2" is recommended.'
            )
            return _write_h5_v1(datas[0], f, overwrite)
        elif h5version == "2":
            return _write_h5_v2(
                datas,
                f,
                overwrite,
                buffer_size,
                compression,
                compression_opts,
                shuffle,
                position_encoding,
                check_sorted,
            )
        else:
            raise ValueError(f"The h5version(={h5version}) should be '1' or '2'.")
    raise ValueError(f"Wrong file name {path}")


def _write_h5_v1(
    data: PatternsSOneBase,
    path: H5Path,
    overwrite: bool,
    start: int = 0,
    end: int | None = None,
) -> None:
    dt = h5py.special_dtype(vlen=np.int32)
    names = ["num_pix", "ones", "multi", "place_ones", "place_multi", "count_multi"]
    invalidate_cache_for_path(path.fn)
    with _open_h5_pattern_target(path, names, overwrite) as fp:
        for k in ("num_pix", "num_data", "position_encoding"):
            fp.attrs.pop(k, None)
        num_pix = fp.create_dataset("num_pix", (1,), dtype=np.int32)
        num_pix[0] = data.num_pix

        place_ones = fp.create_dataset("place_ones", (data.num_data,), dtype=dt)
        place_multi = fp.create_dataset("place_multi", (data.num_data,), dtype=dt)
        count_multi = fp.create_dataset("count_multi", (data.num_data,), dtype=dt)

        if data.num_data > 0:
            ones_idx = _count_offsets(data.ones)
            multi_idx = _count_offsets(data.multi)

            for idx, d in enumerate(np.split(data.place_ones, ones_idx[1:-1]), start):
                place_ones[idx] = d

            for idx, d in enumerate(np.split(data.place_multi, multi_idx[1:-1]), start):
                place_multi[idx] = d

            for idx, d_c in enumerate(
                np.split(data.count_multi, multi_idx[1:-1]), start
            ):
                count_multi[idx] = d_c
        fp.attrs["version"] = "1"
    invalidate_cache_for_path(path.fn)


@implements(np.concatenate)
def _concatenate_emc_pattern_arrays(
    patterns_l: Sequence[PatternsSOne], axis: int = 0, casting: str = "safe"
) -> PatternsSOne:
    "stack pattern sets together"
    if axis == 0:
        num_pix = patterns_l[0].num_pix
        for d in patterns_l:
            if d.num_pix != num_pix:
                raise ValueError(
                    "The numbers of pixels of each pattern are not consistent."
                )
        if casting == "safe":
            ans = PatternsSOne(
                num_pix,
                *[
                    np.concatenate([getattr(d, g) for d in patterns_l])
                    for g in PatternsSOne.ATTRS
                ],
            )
            ans.check()
            return ans
        if (casting == "destroy") and isinstance(patterns_l, list):
            ans = patterns_l.pop(0)
            while len(patterns_l) > 0:
                pat = patterns_l.pop(0)
                pat = {g: getattr(pat, g) for g in PatternsSOne.ATTRS}
                for g in PatternsSOne.ATTRS:
                    b = pat.pop(g)
                    a = getattr(ans, g)
                    a.resize(a.shape[0] + b.shape[0], refcheck=False)
                    a[a.shape[0] - b.shape[0] :] = b[:]
            ans._update_offsets()
            return ans
        raise ValueError(f"Unsupported casting mode: {casting}")
    elif axis == 1:
        ones = cast(csr_array, hstack([d._get_sparse_ones() for d in patterns_l]))
        multi = cast(csr_array, hstack([d._get_sparse_multi() for d in patterns_l]))
        assert ones.shape is not None
        return PatternsSOne(
            ones.shape[1],
            ones=np.diff(ones.indptr).astype(np.uint32, copy=False),
            multi=np.diff(multi.indptr).astype(np.uint32, copy=False),
            place_ones=cast(np.ndarray, ones.indices).astype(np.uint32, copy=False),
            place_multi=cast(np.ndarray, multi.indices).astype(np.uint32, copy=False),
            count_multi=cast(np.ndarray, multi.data).astype(np.int32, copy=False),
        )
    raise ValueError("The axis should be 0 or 1.")


@deprecated("Use numpy.concatenate() instead.")
def concatenate_PatternsSOne(
    patterns_l: Sequence[PatternsSOne], axis: int = 0, casting: str = "safe"
) -> PatternsSOne:
    return _concatenate_emc_pattern_arrays(patterns_l, axis, casting)


def _full(shape: tuple[int, int], val: int) -> PatternsSOne:
    num_data, num_pix = shape
    return PatternsSOne(
        num_pix,
        np.zeros(num_data, dtype=np.uint32),
        np.full(num_data, num_pix, dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.resize(np.arange(num_pix, dtype=np.uint32), num_pix * num_data),
        np.full(num_pix * num_data, val, dtype=np.int32),
    )


def _ones(shape: tuple[int, int]) -> PatternsSOne:
    num_data, num_pix = shape
    return PatternsSOne(
        num_pix,
        np.full(num_data, num_pix, dtype=np.uint32),
        np.zeros(num_data, dtype=np.uint32),
        np.resize(np.arange(num_pix, dtype=np.uint32), num_pix * num_data),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.int32),
    )


def _zeros(shape: tuple[int, int]) -> PatternsSOne:
    num_data, num_pix = shape
    return PatternsSOne(
        num_pix,
        np.zeros(num_data, dtype=np.uint32),
        np.zeros(num_data, dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.int32),
    )
