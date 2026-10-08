from __future__ import annotations

import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    Protocol,
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

from ._formatting import pretty_size
from ._delta import encode_pattern_local_delta_parallel
from ._h5_direct_write import PrefilteredDatasetWriter, available as direct_zstd_available
from ._h5_filters import hdf5plugin
from ._h5_workers import env_workers
from ._hdf5 import PATH_TYPE, H5Path, check_remove_groups, make_path
from ._html_display import html_card
from ._indexing import contiguous_ranges

_log = logging.getLogger(__name__)


class SPARSE_PATTERN(NamedTuple):
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


def _count_offsets(
    counts: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.uint64]:
    """Return cumulative event offsets for per-pattern counts."""
    offsets = np.zeros(len(counts) + 1, dtype=np.uint64)
    np.cumsum(counts, out=offsets[1:])
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
    def __getitem__(self, index: int | np.integer) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE) -> PatternsSOne: ...

    @overload
    def __getitem__(self, index: tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: int | np.integer | TRANGE | tuple[TRANGE, TRANGE],
    ) -> npt.NDArray[np.int32] | PatternsSOne: ...


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
        self.ones = ones
        self.multi = multi
        self.place_ones = place_ones
        self.place_multi = place_multi
        self.count_multi = count_multi
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
        if self.num_data != len(self.multi):
            raise ValueError(
                f"The `multi`{len(self.multi)} has different length with `ones`({self.num_data})"
            )
        ones_total = self.ones.sum()
        if ones_total != len(self.place_ones):
            raise ValueError(
                f"The expected length of `place_ones`({len(self.place_ones)}) should be {ones_total}."
            )

        multi_total = self.multi.sum()
        if multi_total != len(self.place_multi):
            raise ValueError(
                f"The expected length of `place_multi`({len(self.place_multi)}) should be {multi_total}."
            )

        if multi_total != len(self.count_multi):
            raise ValueError(
                f"The expected length of `place_multi`({len(self.count_multi)}) should be {multi_total}."
            )
        return True

    def __len__(self) -> int:
        return self.num_data

    def sparse_pattern(self, idx: int) -> SPARSE_PATTERN:
        return SPARSE_PATTERN(
            self.num_pix,
            self.place_ones[self.ones_idx[idx] : self.ones_idx[idx + 1]],
            self.place_multi[self.multi_idx[idx] : self.multi_idx[idx + 1]],
            self.count_multi[self.multi_idx[idx] : self.multi_idx[idx + 1]],
        )

    @property
    def num_data(self) -> int:
        return len(self.ones)

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_data, self.num_pix

    def mean_photon_count(self) -> float:
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
        for i in PatternsSOne.ATTRS:
            if cast(bool, np.any(getattr(self, i) != getattr(d, i))):
                return False
        return True

    def _get_pattern(self, idx: int) -> npt.NDArray[np.int32]:
        if idx >= self.num_data or idx < 0:
            raise IndexError(f"{idx}")
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
        if axis is None:
            return cast(
                int, len(self.place_ones) + np.sum(self.count_multi, dtype=dtype)
            )
        elif axis == 1:
            ans = self.ones.astype(dtype, copy=True)
            ans += np.squeeze(self._get_sparse_multi().sum(axis=1, dtype=dtype))
            return ans[:, None] if keepdims else ans
        elif axis == 0:
            column_sum: npt.NDArray[Any] = np.zeros(self.num_pix, dtype=dtype)
            np.add.at(column_sum, self.place_ones, 1)
            np.add.at(column_sum, self.place_multi, self.count_multi)
            return column_sum[None, :] if keepdims else column_sum
        raise ValueError(f"Do not support axis={axis}.")

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
    def __getitem__(self, index: int | np.integer) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE) -> PatternsSOne: ...

    @overload
    def __getitem__(self, index: tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: int | np.integer | TRANGE | tuple[TRANGE, TRANGE],
    ) -> npt.NDArray[np.int32] | PatternsSOne:
        if isinstance(index, tuple) and len(index) == 2:
            rows, columns = index
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
                    and source.place_ones.dtype in (np.dtype(np.uint32), np.dtype(np.int32))
                    and source.place_multi.dtype in (np.dtype(np.uint32), np.dtype(np.int32))
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
                indices = place.view(np.int32)
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
        return self._get_sparse_ones() + self._get_sparse_multi()

    def todense(self) -> npt.NDArray[np.int32]:
        """
        To dense ndarray
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
        a = np.subtract(self.place_multi[1:], self.place_multi[:-1], dtype=int)
        a[self.multi_idx[1:-1] - 1] = 1
        if np.any(a <= 0):
            return False
        a = np.subtract(self.place_ones[1:], self.place_ones[:-1], dtype=int)
        a[self.ones_idx[1:-1] - 1] = 1
        return not np.any(a <= 0)

    @deprecated("Use has_sorted_indices() instead.")
    def check_indices_ordered(self) -> bool:
        return self.has_sorted_indices()

    def sort_indices(self) -> None:
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


def _write_bin(datas: Sequence[PatternsSOneBase], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists")
    num_data = np.sum([data.num_data for data in datas])
    num_pix = datas[0].num_pix
    with path.open("wb") as fptr:
        header = np.zeros((256), dtype="i4")
        header[:2] = [num_data, num_pix]
        header.tofile(fptr)
        for g in PatternsSOne.ATTRS:
            for data in datas:
                getattr(data, g).tofile(fptr)


def _write_bytes(datas: Sequence[PatternsSOneBase], path: io.BytesIO) -> None:
    num_data = np.sum([data.num_data for data in datas])
    num_pix = datas[0].num_pix

    header = np.zeros((256), dtype="i4")
    header[:2] = [num_data, num_pix]
    path.write(header.tobytes())
    for g in PatternsSOne.ATTRS:
        for data in datas:
            path.write(getattr(data, g).tobytes())


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
    direct_zstd = position_encoding == "delta" and compression == "zstd" and shuffle and all(size > 0 for size in (num_data, num_ones, num_multi)) and direct_zstd_available()
    chunk_bytes = int(os.environ.get("EMCFILE_H5_DIRECT_CHUNK_BYTES", str(16 << 20)))
    if chunk_bytes <= 0 or chunk_bytes % np.dtype("u4").itemsize:
        raise ValueError("EMCFILE_H5_DIRECT_CHUNK_BYTES must be a positive multiple of 4")
    workers = env_workers("EMCFILE_H5_WRITE_WORKERS", 4)
    kwargs = _h5_filter_kwargs(compression, compression_opts, shuffle)
    with path.open_group("a", "a") as (_, fp):
        assert isinstance(fp, (h5py.Group, h5py.File))
        names = ["ones", "multi", "place_ones", "place_multi", "count_multi"]
        check_remove_groups(fp, names, overwrite)
        shapes = {"ones": (num_data, "i4"), "multi": (num_data, "i4"), "place_ones": (num_ones, "u4"), "place_multi": (num_multi, "u4"), "count_multi": (num_multi, "i4")}
        datasets = {}
        for name, (size, dtype) in shapes.items():
            dataset_kwargs = dict(kwargs)
            if direct_zstd:
                dataset_kwargs["chunks"] = (max(1, min(size, chunk_bytes // 4)),)
            datasets[name] = fp.create_dataset(name, (size,), dtype=dtype, **dataset_kwargs)
        fp.attrs.update(num_pix=num_pix, num_data=num_data, version="2", position_encoding=position_encoding)
        pool = ThreadPoolExecutor(max_workers=workers) if direct_zstd else None
        slots = threading.Semaphore(max(2, workers * 2)) if direct_zstd else None
        writers = {name: PrefilteredDatasetWriter(dataset, 1 if compression_opts is None else int(compression_opts), workers, pool=pool, slots=slots) for name, dataset in datasets.items()} if direct_zstd else None
        offsets = {name: 0 for name in names}
        try:
            for data in datas:
                patterns_per_batch = max(1, buffer_size // max(1, data.nbytes // max(1, data.num_data)))
                for start in range(0, data.num_data, patterns_per_batch):
                    batch = data[start : min(start + patterns_per_batch, data.num_data)]
                    assert isinstance(batch, PatternsSOne)
                    arrays = {
                        "ones": np.asarray(batch.ones), "multi": np.asarray(batch.multi),
                        "place_ones": encode_pattern_local_delta_parallel(batch.place_ones, batch.ones, workers, check_sorted=check_sorted, accelerated=direct_zstd) if position_encoding == "delta" else np.asarray(batch.place_ones),
                        "place_multi": encode_pattern_local_delta_parallel(batch.place_multi, batch.multi, workers, check_sorted=check_sorted, accelerated=direct_zstd) if position_encoding == "delta" else np.asarray(batch.place_multi),
                        "count_multi": np.asarray(batch.count_multi),
                    }
                    for name, array in arrays.items():
                        if writers is None:
                            datasets[name][offsets[name] : offsets[name] + array.size] = array
                            offsets[name] += array.size
                        else:
                            writers[name].write(array)
                            writers[name].drain()
        finally:
            if writers is not None:
                for writer in writers.values():
                    writer.close()
            if pool is not None:
                pool.shutdown(cancel_futures=True)


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
    elif isinstance(f, H5Path):
        if h5version == "1":
            if len(datas) > 1:
                raise NotImplementedError()
            _log.warning(
                'This format has performance issue. h5version = "2" is recommended.'
            )
            return _write_h5_v1(datas[0], f, overwrite)
        elif h5version == "2":
            return _write_h5_v2(datas, f, overwrite, buffer_size, compression, compression_opts, shuffle, position_encoding, check_sorted)
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
    with path.open_group("a", "a") as (_, fp):
        assert isinstance(fp, (h5py.Group, h5py.File))
        check_remove_groups(
            fp,
            ["num_pix", "ones", "multi", "place_ones", "place_multi", "count_multi"],
            overwrite,
        )
        num_pix = fp.create_dataset("num_pix", (1,), dtype=np.int32)
        num_pix[0] = data.num_pix

        place_ones = fp.create_dataset("place_ones", (data.num_data,), dtype=dt)

        ones_idx = _count_offsets(data.ones)
        multi_idx = _count_offsets(data.multi)

        for idx, d in enumerate(np.split(data.place_ones, ones_idx[1:-1]), start):
            place_ones[idx] = d

        place_multi = fp.create_dataset("place_multi", (data.num_data,), dtype=dt)
        for idx, d in enumerate(np.split(data.place_multi, multi_idx[1:-1]), start):
            place_multi[idx] = d

        count_multi = fp.create_dataset("count_multi", (data.num_data,), dtype=dt)
        for idx, d_c in enumerate(np.split(data.count_multi, multi_idx[1:-1]), start):
            count_multi[idx] = d_c
        fp.attrs["version"] = "1"


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
            return ans
        raise ValueError(f"Unsupported casting mode: {casting}")
    elif axis == 1:
        ones = cast(csr_array, hstack([d._get_sparse_ones() for d in patterns_l]))
        multi = cast(csr_array, hstack([d._get_sparse_multi() for d in patterns_l]))
        assert ones.shape is not None
        return PatternsSOne(
            ones.shape[1],
            ones=cast(np.ndarray, ones.indptr[1:] - ones.indptr[:-1]),
            multi=cast(np.ndarray, multi.indptr[1:] - multi.indptr[:-1]),
            place_ones=cast(np.ndarray, ones.indices),
            place_multi=cast(np.ndarray, multi.indices),
            count_multi=cast(np.ndarray, multi.data),
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
