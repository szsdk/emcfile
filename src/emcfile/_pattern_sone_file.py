from __future__ import annotations

import glob
import io
import logging
import os
from collections.abc import Sequence
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Any, Optional, Union, cast, overload

import h5py
import numpy as np
import numpy.typing as npt

from ._h5helper import PATH_TYPE, H5Path, h5path, make_path
from ._html_display import html_card
from ._misc import pretty_size
from ._pattern_sone import (
    SPARSE_PATTERN,
    TRANGE,
    PatternsSOne,
    PatternsSOneBase,
    _count_offsets,
    write_patterns,
)
from ._utils import contiguous_ranges

__all__ = [
    "EMCBinaryPatternFile",
    "EMCPatternCollection",
    "FileBackedEMCPatterns",
    "HDF5PatternFile",
    "LegacyHDF5PatternFile",
    "PatternsSOneEMC",
    "PatternsSOneH5",
    "PatternsSOneList",
    "file_patterns",
    "open_patterns",
]
_log = logging.getLogger(__name__)

I4 = np.dtype("i4").itemsize

INDEX_ARRAY = npt.NDArray[np.integer[Any]]


def read_indexed_array(
    file_obj: Union[BufferedReader, BytesIO],
    index_ranges: INDEX_ARRAY,
    offsets: INDEX_ARRAY,
    current_offset: int,
) -> tuple[npt.NDArray[np.int32], int]:
    if len(index_ranges) == 1:
        start, stop = index_ranges[0]
        stop = offsets[stop]
        start = offsets[start]
        file_obj.seek(I4 * (int(start) - current_offset), os.SEEK_CUR)
        return np.frombuffer(
            file_obj.read(int(stop - start) * I4),
            count=int(stop - start),
            dtype=np.int32,
        ), int(stop) - int(offsets[-1])

    arrays = []
    for start, stop in index_ranges:
        stop = offsets[stop]
        start = offsets[start]
        file_obj.seek(I4 * (int(start) - current_offset), os.SEEK_CUR)
        arrays.append(
            np.frombuffer(file_obj.read(int(stop - start) * I4), dtype=np.int32)
        )
        current_offset = int(stop)
    return (
        np.concatenate(arrays) if arrays else np.array([], np.int32),
        int(current_offset) - int(offsets[-1]),
    )


def read_patterns(
    file_obj: Union[BufferedReader, BytesIO],
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
    def __getitem__(self, index: int | np.integer) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE) -> PatternsSOne: ...

    @overload
    def __getitem__(self, index: tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: int | np.integer | TRANGE | tuple[TRANGE, TRANGE],
    ) -> npt.NDArray[np.int32] | PatternsSOne:
        match index:
            case (ax0, ax1):
                return self[ax0][:, ax1]
            case int() | np.integer():
                index_ranges = np.array([[index, index + 1]])
            case np.ndarray() if np.issubdtype(index.dtype, bool):
                index_ranges = contiguous_ranges(np.where(index)[0])
            case np.ndarray():
                index_ranges = contiguous_ranges(index)
            case slice():
                start = 0 if index.start is None else index.start
                stop = self.num_data if index.stop is None else index.stop
                if index.step is None or index.step == 1:
                    index_ranges = np.array([(start, stop)])
                else:
                    index_ranges = np.array(
                        [(i, i + 1) for i in range(start, stop, index.step)]
                    )

        place_ones, place_multi, count_multi = self._read_patterns(index_ranges)
        match index:
            case int() | np.integer():
                ans = np.zeros(self.num_pix, np.int32)
                ans[place_ones] = 1
                ans[place_multi] = count_multi
                return ans
            case _:
                return PatternsSOne(
                    self.num_pix,
                    self.ones[index],
                    self.multi[index],
                    place_ones,
                    place_multi,
                    count_multi,
                )

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        return self[index : index + 1].sparse_pattern(0)

    def sum(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
        dtype: Optional[npt.DTypeLike] = None,
        chunk_size: Optional[int] = None,
    ) -> Union[int, float, npt.NDArray[Any]]:
        if chunk_size is None:
            chunk_size = max(8, int(self.nbytes / 100_000_000))  # about 100MB
        sums = [
            self[i : min(i + chunk_size, self.num_data)].sum(axis=axis, dtype=dtype)
            for i in range(0, self.num_data, chunk_size)
        ]
        if axis == 1:
            ans_1 = np.concatenate(sums)
            return ans_1.reshape(-1, 1) if keepdims else ans_1
        if axis == 0:
            ans = np.sum(cast(list[npt.NDArray[Any]], sums), axis=0)
            return cast(npt.NDArray[Any], ans.reshape(1, -1) if keepdims else ans)
        return cast(Union[float, int], np.sum(cast(list[Union[int, float]], sums)))


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
            self.num_data = num_data
            self.num_pix = num_pix
        else:
            with open(self._fn, "rb") as fin:
                self.num_data = int(np.fromfile(fin, dtype=np.int32, count=1)[0])
                self.num_pix = int(np.fromfile(fin, dtype=np.int32, count=1)[0])
        self.ndim = 2
        self._init_idx = False

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            return (
                cast(
                    np.ndarray, np.fromfile(fin, dtype=np.uint32, count=self.num_data)
                ),
                cast(
                    np.ndarray, np.fromfile(fin, dtype=np.uint32, count=self.num_data)
                ),
            )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        with self._fn.open("rb") as fin:
            return read_patterns(fin, index_ranges, self.ones_idx, self.multi_idx)

    def open(self) -> PatternsSOneEMCReadBuffer:
        return PatternsSOneEMCReadBuffer(self._fn)

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            fin.seek(I4 * self.num_data * 2, os.SEEK_CUR)
            return cast(
                np.ndarray, np.fromfile(fin, dtype=np.uint32, count=self.ones.sum())
            )

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        self.init_idx()
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            fin.seek(I4 * (self.num_data * 2 + self.ones_idx[-1]), os.SEEK_CUR)
            return cast(
                np.ndarray, np.fromfile(fin, dtype=np.uint32, count=self.multi.sum())
            )

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        self.init_idx()
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            fin.seek(
                I4 * (self.num_data * 2 + self.ones_idx[-1] + self.multi_idx[-1]),
                os.SEEK_CUR,
            )
            return cast(
                np.ndarray, np.fromfile(fin, dtype=np.int32, count=self.multi.sum())
            )


class _PatternsSOneBytes(PatternsSOneFile):
    HEADER_BYTES = 1024

    def __init__(self, fn: BytesIO):
        self._fn = fn
        self._fn.seek(0)
        self.num_data = int(np.frombuffer(self._fn.read(4), dtype=np.int32, count=1)[0])
        self.num_pix = int(np.frombuffer(self._fn.read(4), dtype=np.int32, count=1)[0])
        self.ndim = 2
        self._init_idx = False

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        self._fn.seek(1024)
        return (
            cast(
                np.ndarray,
                np.frombuffer(
                    self._fn.read(I4 * self.num_data),
                    dtype=np.int32,
                    count=self.num_data,
                ),
            ),
            cast(
                np.ndarray,
                np.frombuffer(
                    self._fn.read(I4 * self.num_data),
                    dtype=np.int32,
                    count=self.num_data,
                ),
            ),
        )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        self._fn.seek(0)
        return read_patterns(self._fn, index_ranges, self.ones_idx, self.multi_idx)


class PatternsSOneEMCReadBuffer(PatternsSOneEMC):
    def __init__(self, fn: "str | Path"):
        super().__init__(fn)
        self._file_handle = self._fn.open("rb")

    def __enter__(self) -> PatternsSOneEMCReadBuffer:
        return self

    def close(self) -> None:
        self._file_handle.close()

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.close()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        fin = self._file_handle
        fin.seek(1024)
        return (
            cast(np.ndarray, np.fromfile(fin, dtype=np.int32, count=self.num_data)),
            cast(np.ndarray, np.fromfile(fin, dtype=np.int32, count=self.num_data)),
        )

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        return read_patterns(
            self._file_handle, index_ranges, self.ones_idx, self.multi_idx
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
        with self._fn.open_group() as (_, gp):
            self.num_data = int(cast(int, gp.attrs["num_data"]))
            self.num_pix = int(cast(int, gp.attrs["num_pix"]))
        self.ndim = 2
        self._init_idx = False

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        with self._fn.open_group() as (fp, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            return cast(h5py.Dataset, gp["ones"])[...], cast(h5py.Dataset, gp["multi"])[
                ...
            ]

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            place_ones = read_indexed_array_h5(
                cast(h5py.Dataset, gp["place_ones"]), index_ranges, self.ones_idx
            )
            place_multi = read_indexed_array_h5(
                cast(h5py.Dataset, gp["place_multi"]), index_ranges, self.multi_idx
            )
            count_multi = read_indexed_array_h5(
                cast(h5py.Dataset, gp["count_multi"]), index_ranges, self.multi_idx
            )
            return place_ones.view("u4"), place_multi.view("u4"), count_multi

    def open(self) -> PatternsSOneH5ReadBuffer:
        return PatternsSOneH5ReadBuffer(self._fn)

    @property
    def place_ones(self) -> npt.NDArray[np.uint32]:
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            return cast(h5py.Dataset, gp["place_ones"])[...]

    @property
    def place_multi(self) -> npt.NDArray[np.uint32]:
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            return cast(h5py.Dataset, gp["place_multi"])[...]

    @property
    def count_multi(self) -> npt.NDArray[np.int32]:
        with self._fn.open_group() as (_, gp):
            assert isinstance(gp, (h5py.Group, h5py.File))
            return cast(h5py.Dataset, gp["count_multi"])[...]


class PatternsSOneH5ReadBuffer(PatternsSOneH5):
    def __init__(self, fn: "str | Path | H5Path"):
        super().__init__(fn)
        self._file_handle = h5py.File(self._fn.fn, "r")

    def __enter__(self) -> PatternsSOneH5ReadBuffer:
        return self

    def close(self) -> None:
        assert self._file_handle is not None
        self._file_handle.close()
        self._file_handle = None

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.close()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        assert self._file_handle is not None
        gp = self._file_handle[self._fn.gn]
        assert isinstance(gp, (h5py.Group, h5py.File))
        return cast(h5py.Dataset, gp["ones"])[...], cast(h5py.Dataset, gp["multi"])[...]

    def _read_patterns(
        self, index_ranges: INDEX_ARRAY
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        assert self._file_handle is not None
        gp = self._file_handle[self._fn.gn]
        assert isinstance(gp, (h5py.Group, h5py.File))
        place_ones = read_indexed_array_h5(
            cast(h5py.Dataset, gp["place_ones"]), index_ranges, self.ones_idx
        )
        place_multi = read_indexed_array_h5(
            cast(h5py.Dataset, gp["place_multi"]), index_ranges, self.multi_idx
        )
        count_multi = read_indexed_array_h5(
            cast(h5py.Dataset, gp["count_multi"]), index_ranges, self.multi_idx
        )
        return place_ones.view("u4"), place_multi.view("u4"), count_multi


class PatternsSOneH5V1(PatternsSOneFile):
    def __init__(self, fn: "str | Path | H5Path"):
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
            place_ones = cast(h5py.Dataset, gp["place_ones"])[...]
            ones = np.array([len(i) for i in place_ones], np.uint32)
            place_ones = np.concatenate(place_ones)
            place_multi = cast(h5py.Dataset, gp["place_multi"])[...]
            multi = np.array([len(i) for i in place_multi], np.uint32)
            place_multi = np.concatenate(place_multi)
            count_multi = cast(h5py.Dataset, gp["count_multi"])[...]
            count_multi = np.concatenate(count_multi)
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
    def __getitem__(self, index: int | np.integer) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE) -> PatternsSOne: ...

    @overload
    def __getitem__(self, index: tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def __getitem__(
        self,
        index: int | np.integer | TRANGE | tuple[TRANGE, TRANGE],
    ) -> npt.NDArray[np.int32] | PatternsSOne:
        self.init_idx()
        return self._patterns[index]


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
        pattern_list: Sequence[Union[PATH_TYPE, PatternsSOneBase]],
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
        return np.concatenate([p.ones for p in self.pattern_list]), np.concatenate(
            [p.multi for p in self.pattern_list]
        )

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
        match index:
            case tuple() as axes if len(axes) == 2:
                ax0, ax1 = axes
                return self[ax0][:, ax1]
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

    def write(
        self,
        path: Union[PATH_TYPE, io.BytesIO],
        *,
        h5version: str = "2",
        overwrite: bool = False,
        compression: Union[None, int, str] = None,
        hdf5_version: Optional[str] = None,
    ) -> None:
        return write_patterns(
            self.pattern_list,
            path,
            h5version=h5version,
            overwrite=overwrite,
            compression=compression,
            hdf5_version=hdf5_version,
        )


def file_patterns(fn: Union[Sequence[PATH_TYPE], PATH_TYPE]) -> PatternsSOneFile:
    if (isinstance(fn, str) and glob.has_magic(fn)) or (
        isinstance(fn, Path) and glob.has_magic(str(fn))
    ):
        return PatternsSOneList(sorted(glob.glob(str(fn))))
    if isinstance(fn, (tuple, list)):
        return PatternsSOneList(fn)
    assert isinstance(fn, (str, Path, H5Path))
    p = make_path(fn)
    if not isinstance(p, H5Path):
        if h5py.is_hdf5(p):
            p = h5path(p, "/")
    if not isinstance(p, H5Path):
        return PatternsSOneEMC(p)
    with p.open_group() as (_, gp):
        if gp.attrs.get("version", "1") == "1":
            return PatternsSOneH5V1(p)
        return PatternsSOneH5(p)


def open_patterns(path: Union[Sequence[PATH_TYPE], PATH_TYPE]) -> PatternsSOneFile:
    """Open one or more file-backed EMC pattern sources."""
    return file_patterns(path)


# Descriptive alternatives. These remain aliases, rather than wrapper
# subclasses, to preserve exact ``isinstance`` behavior in downstream code.
FileBackedEMCPatterns = PatternsSOneFile
EMCBinaryPatternFile = PatternsSOneEMC
HDF5PatternFile = PatternsSOneH5
LegacyHDF5PatternFile = PatternsSOneH5V1
EMCPatternCollection = PatternsSOneList
