from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Iterator, TypeVar, Union, cast, overload

import h5py
import numpy as np
import numpy.typing as npt

from ._h5helper import PATH_TYPE, H5Path, h5path, make_path
from ._pattern_sone import SPARSE_PATTERN, PatternsSOne

__all__ = ["PatternsSOneEMC", "PatternsSOneH5", "file_patterns"]
_log = logging.getLogger(__name__)

I4 = np.dtype("i4").itemsize

T1 = TypeVar("T1", bound=npt.NBitBase)


def concat_continous(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Example
        input [0, 1, 3, 4, 6]
        output [[0, 2], [3, 5], [6, 7]]
    """
    if len(a) == 0:
        return np.zeros((0, 2), np.uint64)
    b = np.abs(a[1:] - a[:-1])
    i = np.where(b != 1)[0]
    ans = np.empty((len(i) + 1, 2), np.uint64)
    ans[1:, 0] = a[i + 1]
    ans[:-1, 1] = a[i] + 1
    ans[0, 0] = a[0]
    ans[-1, -1] = a[-1] + 1
    return ans


def read_indexed_array(
    fin: BufferedReader,
    idx_con: npt.NDArray["np.integer[T1]"],
    arr_idx: npt.NDArray["np.integer[T1]"],
    e0: int,
) -> tuple[npt.NDArray[np.int32], int]:
    if len(idx_con) == 1:
        s, e = idx_con[0]
        e = arr_idx[e]
        s = arr_idx[s]
        fin.seek(I4 * int(s - e0), os.SEEK_CUR)
        return np.fromfile(fin, count=int(e - s), dtype=np.int32), int(e) - int(
            arr_idx[-1]
        )

    ans = []
    for s, e in idx_con:
        e = arr_idx[e]
        s = arr_idx[s]
        fin.seek(I4 * int(s - e0), os.SEEK_CUR)
        ans.append(np.frombuffer(fin.read(int(e - s) * I4), dtype=np.int32))
        e0 = e
    return (
        np.concatenate(ans) if len(ans) > 0 else np.array([], np.int32),
        int(e0) - int(arr_idx[-1]),
    )


def read_patterns(
    fn: Path,
    fin: BufferedReader,
    idx_con: npt.NDArray["np.integer[T1]"],
    ones_idx: npt.NDArray["np.integer[T1]"],
    multi_idx: npt.NDArray["np.integer[T1]"],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    seek_start = PatternsSOneEMC.HEADER_BYTES + I4 * (len(ones_idx) - 1) * 2
    fin.seek(seek_start)
    place_ones, e0 = read_indexed_array(fin, idx_con, ones_idx, 0)
    place_multi, e0 = read_indexed_array(fin, idx_con, multi_idx, e0)
    count_multi, e0 = read_indexed_array(fin, idx_con, multi_idx, e0)
    fin.seek(I4 * (-e0), os.SEEK_CUR)
    if fin.read(1):
        total = seek_start + place_ones.nbytes + place_multi.nbytes + count_multi.nbytes
        _log.error(
            "START: %d, place_ones: %d, place_multi: %d, count_multi: %d, total=%d;"
            "filesize = %d; e0: %d",
            seek_start,
            place_ones.nbytes,
            place_multi.nbytes,
            count_multi.nbytes,
            total,
            fn.stat().st_size,
            e0,
        )
        raise ValueError(f"Error when parsing {fn}")
    return place_ones.view("u4"), place_multi.view("u4"), count_multi


class PatternsSOneFile:
    ones: npt.NDArray[np.uint32]
    multi: npt.NDArray[np.uint32]
    num_data: int
    num_pix: int
    _init_idx: bool

    def sparsity(self) -> float:
        self.init_idx()
        nbytes = (
            self.ones.nbytes
            + self.multi.nbytes
            + (self.ones.sum() + self.multi.sum() * 2) * I4
        )
        return float(nbytes / (4 * self.num_data * self.num_pix))

    def _read_patterns(
        self, idx_con: npt.NDArray["np.integer[T1]"]
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        raise NotImplementedError()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        raise NotImplementedError()

    def init_idx(self) -> None:
        if self._init_idx:
            return
        self.ones, self.multi = self._read_ones_multi()
        self.ones_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.ones, out=self.ones_idx[1:])
        self.multi_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.multi, out=self.multi_idx[1:])
        self._init_idx = True

    @overload
    def __getitem__(self, index: int) -> npt.NDArray[np.int32]:
        ...

    @overload
    def __getitem__(
        self, index: "slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[T1]]"
    ) -> PatternsSOne:
        ...

    def __getitem__(
        self,
        index: "slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[T1]] | int",
    ) -> "npt.NDArray[np.int32] | PatternsSOne":
        if isinstance(index, (int, np.integer)):
            idx_con = np.array([[index, index + 1]])
        elif isinstance(index, np.ndarray):
            if index.dtype == bool:
                idx_con = concat_continous(np.where(index)[0])
            else:
                idx_con = concat_continous(index)
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self.num_data if index.stop is None else index.stop
            if index.step is None or index.step == 1:
                idx_con = np.array([(start, stop)])
            else:
                idx_con = np.array([(i, i + 1) for i in range(start, stop, index.step)])

        place_ones, place_multi, count_multi = self._read_patterns(idx_con)
        if not isinstance(index, (int, np.integer)):
            return PatternsSOne(
                self.num_pix,
                self.ones[index],
                self.multi[index],
                place_ones,
                place_multi,
                count_multi,
            )
        ans = np.zeros(self.num_pix, np.int32)
        ans[place_ones] = 1
        ans[place_multi] = count_multi
        return ans

    def sparse_pattern(self, index: int) -> SPARSE_PATTERN:
        return self[index : index + 1].sparse_pattern(0)


class PatternsSOneEMC(PatternsSOneFile):
    HEADER_BYTES = 1024

    def __init__(self, fn: "str | Path"):
        self._fn = Path(fn)
        with open(self._fn, "rb") as fin:
            self.num_data = np.fromfile(fin, dtype=np.int32, count=1)[0]
            self.num_pix = np.fromfile(fin, dtype=np.int32, count=1)[0]
        self.ndim = 2
        self.shape = (self.num_data, self.num_pix)
        self._init_idx = False

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        with self._fn.open("rb") as fin:
            fin.seek(1024)
            return np.fromfile(fin, dtype=np.int32, count=self.num_data), np.fromfile(
                fin, dtype=np.int32, count=self.num_data
            )

    def _read_patterns(
        self, idx_con: npt.NDArray["np.integer[T1]"]
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        with self._fn.open("rb") as fin:
            return read_patterns(self._fn, fin, idx_con, self.ones_idx, self.multi_idx)

    def open(self) -> PatternsSOneEMCReadBuffer:
        return PatternsSOneEMCReadBuffer(self._fn)


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
        return np.fromfile(fin, dtype=np.int32, count=self.num_data), np.fromfile(
            fin, dtype=np.int32, count=self.num_data
        )

    def _read_patterns(
        self, idx_con: npt.NDArray["np.integer[T1]"]
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        return read_patterns(
            self._fn, self._file_handle, idx_con, self.ones_idx, self.multi_idx
        )


def read_indexed_array_h5(
    fin: h5py.Dataset,
    idx_con: npt.NDArray["np.integer[T1]"],
    arr_idx: npt.NDArray["np.integer[T1]"],
) -> npt.NDArray[np.int32]:
    if len(idx_con) == 1:
        s, e = idx_con[0]
        e = arr_idx[e]
        s = arr_idx[s]
        return cast(npt.NDArray[np.int32], fin[s:e])

    ans = []
    for s, e in idx_con:
        e = arr_idx[e]
        s = arr_idx[s]
        ans.append(fin[s:e])
    return (
        cast(npt.NDArray[np.int32], np.concatenate(ans))
        if len(ans) > 0
        else np.array([], np.int32)
    )


class PatternsSOneH5(PatternsSOneFile):
    def __init__(self, fn: "str | Path | H5Path"):
        self._fn = h5path(fn)
        with self._fn.open_group() as (_, gp):
            self.num_data = int(gp.attrs["num_data"])
            self.num_pix = int(gp.attrs["num_pix"])
        self.ndim = 2
        self.shape = (self.num_data, self.num_pix)
        self._init_idx = False

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        with self._fn.open_group() as (fp, gp):
            return gp["ones"][...], gp["multi"][...]

    def _read_patterns(
        self, idx_con: npt.NDArray["np.integer[T1]"]
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        with self._fn.open_group() as (fp, gp):
            place_ones = read_indexed_array_h5(gp["place_ones"], idx_con, self.ones_idx)
            place_multi = read_indexed_array_h5(
                gp["place_multi"], idx_con, self.multi_idx
            )
            count_multi = read_indexed_array_h5(
                gp["count_multi"], idx_con, self.multi_idx
            )
            return place_ones.view("u4"), place_multi.view("u4"), count_multi

    def open(self) -> PatternsSOneH5ReadBuffer:
        return PatternsSOneH5ReadBuffer(self._fn)


class PatternsSOneH5ReadBuffer(PatternsSOneH5):
    def __init__(self, fn: "str | Path | H5Path"):
        super().__init__(fn)
        self._file_handle = h5py.File(self._fn.fn, "r")

    def __enter__(self) -> PatternsSOneH5ReadBuffer:
        return self

    def close(self) -> None:
        self._file_handle.close()
        self._file_handle = None

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.close()

    def _read_ones_multi(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        gp = self._file_handle[self._fn.gn]
        return gp["ones"][...], gp["multi"][...]

    def _read_patterns(
        self, idx_con: npt.NDArray["np.integer[T1]"]
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
        self.init_idx()
        gp = self._file_handle[self._fn.gn]
        place_ones = read_indexed_array_h5(gp["place_ones"], idx_con, self.ones_idx)
        place_multi = read_indexed_array_h5(gp["place_multi"], idx_con, self.multi_idx)
        count_multi = read_indexed_array_h5(gp["count_multi"], idx_con, self.multi_idx)
        return place_ones.view("u4"), place_multi.view("u4"), count_multi


class PatternsSOneH5V1(PatternsSOneFile):
    def __init__(self, fn: "str | Path | H5Path"):
        _log.warning(
            "This format has performance issue. `PatternsSOneH5` is recommended"
        )
        self._fn = h5path(fn)
        with self._fn.open_group() as (_, gp):
            self.num_data = len(gp["place_ones"])
            self.num_pix = gp["num_pix"][:][0]
        self.ndim = 2
        self.shape = (self.num_data, self.num_pix)
        self._init_idx = False

    def init_idx(self) -> None:
        if self._init_idx:
            return
        with self._fn.open_group() as (_, gp):
            place_ones = gp["place_ones"][...]
            ones = np.array([len(i) for i in place_ones], np.uint32)
            place_ones = np.concatenate(place_ones)
            place_multi = gp["place_multi"][...]
            multi = np.array([len(i) for i in place_multi], np.uint32)
            place_multi = np.concatenate(place_multi)
            count_multi = gp["count_multi"][...]
            count_multi = np.concatenate(count_multi)
        self._patterns = PatternsSOne(
            self.num_pix, ones, multi, place_ones, place_multi, count_multi
        )
        self.ones = self._patterns.ones
        self.multi = self._patterns.multi
        self.ones_idx = self._patterns.ones_idx
        self.multi_idx = self._patterns.multi_idx
        self._init_idx = True

    @overload
    def __getitem__(self, index: int) -> npt.NDArray[np.int32]:
        ...

    @overload
    def __getitem__(
        self, index: "slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[T1]]"
    ) -> PatternsSOne:
        ...

    def __getitem__(
        self,
        index: "slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer[T1]] | int",
    ) -> "npt.NDArray[np.int32] | PatternsSOne":
        self.init_idx()
        return self._patterns[index]


def file_patterns(fn: PATH_TYPE) -> PatternsSOneFile:
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
