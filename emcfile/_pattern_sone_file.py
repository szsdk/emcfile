import logging
import os
from pathlib import Path

import numpy as np

from ._h5helper import H5Path, h5path, make_path
from ._pattern_sone import PatternsSOne

__all__ = ["PatternsSOneEMC", "PatternsSOneH5", "file_patterns"]
_log = logging.getLogger(__name__)

I4 = np.dtype("i4").itemsize


def concat_continous(a):
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


def read_indexed_array(fin, idx_con, arr_idx, e0):
    ans = []
    if len(idx_con) == 1:
        s, e = idx_con[0]
        e = arr_idx[e]
        s = arr_idx[s]
        fin.seek(I4 * int(s - e0), os.SEEK_CUR)
        ans = np.fromfile(fin, count=int(e - s), dtype=np.int32)
        return ans, int(e) - int(arr_idx[-1])

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


def read_patterns(fn: Path, idx_con, ones_idx: np.ndarray, multi_idx: np.ndarray):
    seek_start = PatternsSOneEMC.HEADER_BYTES + I4 * (len(ones_idx) - 1) * 2
    with fn.open("rb") as fin:
        fin.seek(seek_start)
        place_ones, e0 = read_indexed_array(fin, idx_con, ones_idx, 0)
        place_multi, e0 = read_indexed_array(fin, idx_con, multi_idx, e0)
        count_multi, e0 = read_indexed_array(fin, idx_con, multi_idx, e0)
        fin.seek(I4 * (-e0), os.SEEK_CUR)
        if fin.read(1):
            total = (
                seek_start + place_ones.nbytes + place_multi.nbytes + count_multi.nbytes
            )
            _log.error(
                f"START: {seek_start}, place_ones: {place_ones.nbytes}, place_multi: {place_multi.nbytes}, count_multi: {count_multi.nbytes}, total={total}; filesize = {fn.stat().st_size}; e0: {e0}"
            )
            raise ValueError(f"Error when parsing {fn}")
    return place_ones, place_multi, count_multi


class PatternsSOneFile:
    def sparsity(self) -> float:
        nbytes = (
            self.ones.nbytes
            + self.multi.nbytes
            + (self.ones.sum() + self.multi.sum() * 2) * I4
        )
        return nbytes / (4 * self.num_data * self.num_pix)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx_con = [(idx, idx + 1)]
            ans = np.zeros(self.num_pix, np.int32)
            place_ones, place_multi, count_multi = self._read_patterns(idx_con)
            ans[place_ones] = 1
            ans[place_multi] = count_multi
            return ans

        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                idx_con = concat_continous(np.where(idx)[0])
            else:
                idx_con = concat_continous(idx)
        elif isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = self.num_data if idx.stop is None else idx.stop
            if idx.step is None or idx.step == 1:
                idx_con = [(start, stop)]
            else:
                idx_con = [(i, i + 1) for i in range(start, stop, idx.step)]

        return PatternsSOne(
            self.num_pix,
            self.ones[idx],
            self.multi[idx],
            *self._read_patterns(idx_con),
        )


class PatternsSOneEMC(PatternsSOneFile):
    HEADER_BYTES = 1024

    def __init__(self, fn):
        self._fn = Path(fn)
        with open(self._fn, "rb") as fin:
            self.num_data = np.fromfile(fin, dtype=np.int32, count=1)[0]
            self.num_pix = np.fromfile(fin, dtype=np.int32, count=1)[0]
            fin.seek(1024)
            self.ones = np.fromfile(fin, dtype=np.int32, count=self.num_data)
            self.multi = np.fromfile(fin, dtype=np.int32, count=self.num_data)
        self._ones_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.ones, out=self._ones_idx[1:])
        self._multi_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.multi, out=self._multi_idx[1:])
        self.ndim = 2
        self.shape = (self.num_data, self.num_pix)

    def _read_patterns(self, idx_con):
        return read_patterns(self._fn, idx_con, self._ones_idx, self._multi_idx)


def read_indexed_array_h5(fin, idx_con, arr_idx):
    ans = []
    if len(idx_con) == 1:
        s, e = idx_con[0]
        e = arr_idx[e]
        s = arr_idx[s]
        ans = fin[s:e]
        return ans

    for s, e in idx_con:
        e = arr_idx[e]
        s = arr_idx[s]
        ans.append(fin[s:e])
    return np.concatenate(ans) if len(ans) > 0 else np.array([], np.int32)


def read_patterns_h5(fn: H5Path, idx_con, ones_idx: np.ndarray, multi_idx: np.ndarray):
    with fn.open_group() as (_, gp):
        place_ones = read_indexed_array_h5(gp["place_ones"], idx_con, ones_idx)
        place_multi = read_indexed_array_h5(gp["place_multi"], idx_con, multi_idx)
        count_multi = read_indexed_array_h5(gp["count_multi"], idx_con, multi_idx)
    return place_ones, place_multi, count_multi


class PatternsSOneH5(PatternsSOneFile):
    def __init__(self, fn):
        self._fn = h5path(fn)
        with self._fn.open_group() as (_, gp):
            self.num_data = gp.attrs["num_data"]
            self.num_pix = gp.attrs["num_pix"]
            self.ones = gp["ones"][...]
            self.multi = gp["multi"][...]
        self._ones_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.ones, out=self._ones_idx[1:])
        self._multi_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.multi, out=self._multi_idx[1:])
        self.ndim = 2
        self.shape = (self.num_data, self.num_pix)

    def _read_patterns(self, idx_con):
        return read_patterns_h5(self._fn, idx_con, self._ones_idx, self._multi_idx)


def file_patterns(fn):
    p = make_path(fn)
    if isinstance(p, H5Path):
        ish5 = True
    else:
        with open(p, "rb") as fp:
            ish5 = fp.read(8) == b"\x89HDF\r\n\x1a\n"
    if ish5:
        return PatternsSOneH5(h5path(fn))
    else:
        return PatternsSOneEMC(fn)
