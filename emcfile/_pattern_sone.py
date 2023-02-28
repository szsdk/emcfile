from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union, cast

import h5py
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from ._h5helper import PATH_TYPE, H5Path, check_remove_groups, make_path
from ._misc import pretty_size

_log = logging.getLogger(__name__)

SPARSE_PATTERN = namedtuple(
    "SPARSE_PATTERN", ["place_ones", "place_multi", "count_multi"]
)

HANDLED_FUNCTIONS = {}


class PatternsSOne:
    """
    The class for dragonfly photon dataset. The main difference between this
    format with a normal csr sparse matrix is that the 1-photon pixels positions
    are stored separately. Then we do not need to store these values since they
    are just 1.
    """

    ATTRS = ["ones", "multi", "place_ones", "place_multi", "count_multi"]

    def __init__(
        self,
        num_pix: int,
        ones: npt.NDArray,
        multi: npt.NDArray,
        place_ones: npt.NDArray,
        place_multi: npt.NDArray,
        count_multi: npt.NDArray,
    ) -> None:
        self.ndim = 2
        self.num_pix = num_pix
        self.ones = ones
        self.multi = multi
        self.place_ones = place_ones
        self.place_multi = place_multi
        self.count_multi = count_multi
        self.update_idx()

    def update_idx(self) -> None:
        self.ones_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.ones, out=self.ones_idx[1:])
        self.multi_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self.multi, out=self.multi_idx[1:])

    def check(self) -> bool:
        if self.num_data != len(self.multi):
            raise Exception(
                f"The `multi`{len(self.multi)} has different length with `ones`({self.num_data})"
            )
        ones_total = self.ones.sum()
        if ones_total != len(self.place_ones):
            raise Exception(
                f"The expected length of `place_ones`({len(self.place_ones)}) should be {ones_total}."
            )

        multi_total = self.multi.sum()
        if multi_total != len(self.place_multi):
            raise Exception(
                f"The expected length of `place_multi`({len(self.place_multi)}) should be {multi_total}."
            )

        if multi_total != len(self.count_multi):
            raise Exception(
                f"The expected length of `place_multi`({len(self.count_multi)}) should be {multi_total}."
            )
        return True

    def __len__(self) -> int:
        return self.num_data

    def sparse_pattern(self, idx: int) -> SPARSE_PATTERN:
        return SPARSE_PATTERN(
            self.place_ones[self.ones_idx[idx] : self.ones_idx[idx + 1]],
            self.place_multi[self.multi_idx[idx] : self.multi_idx[idx + 1]],
            self.count_multi[self.multi_idx[idx] : self.multi_idx[idx + 1]],
        )

    @property
    def num_data(self) -> int:
        return len(self.ones)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.num_data, self.num_pix

    def get_mean_count(self) -> float:
        return cast(int, self.sum()) / self.num_data

    def __repr__(self) -> str:
        return f"""Pattern(1-sparse) <{hex(id(self))}>
  Number of patterns: {self.num_data}
  Number of pixels: {self.num_pix}
  Mean number of counts: {self.get_mean_count():.3f}
  Size: {pretty_size(self.nbytes)}
  Sparsity: {self.sparsity() * 100:.2f} %
"""

    @property
    def nbytes(self) -> int:
        return int(np.sum([getattr(self, i).nbytes for i in PatternsSOne.ATTRS]))

    def sparsity(self) -> float:
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

    def _get_pattern(self, idx: int) -> npt.NDArray:
        if idx >= self.num_data or idx < 0:
            raise IndexError(f"{idx}")
        pattern = np.zeros(self.num_pix, "uint32")
        pattern[self.place_ones[self.ones_idx[idx] : self.ones_idx[idx + 1]]] = 1
        r = slice(*self.multi_idx[idx : idx + 2])
        pattern[self.place_multi[r]] = self.count_multi[r]
        return pattern

    def _get_subdataset(self, idx: Any) -> PatternsSOne:
        so = self._get_sparse_ones().__getitem__(*idx)
        sm = self._get_sparse_multi().__getitem__(*idx)
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

    def sum(
        self, axis: Optional[int] = None, keepdims: bool = False
    ) -> Union[npt.NDArray, int]:
        if axis is None:
            return len(self.place_ones) + cast(int, np.sum(self.count_multi))
        elif axis == 1:
            ans: npt.NDArray = self.ones + np.squeeze(
                np.asarray(self._get_sparse_multi().sum(axis=1))
            )
            return ans[:, None] if keepdims else ans
        elif axis == 0:
            ans = np.squeeze(np.asarray(self._get_sparse_ones().sum(axis=0)))
            ans = ans + np.squeeze(np.asarray(self._get_sparse_multi().sum(axis=0)))
            return ans[None, :] if keepdims else ans
        raise ValueError(f"Do not support axis={axis}.")

    def __getitem__(self, *idx: Any) -> Union[PatternsSOne, npt.NDArray]:
        if len(idx) == 1 and isinstance(idx[0], (int, np.integer)):
            return self._get_pattern(int(idx[0]))
        else:
            return self._get_subdataset(idx)

    def write(
        self,
        path: PATH_TYPE,
        *,
        h5version: str = "2",
        overwrite: bool = False,
    ) -> None:
        return write_patterns([self], path, h5version=h5version, overwrite=overwrite)

    def _get_sparse_ones(self) -> csr_matrix:
        _one = np.ones(1, "i4")
        _one = np.lib.stride_tricks.as_strided(  # type: ignore
            _one, shape=(self.place_ones.shape[0],), strides=(0,)
        )
        return csr_matrix((_one, self.place_ones, self.ones_idx), shape=self.shape)

    def _get_sparse_multi(self) -> csr_matrix:
        return csr_matrix(
            (self.count_multi, self.place_multi, self.multi_idx), shape=self.shape
        )

    def todense(self) -> npt.NDArray[np.int32]:
        """
        To dense ndarray
        """
        return cast(
            npt.NDArray[np.int32],
            np.squeeze(
                self._get_sparse_ones().todense() + self._get_sparse_multi().todense()
            ),
        )

    def __array__(self):
        return self.todense()

    def __matmul__(self, mtx: npt.ArrayLike) -> npt.NDArray:
        return self._get_sparse_ones() * mtx + self._get_sparse_multi() * mtx

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override

        # __array_function__ to handle PatternsSOne objects.

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def check_indices_ordered(self) -> bool:
        a = np.subtract(self.place_multi[1:], self.place_multi[:-1], dtype=int)
        a[self.multi_idx[1:-1] - 1] = 1
        if np.any(a <= 0):
            return False
        a = np.subtract(self.place_ones[1:], self.place_ones[:-1], dtype=int)
        a[self.ones_idx[1:-1] - 1] = 1
        if np.any(a <= 0):
            return False
        return True

    def ensure_indices_ordered(self) -> None:
        if self.check_indices_ordered():
            return
        for i in range(self.num_data):
            s, e = self.ones_idx[i], self.ones_idx[i + 1]
            t = np.argsort(self.place_ones[s:e])
            self.place_ones[s:e] = self.place_ones[s:e][t]
            s, e = self.multi_idx[i], self.multi_idx[i + 1]
            t = np.argsort(self.place_multi[s:e])
            self.place_multi[s:e] = self.place_multi[s:e][t]
            self.count_multi[s:e] = self.count_multi[s:e][t]


def implements(np_function):
    "Register an __array_function__ implementation for PatternsSOne objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def iter_array_buffer(
    datas: list[PatternsSOne], buffer_size: int, g: str
) -> Iterable[npt.NDArray]:
    buffer = []
    nbytes = 0
    for a in datas:
        ag = getattr(a, g)
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


def _write_bin(datas: list[PatternsSOne], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise Exception(f"{path} exists")
    num_data = np.sum([data.num_data for data in datas])
    num_pix = datas[0].num_pix
    with path.open("wb") as fptr:
        header = np.zeros((256), dtype="i4")
        header[:2] = [num_data, num_pix]
        header.tofile(fptr)
        for g in PatternsSOne.ATTRS:
            for data in datas:
                getattr(data, g).tofile(fptr)


def _write_h5_v2(
    datas: list[PatternsSOne], path: H5Path, overwrite: bool, buffer_size: int
) -> None:
    num_ones = np.sum([d.ones.sum() for d in datas])
    num_multi = np.sum([d.multi.sum() for d in datas])
    num_data = np.sum([data.num_data for data in datas])
    num_pix = datas[0].num_pix
    with path.open_group("a", "a") as (_, fp):
        check_remove_groups(
            fp, ["ones", "multi", "place_ones", "place_multi", "count_multi"], overwrite
        )
        fp.create_dataset("ones", (num_data,), dtype="i4")
        fp.create_dataset("multi", (num_data,), dtype="i4")
        fp.create_dataset("place_ones", (num_ones,), dtype="i4")
        fp.create_dataset("place_multi", (num_multi,), dtype="i4")
        fp.create_dataset("count_multi", (num_multi,), dtype="i4")
        fp.attrs["num_pix"] = num_pix
        fp.attrs["num_data"] = num_data
        fp.attrs["version"] = "2"
        for g in PatternsSOne.ATTRS:
            n = 0
            for a in iter_array_buffer(datas, buffer_size, g):
                fp[g][n : n + a.shape[0]] = a
                n += a.shape[0]


def write_patterns(
    datas: list[PatternsSOne],
    path: PATH_TYPE,
    *,
    h5version: str = "2",
    overwrite: bool = False,
    buffer_size: int = 1073741824,  # 2 ** 30 bytes = 1 GB
) -> None:
    # TODO: performance test
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
            return _write_h5_v2(datas, f, overwrite, buffer_size)
        else:
            raise ValueError(f"The h5version(={h5version}) should be '1' or '2'.")
    raise ValueError(f"Wrong file name {path}")


def _write_h5_v1(
    data: PatternsSOne,
    path: H5Path,
    overwrite: bool,
    start: int = 0,
    end: Optional[int] = None,
) -> None:
    dt = h5py.special_dtype(vlen=np.int32)
    with path.open_group("a", "a") as (_, fp):
        check_remove_groups(
            fp,
            ["num_pix", "ones", "multi", "place_ones", "place_multi", "count_multi"],
            overwrite,
        )
        num_pix = fp.create_dataset("num_pix", (1,), dtype=np.int32)
        num_pix[0] = data.num_pix

        place_ones = fp.create_dataset("place_ones", (data.num_data,), dtype=dt)

        for idx, d in enumerate(np.split(data.place_ones, data.ones_idx[1:-1]), start):
            place_ones[idx] = d

        place_multi = fp.create_dataset("place_multi", (data.num_data,), dtype=dt)
        for idx, d in enumerate(
            np.split(data.place_multi, data.multi_idx[1:-1]), start
        ):
            place_multi[idx] = d

        count_multi = fp.create_dataset("count_multi", (data.num_data,), dtype=dt)
        for idx, d in enumerate(
            np.split(data.count_multi, data.multi_idx[1:-1]), start
        ):
            count_multi[idx] = d
        fp.attrs["version"] = "1"


@implements(np.concatenate)
def concatenate_PatternsSOne(
    patterns_l: list[PatternsSOne], casting="safe"
) -> PatternsSOne:
    "stack pattern sets together"
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
    elif casting == "destroy":
        ans = patterns_l.pop(0)
        while len(patterns_l) > 0:
            pat = patterns_l.pop(0)
            pat = {g: getattr(pat, g) for g in PatternsSOne.ATTRS}
            for g in PatternsSOne.ATTRS:
                b = pat.pop(g)
                a = getattr(ans, g)
                a.resize(a.shape[0] + b.shape[0], refcheck=False)
                a[a.shape[0] - b.shape[0] :] = b[:]
    else:
        raise Exception(casting)
    return ans
