from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union, cast

import h5py
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix

from emcfile import PATH_TYPE, H5Path, check_remove_groups, make_path

from ._misc import pretty_size

_log = logging.getLogger(__name__)

PATTERNS_HEADER = namedtuple(
    "PATTERNS_HEADER", ["version", "num_data", "num_pix", "file"]
)

SPARSE_PATTERN = namedtuple(
    "SPARSE_PATTERN", ["place_ones", "place_multi", "count_multi"]
)


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
        ones: np.ndarray,
        multi: np.ndarray,
        place_ones: np.ndarray,
        place_multi: np.ndarray,
        count_multi: np.ndarray,
    ) -> None:
        self.ndim = 2
        self.num_pix = num_pix
        self._ones = ones
        self._multi = multi
        self._place_ones = place_ones
        self._place_multi = place_multi
        self._count_multi = count_multi
        self.update_idx()

    def update_idx(self) -> None:
        self._ones_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self._ones, out=self._ones_idx[1:])
        self._multi_idx = np.zeros(self.num_data + 1, dtype="u8")
        np.cumsum(self._multi, out=self._multi_idx[1:])

    def check(self) -> bool:
        if self.num_data != len(self._multi):
            raise Exception(
                f"The `multi`{len(self._multi)} has different length with `ones`({self.num_data})"
            )
        ones_total = self._ones.sum()
        if ones_total != len(self._place_ones):
            raise Exception(
                f"The expected length of `place_ones`({len(self._place_ones)}) should be {ones_total}."
            )

        multi_total = self._multi.sum()
        if multi_total != len(self._place_multi):
            raise Exception(
                f"The expected length of `place_multi`({len(self._place_multi)}) should be {multi_total}."
            )

        if multi_total != len(self._count_multi):
            raise Exception(
                f"The expected length of `place_multi`({len(self._count_multi)}) should be {multi_total}."
            )
        return True

    def __len__(self) -> int:
        return self.num_data

    def sparse_pattern(self, idx: int) -> SPARSE_PATTERN:
        return SPARSE_PATTERN(
            self._place_ones[self._ones_idx[idx] : self._ones_idx[idx + 1]],
            self._place_multi[self._multi_idx[idx] : self._multi_idx[idx + 1]],
            self._count_multi[self._multi_idx[idx] : self._multi_idx[idx + 1]],
        )

    @property
    def num_data(self) -> int:
        return len(self._ones)

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
        return int(np.sum([self.attrs(i).nbytes for i in PatternsSOne.ATTRS]))

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
            if cast(bool, np.any(self.attrs(i) != d.attrs(i))):
                return False
        return True

    @staticmethod
    def file_header(
        filename: PATH_TYPE,
    ) -> PATTERNS_HEADER:
        header: dict[str, Any] = {"file": str(filename)}
        f = make_path(filename)
        if isinstance(f, H5Path):
            with f.open_group("r", "r") as (_, fp):
                header["version"] = fp.attrs.get("version", "1")
                if header["version"] == "1":
                    header["num_data"] = len(fp["place_ones"])
                    header["num_pix"] = fp["num_pix"][:][0]
                elif header["version"] == "2":
                    header["num_data"] = fp.attrs["num_data"]
                    header["num_pix"] = fp.attrs["num_pix"]
                else:
                    raise ValueError("Cannot decide the version of the input h5 file.")
        elif isinstance(f, Path):
            header["version"] = "sparse one"
            with f.open("rb") as fin:
                header["num_data"] = np.fromfile(fin, dtype=np.int32, count=1)[0]
                header["num_pix"] = np.fromfile(fin, dtype=np.int32, count=1)[0]
        return PATTERNS_HEADER(**header)

    def _get_pattern(self, idx: int) -> np.ndarray:
        if idx >= self.num_data or idx < 0:
            raise IndexError(f"{idx}")
        pattern = np.zeros(self.num_pix, "uint32")
        pattern[self._place_ones[self._ones_idx[idx] : self._ones_idx[idx + 1]]] = 1
        r = slice(*self._multi_idx[idx : idx + 2])
        pattern[self._place_multi[r]] = self._count_multi[r]
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
    ) -> Union[np.ndarray, int]:
        if axis is None:
            return len(self._place_ones) + cast(int, np.sum(self._count_multi))
        elif axis == 1:
            ans: np.ndarray = self._ones + np.squeeze(
                np.asarray(self._get_sparse_multi().sum(axis=1))
            )
            return ans[:, None] if keepdims else ans
        elif axis == 0:
            ans = np.squeeze(np.asarray(self._get_sparse_ones().sum(axis=0)))
            ans = ans + np.squeeze(np.asarray(self._get_sparse_multi().sum(axis=0)))
            return ans[None, :] if keepdims else ans
        raise ValueError(f"Do not support axis={axis}.")

    def __getitem__(self, *idx: Any) -> Union[PatternsSOne, np.ndarray]:
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
        return write_photons([self], path, h5version=h5version, overwrite=overwrite)

    def attrs(self, g: str) -> np.ndarray:
        if g == "ones_idx":
            return self._ones_idx
        if g == "multi_idx":
            return self._multi_idx
        if g == "ones":
            return self._ones
        if g == "multi":
            return self._multi
        if g == "place_ones":
            return self._place_ones
        if g == "place_multi":
            return self._place_multi
        if g == "count_multi":
            return self._count_multi
        raise ValueError(f"What is {g}?")

    @property
    def ones(self) -> npt.NDArray:
        return self._ones

    @property
    def multi(self) -> npt.NDArray:
        return self._multi

    def _get_sparse_ones(self) -> csr_matrix:
        _one = np.ones(1, "i4")
        _one = np.lib.stride_tricks.as_strided(  # type: ignore
            _one, shape=(self._place_ones.shape[0],), strides=(0,)
        )
        return csr_matrix((_one, self._place_ones, self._ones_idx), shape=self.shape)

    def _get_sparse_multi(self) -> csr_matrix:
        return csr_matrix(
            (self._count_multi, self._place_multi, self._multi_idx), shape=self.shape
        )

    def todense(self) -> npt.NDArray:
        """
        To dense ndarray
        """
        return cast(
            np.ndarray,
            np.squeeze(
                self._get_sparse_ones().todense() + self._get_sparse_multi().todense()
            ),
        )

    def __matmul__(self, mtx: npt.ArrayLike) -> npt.NDArray:
        return self._get_sparse_ones() * mtx + self._get_sparse_multi() * mtx


def iter_array_buffer(
    datas: list[PatternsSOne], buffer_size: int, g: str
) -> Iterable[npt.NDArray]:
    buffer = []
    nbytes = 0
    for a in datas:
        ag = a.attrs(g)
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
                data.attrs(g).tofile(fptr)


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


def write_photons(
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
        num_pix = fp.create_dataset("num_pix", (1,))
        num_pix[0] = data.num_pix

        place_ones = fp.create_dataset("place_ones", (data.num_data,), dtype=dt)

        for idx, d in enumerate(
            np.split(data._place_ones, data._ones_idx[1:-1]), start
        ):
            place_ones[idx] = d

        place_multi = fp.create_dataset("place_multi", (data.num_data,), dtype=dt)
        for idx, d in enumerate(
            np.split(data._place_multi, data._multi_idx[1:-1]), start
        ):
            place_multi[idx] = d

        count_multi = fp.create_dataset("count_multi", (data.num_data,), dtype=dt)
        for idx, d in enumerate(
            np.split(data._count_multi, data._multi_idx[1:-1]), start
        ):
            count_multi[idx] = d
        fp.attrs["version"] = "1"


def vstack(patterns_l: list[PatternsSOne], destroy: bool = False) -> PatternsSOne:
    "stack pattern sets together"
    num_pix = patterns_l[0].num_pix
    for d in patterns_l:
        if d.num_pix != num_pix:
            raise ValueError(
                "The numbers of pixels of each pattern are not consistent."
            )
    if not destroy:
        ans = PatternsSOne(
            num_pix,
            *[
                np.concatenate([d.attrs(g) for d in patterns_l])
                for g in PatternsSOne.ATTRS
            ],
        )
        ans.check()
    else:
        ans = patterns_l.pop(0)
        while len(patterns_l) > 0:
            pat = patterns_l.pop(0)
            pat = {g: pat.attrs(g) for g in PatternsSOne.ATTRS}
            for g in PatternsSOne.ATTRS:
                b = pat.pop(g)
                a = ans.attrs(g)
                a.resize(a.shape[0] + b.shape[0], refcheck=False)
                a[a.shape[0] - b.shape[0] :] = b[:]
    return ans
