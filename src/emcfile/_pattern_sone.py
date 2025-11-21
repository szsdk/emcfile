from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import h5py
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, hstack

from ._h5helper import PATH_TYPE, H5Path, check_remove_groups, make_path
from ._misc import pretty_size
from ._utils import concat_continous

_log = logging.getLogger(__name__)


class SPARSE_PATTERN(NamedTuple):
    num_pix: int
    place_ones: npt.NDArray[np.uint32]
    place_multi: npt.NDArray[np.uint32]
    count_multi: npt.NDArray[np.int32]


HANDLED_FUNCTIONS: Dict[Callable[..., Any], Callable[..., Any]] = {}


TRANGE = slice | npt.NDArray[np.bool_ | np.int32 | np.int64 | np.uint32 | np.uint64]


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


class PatternsSOne:
    """
    Represents a collection of diffraction patterns in a sparse format.

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
            self.num_pix,
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
        axis: Optional[int] = None,
        keepdims: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Union[
        npt.NDArray[Any],
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        int,
        float,
    ]:
        if axis is None:
            return cast(
                int, len(self.place_ones) + np.sum(self.count_multi, dtype=dtype)
            )
        elif axis == 1:
            ans = self.ones.astype(dtype, copy=True)
            ans += np.squeeze(self._get_sparse_multi().sum(axis=1, dtype=dtype))
            return ans[:, None] if keepdims else ans
        elif axis == 0:
            ans: npt.NDArray[Any] = np.zeros(self.num_pix, dtype=dtype)
            np.add.at(ans, self.place_ones, 1)
            np.add.at(ans, self.place_multi, self.count_multi)
            return ans[None, :] if keepdims else ans
        raise ValueError(f"Do not support axis={axis}.")

    @overload
    def __getitem__(self, index: int | np.integer) -> npt.NDArray[np.int32]: ...

    @overload
    def __getitem__(self, index: TRANGE) -> PatternsSOne: ...

    @overload
    def __getitem__(self, index: tuple[TRANGE, TRANGE]) -> PatternsSOne: ...

    def _get_subdataset0(self, i) -> PatternsSOne:
        if len(i) == 0:
            return _zeros((0, self.num_pix))
        c = concat_continous(i)
        multi_s = self.multi_idx[c]
        return PatternsSOne(
            num_pix=self.num_pix,
            ones=self.ones[i],
            place_ones=np.concatenate(
                [self.place_ones[s:e] for s, e in self.ones_idx[c]]
            ),
            multi=self.multi[i],
            place_multi=np.concatenate([self.place_multi[s:e] for s, e in multi_s]),
            count_multi=np.concatenate([self.count_multi[s:e] for s, e in multi_s]),
        )

    def __getitem__(
        self,
        index: int | np.integer | TRANGE | tuple[TRANGE, TRANGE],
    ) -> Union[npt.NDArray[np.int32], PatternsSOne]:
        match index:
            case int() | np.integer():
                return self._get_pattern(int(index))
            case np.ndarray() if np.issubdtype(index.dtype, bool):
                return self._get_subdataset0(np.where(index)[0])
            case np.ndarray() if np.issubdtype(index.dtype, np.integer):
                return self._get_subdataset0(index)
            case slice():
                return self._get_subdataset((index,))
            case _:
                return self._get_subdataset(index)

    def write(
        self,
        path: Union[PATH_TYPE, io.BytesIO],
        *,
        h5version: str = "2",
        overwrite: bool = False,
        compression: Union[None, int, str] = None,
    ) -> None:
        return write_patterns(
            [self],
            path,
            h5version=h5version,
            overwrite=overwrite,
            compression=compression,
        )

    def _get_sparse_ones(self) -> csr_array:
        _one = np.ones(1, "i4")
        _one = np.lib.stride_tricks.as_strided(
            _one, shape=(self.place_ones.shape[0],), strides=(0,)
        )
        return csr_array((_one, self.place_ones, self.ones_idx), shape=self.shape)

    def _get_sparse_multi(self) -> csr_array:
        return csr_array(
            (self.count_multi, self.place_multi, self.multi_idx), shape=self.shape
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

    def __array__(self) -> npt.NDArray[np.int32]:
        return self.todense()

    def __matmul__(self, mtx: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return cast(
            npt.NDArray[Any],
            self._get_sparse_ones() @ mtx + self._get_sparse_multi() @ mtx,
        )

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[Type[object]],
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

    def check_indices_ordered(self) -> bool:
        a = np.subtract(self.place_multi[1:], self.place_multi[:-1], dtype=int)
        a[self.multi_idx[1:-1] - 1] = 1
        if np.any(a <= 0):
            return False
        a = np.subtract(self.place_ones[1:], self.place_ones[:-1], dtype=int)
        a[self.ones_idx[1:-1] - 1] = 1
        return not np.any(a <= 0)

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


FT = TypeVar("FT", bound=Callable[..., Any])


def implements(np_function: Callable[..., Any]) -> Callable[[FT], FT]:
    "Register an __array_function__ implementation for PatternsSOne objects."

    def decorator(func: FT) -> FT:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def iter_array_buffer(
    datas: Sequence[PatternsSOneBase], buffer_size: int, g: str
) -> Iterable[Union[npt.NDArray[np.int32], npt.NDArray[np.uint32]]]:
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


def _write_bin(datas: Sequence[PatternsSOneBase], path: Path, overwrite: bool) -> None:
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


def _write_bytes(datas: Sequence[PatternsSOneBase], path: io.BytesIO) -> None:
    num_data = np.sum([data.num_data for data in datas])
    num_pix = datas[0].num_pix

    header = np.zeros((256), dtype="i4")
    header[:2] = [num_data, num_pix]
    path.write(header.tobytes())
    for g in PatternsSOne.ATTRS:
        for data in datas:
            path.write(getattr(data, g).tobytes())


def _write_h5_v2(
    datas: Sequence[PatternsSOneBase],
    path: H5Path,
    overwrite: bool,
    buffer_size: int,
    compression: Union[None, int, str] = None,
) -> None:
    num_ones = np.sum([d.ones.sum() for d in datas])
    num_multi = np.sum([d.multi.sum() for d in datas])
    num_data = np.sum([data.num_data for data in datas])
    num_pix = datas[0].num_pix
    with path.open_group("a", "a") as (_, fp):
        assert isinstance(fp, (h5py.Group, h5py.File))
        check_remove_groups(
            fp, ["ones", "multi", "place_ones", "place_multi", "count_multi"], overwrite
        )
        fp.create_dataset("ones", (num_data,), dtype="i4", compression=compression)
        fp.create_dataset("multi", (num_data,), dtype="i4", compression=compression)
        fp.create_dataset(
            "place_ones", (num_ones,), dtype="i4", compression=compression
        )
        fp.create_dataset(
            "place_multi", (num_multi,), dtype="i4", compression=compression
        )
        fp.create_dataset(
            "count_multi", (num_multi,), dtype="i4", compression=compression
        )
        fp.attrs["num_pix"] = num_pix
        fp.attrs["num_data"] = num_data
        fp.attrs["version"] = "2"
        for g in PatternsSOne.ATTRS:
            n = 0
            for a in iter_array_buffer(datas, buffer_size, g):
                fg = fp[g]
                assert isinstance(fg, h5py.Dataset)
                fg[n : n + a.shape[0]] = a
                n += a.shape[0]


def write_patterns(
    datas: Sequence[PatternsSOneBase],
    path: Union[PATH_TYPE, io.BytesIO],
    *,
    h5version: str = "2",
    overwrite: bool = False,
    buffer_size: int = 1073741824,  # 2 ** 30 bytes = 1 GB
    compression: Union[None, int, str] = None,
) -> None:
    # TODO: performance test
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
            return _write_h5_v2(datas, f, overwrite, buffer_size, compression)
        else:
            raise ValueError(f"The h5version(={h5version}) should be '1' or '2'.")
    raise ValueError(f"Wrong file name {path}")


def _write_h5_v1(
    data: PatternsSOneBase,
    path: H5Path,
    overwrite: bool,
    start: int = 0,
    end: Optional[int] = None,
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

        ones_idx = np.zeros(data.num_data + 1, dtype="u8")
        np.cumsum(data.ones, out=ones_idx[1:])
        multi_idx = np.zeros(data.num_data + 1, dtype="u8")
        np.cumsum(data.multi, out=multi_idx[1:])

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
def concatenate_PatternsSOne(
    patterns_l: "Sequence[PatternsSOne]", axis: int = 0, casting: str = "safe"
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
        raise Exception(casting)
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


def _full(shape: Tuple[int, int], val: int) -> PatternsSOne:
    num_data, num_pix = shape
    return PatternsSOne(
        num_pix,
        np.zeros(num_data, dtype=np.uint32),
        np.full(num_data, num_pix, dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.resize(np.arange(num_pix, dtype=np.uint32), num_pix * num_data),
        np.full(num_pix * num_data, val, dtype=np.int32),
    )


def _ones(shape: Tuple[int, int]) -> PatternsSOne:
    num_data, num_pix = shape
    return PatternsSOne(
        num_pix,
        np.full(num_data, num_pix, dtype=np.uint32),
        np.zeros(num_data, dtype=np.uint32),
        np.resize(np.arange(num_pix, dtype=np.uint32), num_pix * num_data),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.int32),
    )


def _zeros(shape: Tuple[int, int]) -> PatternsSOne:
    num_data, num_pix = shape
    return PatternsSOne(
        num_pix,
        np.zeros(num_data, dtype=np.uint32),
        np.zeros(num_data, dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.int32),
    )
