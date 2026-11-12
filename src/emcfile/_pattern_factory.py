from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import (
    coo_array,
    coo_matrix,
    csr_array,
    csr_matrix,
    sparray,
    spmatrix,
)
from typing_extensions import deprecated

from ._emc_accel import emc_workers, get_emc_rs
from ._emc_patterns import (
    _I32_MAX,
    _U32_MAX,
    SPARSE_PATTERN,
    PatternsSOne,
    _ensure_32bit_signed,
    _ensure_32bit_unsigned,
    _full,
    _ones,
    _zeros,
)
from ._hdf5 import PATH_TYPE, H5Path
from ._pattern_files import _PatternsSOneBytes, file_patterns

__all__ = ["patterns"]


T1 = TypeVar("T1", bound=npt.NBitBase)


def _from_dense_array(arr: npt.NDArray[np.integer[T1]]) -> PatternsSOne:
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(f"Dense pattern array must be 1D or 2D, got {arr.ndim}D")
    rs = get_emc_rs()
    if rs is not None and arr.shape[1] <= _U32_MAX + 1:
        arr_c = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
        res = rs.from_dense_2d(arr_c, emc_workers())
        if res is not None:
            num_pix, parts = res
            return PatternsSOne._from_raw_parts(int(num_pix), *parts)
    idx = arr == 1
    ones = idx.sum(axis=1)
    place_ones = idx.nonzero()[1]
    idx = arr > 1
    multi = idx.sum(axis=1)
    nz = idx.nonzero()
    place_multi = nz[1]
    count_multi = arr[nz[0], nz[1]]
    return PatternsSOne(
        arr.shape[1],
        _ensure_32bit_unsigned(ones, "ones"),
        _ensure_32bit_unsigned(multi, "multi"),
        _ensure_32bit_unsigned(place_ones, "place_ones"),
        _ensure_32bit_unsigned(place_multi, "place_multi"),
        _ensure_32bit_signed(count_multi, "count_multi"),
    )


def _from_coo_array(coo: coo_matrix | coo_array) -> PatternsSOne:
    if coo.ndim != 2:
        raise ValueError(f"Sparse pattern input must be 2D, got {coo.ndim}D")
    if not np.issubdtype(coo.data.dtype, np.integer):
        raise ValueError(f"{coo.data.dtype} is not supported")
    assert coo.shape is not None
    r_cnt = int(cast(int, coo.shape[0]))
    c_cnt = int(cast(int, coo.shape[1]))
    rs = get_emc_rs()
    if rs is not None and r_cnt <= _U32_MAX + 1 and c_cnt <= _U32_MAX + 1:
        if coo.data.dtype == np.uint64 and coo.data.size > 0:
            if int(coo.data.max()) > _I32_MAX:
                raise ValueError("count_multi values exceed int32 maximum")
        rows_i64 = np.ascontiguousarray(coo.row, dtype=np.int64)
        cols_i64 = np.ascontiguousarray(coo.col, dtype=np.int64)
        data_i64 = np.ascontiguousarray(coo.data, dtype=np.int64)
        parts = rs.from_coo(r_cnt, c_cnt, rows_i64, cols_i64, data_i64, emc_workers())
        return PatternsSOne._from_raw_parts(c_cnt, *parts)
    coo = coo.copy()
    coo.sum_duplicates()
    neg = coo.data < 0
    if np.any(neg):
        coo.data[neg] = 0
    idx = coo.data == 1
    c = coo_matrix((np.ones(idx.sum(), "i4"), (coo.row[idx], coo.col[idx])), coo.shape)
    coo.data[idx] = 0
    coo.eliminate_zeros()
    coo_csr = coo.tocsr()
    c_csr = c.tocsr()
    return PatternsSOne(
        c_cnt,
        np.diff(c_csr.indptr).astype(np.uint32),
        np.diff(coo_csr.indptr).astype(np.uint32),
        cast(np.ndarray, c_csr.indices).astype(np.uint32),
        cast(np.ndarray, coo_csr.indices).astype(np.uint32),
        _ensure_32bit_signed(cast(np.ndarray, coo_csr.data), "count_multi"),
    )


def _try_from_csr_fast(csr: csr_array | csr_matrix) -> PatternsSOne | None:
    if not np.issubdtype(csr.data.dtype, np.integer):
        raise ValueError(f"{csr.data.dtype} is not supported")
    rs = get_emc_rs()
    if rs is None or not csr.has_canonical_format:
        return None
    assert csr.shape is not None
    r_cnt = int(cast(int, csr.shape[0]))
    c_cnt = int(cast(int, csr.shape[1]))
    if c_cnt > _U32_MAX + 1:
        return None
    if csr.data.dtype == np.uint64 and csr.data.size > 0:
        if int(csr.data.max()) > _I32_MAX:
            raise ValueError("count_multi values exceed int32 maximum")
    indptr_i64 = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices_i64 = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_i64 = np.ascontiguousarray(csr.data, dtype=np.int64)
    parts = rs.from_canonical_csr(
        r_cnt, c_cnt, indptr_i64, indices_i64, data_i64, emc_workers()
    )
    return PatternsSOne._from_raw_parts(c_cnt, *parts)


@deprecated("Use patterns() with a dense array instead.")
def dense_to_PatternsSOne(
    arr: npt.NDArray[np.integer[T1]],
) -> PatternsSOne:
    return _from_dense_array(arr)


@deprecated("Use patterns() with a COO array instead.")
def coo_to_SOne_kernel(coo: coo_matrix | coo_array) -> PatternsSOne:
    return _from_coo_array(coo)


def _from_sparse_patterns(src: Sequence[SPARSE_PATTERN]) -> PatternsSOne:
    if len(src) == 0:
        raise ValueError("SPARSE_PATTERN sequence must not be empty")
    num_pix = int(src[0].num_pix)
    if num_pix < 0:
        raise ValueError("num_pix must be non-negative")
    if any(int(s.num_pix) != num_pix for s in src):
        raise ValueError("Inconsistent number of pixels in SPARSE_PATTERN sequence")
    return PatternsSOne(
        num_pix=num_pix,
        ones=np.array([len(s.place_ones) for s in src], dtype=np.uint32),
        multi=np.array([len(s.place_multi) for s in src], dtype=np.uint32),
        place_ones=np.concatenate([s.place_ones for s in src]).astype(np.uint32),
        place_multi=np.concatenate([s.place_multi for s in src]).astype(np.uint32),
        count_multi=np.concatenate([s.count_multi for s in src]).astype(np.int32),
    )


def patterns(
    src: PATH_TYPE
    | io.BytesIO
    | npt.NDArray[np.integer[T1]]
    | spmatrix
    | int
    | tuple[tuple[int, int], int]
    | PatternsSOne
    | Sequence[SPARSE_PATTERN],
    /,
    *,
    start: int | None = None,
    end: int | None = None,
) -> PatternsSOne:
    """Create an in-memory sparse photon-count array.

    Parameters
    ----------
    src
        Pattern file path, EMC binary ``BytesIO`` stream, 2D integer NumPy
        array, SciPy sparse matrix or array, or existing ``EMCPatternArray``.
        An integer creates an empty dataset with that many pixels.
        ``((num_patterns, num_pixels), value)`` creates a constant dataset.
        A nonempty list of ``SparsePattern`` records is also accepted.
    start, end : int, optional
        Inclusive start and exclusive end of the row range. Applied to file,
        byte-stream, dense-array, and existing pattern-array inputs. Other
        input forms currently ignore these arguments.

    Returns
    -------
    EMCPatternArray
        Sparse data in memory, with shape ``(num_patterns, num_pixels)``.
        Slicing an existing pattern array can share its event buffers.

    Raises
    ------
    ValueError
        If a dense array does not have an integer dtype.
    TypeError
        If the input type is unsupported.

    Notes
    -----
    Photon counts should be nonnegative integers. Dense conversion stores
    values equal to one and greater than one; negative values are omitted.
    File inputs load the selected events into memory. Use ``open_patterns``
    for file-backed access to large datasets.

    Examples
    --------
    >>> import numpy as np
    >>> import emcfile as ef
    >>> data = ef.patterns(np.array([[0, 1, 3], [1, 0, 0]], dtype=np.int32))
    >>> data.shape
    (2, 3)
    >>> data[0].tolist()
    [0, 1, 3]
    >>> ef.patterns(1024).shape
    (0, 1024)

    See Also
    --------
    open_patterns : Access pattern files without loading all events.
    EMCPatternArray : In-memory sparse representation.
    """
    match src:
        case io.BytesIO():
            return _PatternsSOneBytes(src)[start:end]
        case str() | Path() | H5Path():
            return file_patterns(src)[start:end]
        case PatternsSOne():
            return src[start:end]
        case bool() | np.bool_():
            raise TypeError(f"Unsupported pattern source type: {type(src).__name__}")
        case int() | np.integer():
            if int(src) < 0:
                raise ValueError("Number of pixels must be non-negative")
            return PatternsSOne(
                int(src),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.int32),
            )
        case (
            (int() | np.integer(), int() | np.integer()) as shape,
            int() | np.integer() as v,
        ) if not (
            isinstance(shape[0], (bool, np.bool_))
            or isinstance(shape[1], (bool, np.bool_))
            or isinstance(v, (bool, np.bool_))
        ):
            r, c = int(shape[0]), int(shape[1])
            val = int(v)
            if r < 0 or c < 0:
                raise ValueError("Pattern dimensions must be non-negative")
            if val < 0:
                raise ValueError("Constant photon count must be non-negative")
            if val == 0:
                ans = _zeros((r, c))
            elif val == 1:
                ans = _ones((r, c))
            else:
                ans = _full((r, c), val)
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
        case np.ndarray() if (
            src.dtype == object
            and src.ndim == 1
            and (len(src) == 0 or all(isinstance(i, SPARSE_PATTERN) for i in src))
        ):
            ans = _from_sparse_patterns(cast(Sequence[SPARSE_PATTERN], list(src)))
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
        case np.ndarray():
            if not np.issubdtype(src.dtype, np.integer):
                raise ValueError(f"{src.dtype} is not supported")
            if src.ndim == 2 and (start is not None or end is not None):
                src = src[start:end]
                start = None
                end = None
            ans = _from_dense_array(src)
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
        case coo_array() | coo_matrix():
            ans = _from_coo_array(src)
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
        case sparray() | spmatrix():
            if src.ndim != 2:
                raise ValueError(f"Sparse pattern input must be 2D, got {src.ndim}D")
            sliced = src[start:end] if (start is not None or end is not None) else src
            if isinstance(sliced, (csr_array, csr_matrix)):
                fast_csr = _try_from_csr_fast(sliced)
                if fast_csr is not None:
                    return fast_csr
            return _from_coo_array(sliced.tocoo())
        case list() | tuple() if len(src) == 0 or all(
            isinstance(i, SPARSE_PATTERN) for i in src
        ):
            ans = _from_sparse_patterns(cast(Sequence[SPARSE_PATTERN], src))
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
    raise TypeError(f"Unsupported pattern source type: {type(src).__name__}")
