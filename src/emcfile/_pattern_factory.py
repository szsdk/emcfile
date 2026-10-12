from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, coo_matrix, sparray, spmatrix
from typing_extensions import deprecated

from ._emc_patterns import SPARSE_PATTERN, PatternsSOne, _full, _ones, _zeros
from ._hdf5 import PATH_TYPE, H5Path
from ._indexing import split_range
from ._pattern_files import _PatternsSOneBytes, file_patterns

__all__ = ["patterns"]


T1 = TypeVar("T1", bound=npt.NBitBase)


def _from_dense_array(arr: npt.NDArray[np.integer[T1]]) -> PatternsSOne:
    idx = arr == 1
    ones = idx.sum(axis=1)
    place_ones = idx.nonzero()[1]
    idx = arr > 1
    multi = idx.sum(axis=1)
    idx = idx.nonzero()
    place_multi = idx[1]
    count_multi = arr[idx[0], idx[1]]
    return PatternsSOne(
        arr.shape[1],
        ones.astype(np.uint32),
        multi.astype(np.uint32),
        place_ones.astype(np.uint32),
        place_multi.astype(np.uint32),
        count_multi.astype(np.int32),
    )


def _from_coo_array(coo: coo_matrix | coo_array) -> PatternsSOne:
    coo = coo.copy()
    idx = coo.data == 1
    c = coo_matrix((np.ones(idx.sum(), "i4"), (coo.row[idx], coo.col[idx])), coo.shape)
    coo.data[idx] = 0
    coo.eliminate_zeros()
    coo_csr = coo.tocsr()
    c_csr = c.tocsr()
    assert coo.shape is not None
    return PatternsSOne(
        int(cast(int, coo.shape[1])),
        np.diff(c_csr.indptr),
        np.diff(coo_csr.indptr),
        cast(np.ndarray, c_csr.indices),
        cast(np.ndarray, coo_csr.indices),
        cast(np.ndarray, coo_csr.data),
    )


@deprecated("Use patterns() with a dense array instead.")
def dense_to_PatternsSOne(
    arr: npt.NDArray[np.integer[T1]],
) -> PatternsSOne:
    return _from_dense_array(arr)


@deprecated("Use patterns() with a COO array instead.")
def coo_to_SOne_kernel(coo: coo_matrix | coo_array) -> PatternsSOne:
    return _from_coo_array(coo)


def _from_sparse_patterns(src: Sequence[SPARSE_PATTERN]) -> PatternsSOne:
    return PatternsSOne(
        num_pix=int(src[0].num_pix),
        ones=np.array([len(s.place_ones) for s in src]).astype(np.uint32),
        multi=np.array([len(s.place_multi) for s in src]).astype(np.uint32),
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
        case int() | np.integer():
            return PatternsSOne(
                int(src),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.uint32),
                np.empty((0,), np.int32),
            )
        case ((int() | np.integer(), int() | np.integer()) as shape, 0):
            return _zeros(cast(tuple[int, int], shape))
        case ((int() | np.integer(), int() | np.integer()) as shape, 1):
            return _ones(cast(tuple[int, int], shape))
        case (
            (int() | np.integer(), int() | np.integer()) as shape,
            int() | np.integer() as v,
        ):
            return _full(cast(tuple[int, int], shape), int(v))
        case np.ndarray():
            if not np.issubdtype(src.dtype, np.integer):
                raise ValueError(f"{src.dtype} is not supported")
            ans = _from_dense_array(src)
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
        case coo_array() | coo_matrix():
            return _from_coo_array(src)
        case sparray() | spmatrix():
            return cast(
                PatternsSOne,
                np.concatenate(
                    [
                        patterns(np.asarray((src[a:b]).todense()))
                        for a, b in split_range(
                            0, src.shape[0], src.shape[0] // 1024 + 1
                        )
                    ]
                ),
            )
        case list() | np.ndarray() if all(isinstance(i, SPARSE_PATTERN) for i in src):
            return _from_sparse_patterns(cast(Sequence[SPARSE_PATTERN], src))
    raise TypeError(f"Unsupported pattern source type: {type(src).__name__}")
