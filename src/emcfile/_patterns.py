from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, coo_matrix, sparray, spmatrix

from ._h5helper import PATH_TYPE, H5Path
from ._misc import divide_range
from ._pattern_sone import SPARSE_PATTERN, PatternsSOne, _full, _ones, _zeros
from ._pattern_sone_file import _PatternsSOneBytes, file_patterns

__all__ = ["patterns"]


T1 = TypeVar("T1", bound=npt.NBitBase)


def dense_to_PatternsSOne(arr: npt.NDArray["np.integer[T1]"]) -> PatternsSOne:
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


def coo_to_SOne_kernel(coo: coo_matrix | coo_array) -> PatternsSOne:
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
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> PatternsSOne:
    """
    The `patterns` function is the primary interface for creating `PatternsSOne`
    objects. It can read pattern sets from files, convert from arrays, or
    initialize an empty object.

    This function is highly flexible and accepts various input types to
    streamline the process of creating pattern sets for analysis.

    Parameters
    ----------
    src
        The source of the pattern set. It can be one of the following:
        - `str` or `pathlib.Path` or `H5Path`: The path to a file containing
          the pattern data.
        - `io.BytesIO`: A stream of bytes containing the pattern data.
        - `numpy.ndarray`: A dense numpy array representing the patterns.
        - `scipy.sparse.spmatrix`: A sparse matrix representing the patterns.
        - `int`: The number of pixels, used to create an empty `PatternsSOne`
          object.
        - `tuple[tuple[int, int], int]`: A shape and a value to create a
          uniform pattern.
        - `PatternsSOne`: An existing `PatternsSOne` object.
        - `Sequence[SPARSE_PATTERN]`: A sequence of `SPARSE_PATTERN` objects.
    start
        The starting index of the patterns to read from the source.
        If `None`, reading starts from the beginning. Defaults to `None`.
    end
        The ending index of the patterns to read from the source.
        If `None`, reading continues to the end. Defaults to `None`.

    Returns
    -------
    PatternsSOne
        A `PatternsSOne` object containing the pattern data.

    Raises
    ------
    ValueError
        If the input `numpy.ndarray` has an unsupported data type.
    Exception
        If the source type is not recognized or cannot be parsed.

    See Also
    --------
    PatternsSOne : The underlying class for storing and managing patterns.
    file_patterns : A function for reading patterns from a file.

    Examples
    --------
    >>> import numpy as np
    >>> from emcfile import patterns

    Create patterns from a numpy array:
    >>> arr = np.random.randint(0, 5, size=(10, 100))
    >>> p = patterns(arr)
    >>> p.num_data
    10

    Create an empty pattern set with a specific number of pixels:
    >>> empty_p = patterns(1024)
    >>> empty_p.num_pix
    1024
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
            return _zeros(shape)
        case ((int() | np.integer(), int() | np.integer()) as shape, 1):
            return _ones(shape)
        case (
            (int() | np.integer(), int() | np.integer()) as shape,
            int() | np.integer() as v,
        ):
            return _full(shape, v)
        case np.ndarray():
            if not np.issubdtype(src.dtype, np.integer):
                raise ValueError(f"{src.dtype} is not supported")
            ans = dense_to_PatternsSOne(src)
            if (start is not None) or (end is not None):
                ans = ans[start:end]
            return ans
        case coo_array() | coo_matrix():
            return coo_to_SOne_kernel(src)
        case sparray() | spmatrix():
            return cast(
                PatternsSOne,
                np.concatenate(
                    [
                        patterns(np.asarray((src[a:b]).todense()))
                        for a, b in divide_range(
                            0, src.shape[0], src.shape[0] // 1024 + 1
                        )
                    ]
                ),
            )
        case list() | np.ndarray() if all(isinstance(i, SPARSE_PATTERN) for i in src):
            return _from_sparse_patterns(src)
    raise Exception()
