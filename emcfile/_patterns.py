from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, spmatrix

from ._h5helper import PATH_TYPE, H5Path
from ._misc import divide_range
from ._pattern_sone import SPARSE_PATTERN, PatternsSOne, _full, _ones, _zeros
from ._pattern_sone_file import file_patterns

__all__ = ["patterns"]

_log = logging.getLogger(__name__)

T1 = TypeVar("T1", bound=npt.NBitBase)


def _get_start_end(
    num_data: int,
    start: Optional[int],
    end: Optional[int],
) -> tuple[int, int]:
    if start is not None and end is not None:
        return int(start), int(end)
    start = 0 if start is None else start
    end = num_data if end is None else end
    return int(start), int(end)


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


def coo_to_SOne_kernel(coo: coo_matrix) -> PatternsSOne:
    coo = coo.copy()
    idx = coo.data == 1
    c = coo_matrix((np.ones(idx.sum(), "i4"), (coo.row[idx], coo.col[idx])), coo.shape)
    coo.data[idx] = 0
    coo.eliminate_zeros()
    coo_csr = coo.tocsr()
    c_csr = c.tocsr()
    return PatternsSOne(
        coo.shape[1],
        c_csr.indptr[1:] - c_csr.indptr[:-1],
        coo_csr.indptr[1:] - coo_csr.indptr[:-1],
        c_csr.indices,
        coo_csr.indices,
        coo_csr.data,
    )


def _from_sparse_patterns(src: Sequence[SPARSE_PATTERN]) -> PatternsSOne:
    return PatternsSOne(
        num_pix=src[0].num_pix,
        ones=np.array([len(s.place_ones) for s in src]).astype(np.uint32),
        multi=np.array([len(s.place_multi) for s in src]).astype(np.uint32),
        place_ones=np.concatenate([s.place_ones for s in src]).astype(np.uint32),
        place_multi=np.concatenate([s.place_multi for s in src]).astype(np.uint32),
        count_multi=np.concatenate([s.count_multi for s in src]).astype(np.int32),
    )


def patterns(
    src: "PATH_TYPE"
    "| npt.NDArray[np.integer[T1]]"
    "| spmatrix"
    "| int"
    "| tuple[tuple[int, int], int]"
    "| PatternsSOne"
    "| Sequence[SPARSE_PATTERN]",
    /,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> PatternsSOne:
    """
    The interface function for reading a pattern set from a file, converting from a array, or
    building an empty `PatternsSOne` with a given number of pixels.

    Parameters
    ----------
    src : Union[pathlib.Path, numpy.ndarray, scipy.sparse.spmatrix, int, numpy.integer,
                PatternsSOne, list[SPARSE_PATTERN]]
        The source of the pattern set. Can be a file path, a dense numpy array,
        a sparse matrix, an integer, or another `PatternsSOne` object.

    start : Union[None, int, numpy.integer]
        The starting pattern index. Defaults to `None`.

    end : Union[None, int, numpy.integer]
        The ending pattern index. Defaults to `None`.

    Returns
    -------
    my_module.PatternsSOne
        The created pattern set.

    Raises
    ------
    Exception
        If the source cannot be parsed, or if the start and end indices are provided
        for a dense numpy array.

    Notes
    -----
    The function converts the input pattern set to a `PatternsSOne` object, which is a custom data
    structure used in our module to store and manipulate patterns. The function accepts the
    following types of input:

    - A file path: The function reads the binary pattern data from the specified file and
      creates a `PatternsSOne` object.
    - A dense numpy array: The function converts the numpy array to a `PatternsSOne` object.
    - A sparse matrix: The function converts the sparse matrix to a `PatternsSOne` object. If
      the matrix is not in COO format, it is divided into smaller chunks for processing.
    - An integer: The function creates a new `PatternsSOne` object with the specified number of
      pixels and no binary data.
    - Another `PatternsSOne` object: The function returns a subset or a copy of the input object,
      starting from index `start` and ending at index `end`.
    """

    if isinstance(src, (str, Path, H5Path)):
        return file_patterns(src)[start:end]
    if isinstance(src, PatternsSOne):
        return src[start:end]
    elif isinstance(src, (int, np.integer)):
        return PatternsSOne(
            int(src),
            np.empty((0,), np.uint32),
            np.empty((0,), np.uint32),
            np.empty((0,), np.uint32),
            np.empty((0,), np.uint32),
            np.empty((0,), np.int32),
        )
    elif (
        isinstance(src, tuple)
        and (len(src) == 2)
        and isinstance(src[0], tuple)
        and (len(src[0]) == 2)
        and isinstance(src[0][0], int)
        and isinstance(src[0][1], int)
        and isinstance(src[1], int)
    ):
        if src[1] == 0:
            return _zeros((src[0][0], src[0][1]))
        elif src[1] == 1:
            return _ones((src[0][0], src[0][1]))
        else:
            return _full((src[0][0], src[0][1]), src[1])
    elif isinstance(src, np.ndarray) and np.issubdtype(src.dtype, np.integer):
        if start is not None or end is not None:
            raise Exception()
        return dense_to_PatternsSOne(src)
    elif isinstance(src, coo_matrix):
        return coo_to_SOne_kernel(src)
    elif isinstance(src, spmatrix):
        return cast(
            PatternsSOne,
            np.concatenate(
                [
                    patterns(np.asarray((src[a:b]).todense()))
                    for a, b in divide_range(0, src.shape[0], src.shape[0] // 1024 + 1)
                ]
            ),
        )
    elif isinstance(src, list) and isinstance(src[0], SPARSE_PATTERN):
        return _from_sparse_patterns(src)
    else:
        raise Exception()
