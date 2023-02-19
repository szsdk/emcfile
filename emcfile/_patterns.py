from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from scipy.sparse import coo_matrix, spmatrix

from ._h5helper import PATH_TYPE, H5Path, make_path
from ._misc import divide_range
from ._pattern_sone import PatternsSOne
from ._pattern_sone_file import PatternsSOneEMC, PatternsSOneH5

__all__ = ["patterns"]

_log = logging.getLogger(__name__)
PATTENS_TYPE = Tuple[
    int,
    Tuple[int, int],
    int,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]


def _get_start_end(
    num_data: int,
    start: Union[None, int, np.integer],
    end: Union[None, int, np.integer],
) -> tuple[int, int]:
    if start is not None and end is not None:
        return int(start), int(end)
    start = 0 if start is None else start
    end = num_data if end is None else end
    return int(start), int(end)


def _parse_h5_PatternsSOne_v2(
    path: H5Path, start: Union[None, int, np.integer], end: Union[None, int, np.integer]
) -> tuple[int, tuple[int, int], PatternsSOne]:
    file = PatternsSOneH5(path)
    start, end = _get_start_end(file.num_data, start, end)
    return file.num_data, (start, end), file[start:end]


def _parse_h5_PatternsSOne_v1(
    path: H5Path, start: Union[None, int, np.integer], end: Union[None, int, np.integer]
) -> PATTENS_TYPE:
    with path.open_group("r", "r") as (_, fp):
        num_data = len(fp["place_ones"])
        start, end = _get_start_end(num_data, start, end)
        num_pix = fp["num_pix"][:][0]
        place_ones = fp["place_ones"][start:end]
        ones = np.array([len(i) for i in place_ones])
        place_ones = np.concatenate(place_ones)
        place_multi = fp["place_multi"][start:end]
        multi = np.array([len(i) for i in place_multi])
        place_multi = np.concatenate(place_multi)
        count_multi = fp["count_multi"][start:end]
        count_multi = np.concatenate(count_multi)
    return (
        num_data,
        (start, end),
        num_pix,
        ones,
        multi,
        place_ones,
        place_multi,
        count_multi,
    )


def _parse_bin_PatternsSOne(
    path: Path, start: Union[None, int, np.integer], end: Union[None, int, np.integer]
) -> tuple[int, tuple[int, int], PatternsSOne]:
    file = PatternsSOneEMC(path)
    start, end = _get_start_end(file.num_data, start, end)
    return file.num_data, (start, end), file[start:end]


PATTERNS_HEADER = namedtuple(
    "PATTERNS_HEADER", ["version", "num_data", "num_pix", "file"]
)


def patterns_header(
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


def _parse_file_PatternsSOne(
    path: Union[str, Path, H5Path],
    start: Union[None, int, np.integer] = None,
    end: Union[None, int, np.integer] = None,
) -> PatternsSOne:
    f = make_path(path)
    if isinstance(f, H5Path):
        header = patterns_header(f)
        if header.version == "1":
            num_data, offset, *data = _parse_h5_PatternsSOne_v1(f, start, end)
            return PatternsSOne(*data)  # type: ignore
        # if header.version == "2":
        num_data, offset, dataset = _parse_h5_PatternsSOne_v2(f, start, end)
    elif isinstance(f, Path):
        num_data, offset, dataset = _parse_bin_PatternsSOne(f, start, end)

    _log.info(
        f"read dataset ({num_data} frames, {dataset.num_pix} pixels, {dataset.get_mean_count():.2f} photons/frame) from {path}"
    )
    return dataset


def dense_to_PatternsSOne(arr: npt.NDArray) -> PatternsSOne:
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


@beartype
def patterns(
    src: Union[PATH_TYPE, npt.NDArray, spmatrix, int, np.integer, PatternsSOne],
    /,
    *,
    start: Union[None, int, np.integer] = None,
    end: Union[None, int, np.integer] = None,
) -> PatternsSOne:
    """
    The interface function for reading a pattern set from a file, converting from a array, or
    building an empty `PatternsSOne` with a given number of pixels.

    Parameters
    ----------
    src : Union[pathlib.Path, numpy.ndarray, scipy.sparse.spmatrix, int, numpy.integer, PatternsSOne]
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
        ans = _parse_file_PatternsSOne(src, start=start, end=end)
        return ans
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
    elif isinstance(src, np.ndarray) and np.issubdtype(src.dtype, np.integer):
        if start is not None or end is not None:
            raise Exception()
        return dense_to_PatternsSOne(src)
    elif isinstance(src, coo_matrix):
        return coo_to_SOne_kernel(src)
    elif isinstance(src, spmatrix):
        return np.concatenate(
            [
                patterns(np.asarray((src[a:b]).todense()))
                for a, b in divide_range(0, src.shape[0], src.shape[0] // 1024 + 1)
            ]
        )
    else:
        raise Exception()
