from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from scipy.sparse import coo_matrix

from emcfile import (
    PATH_TYPE,
    H5Path,
    PatternsSOne,
    PatternsSOneEMC,
    PatternsSOneH5,
    make_path,
)

__all__ = ["patterns"]

_log = logging.getLogger(__name__)
PATTENS_TYPE = tuple[
    int,
    tuple[int, int],
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
    src: Union[PATH_TYPE, npt.NDArray, coo_matrix, int, np.integer, PatternsSOne],
    /,
    *,
    start: Union[None, int, np.integer] = None,
    end: Union[None, int, np.integer] = None,
) -> PatternsSOne:
    """
    The interface function for read pattern set from file or converting from a dense numpy array.

    Parameters
    ----------
    src : Union[PATH_TYPE, npt.NDArray, coo_matrix, int, np.integer, PatternsSOne]
        Create a PatternsSOne object from a source file, a dense numpy array, a coo sparse matrix,
        an integer or another `PatternsSOne` file.

    start : Union[None, int, np.integer]
        The starting pattern index

    end : Union[None, int, np.integer]
        The end pattern index

    Returns
    -------
    PatternsSOne:
        Created pattern set.

    Raises
    -------
    Exception:
        If the source cannot be parsed or the start and end indices are provided for dense numpy array.
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
    else:
        raise Exception()
