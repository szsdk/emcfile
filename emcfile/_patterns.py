from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
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
PATTENS_TYPE = Tuple[
    int,
    Tuple[int, int],
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _get_start_end(
    num_data: int,
    start: Union[None, int, np.integer],
    end: Union[None, int, np.integer],
) -> Tuple[int, int]:
    if start is not None and end is not None:
        return int(start), int(end)
    start = 0 if start is None else start
    end = num_data if end is None else end
    return int(start), int(end)


def _parse_h5_PatternsSOne_v2(
    path: H5Path, start: Union[None, int, np.integer], end: Union[None, int, np.integer]
) -> Tuple[int, Tuple[int, int], PatternsSOne]:
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
) -> Tuple[int, Tuple[int, int], PatternsSOne]:
    file = PatternsSOneEMC(path)
    start, end = _get_start_end(file.num_data, start, end)
    return file.num_data, (start, end), file[start:end]


def _parse_file_PatternsSOne(
    path: Union[str, Path, H5Path],
    start: Union[None, int, np.integer] = None,
    end: Union[None, int, np.integer] = None,
) -> PatternsSOne:
    f = make_path(path)
    if isinstance(f, H5Path):
        header = PatternsSOne.file_header(f)
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


def dense_to_PatternsSOne(arr: np.ndarray) -> PatternsSOne:
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
    src: Union[PATH_TYPE, np.ndarray, coo_matrix, int, np.integer],
    /,
    *,
    start: Union[None, int, np.integer] = None,
    end: Union[None, int, np.integer] = None,
) -> PatternsSOne:
    """
    The interface function for read pattern set from file or converting from a dense numpy array.

    Parameters
    ----------
    src : Union[str, Path, np.ndarray, coo_matrix]

    start : Optional[int]
        The starting pattern index

    end : Optional[int]
        The end pattern index

    Returns
    -------
    PatternsSOne:
    """
    if isinstance(src, (str, Path, H5Path)):
        ans = _parse_file_PatternsSOne(src, start=start, end=end)
        return ans
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
