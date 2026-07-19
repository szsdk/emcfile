from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated


def contiguous_ranges(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert consecutive integers into half-open start and end ranges.

    This function takes a sorted array of integers and identifies contiguous
    blocks of numbers, representing them as `[start, end)` pairs.

    Parameters
    ----------
    a
        A 1D NumPy array of sorted integers.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape `(N, 2)`, where `N` is the number of
        contiguous blocks. Each row represents a block with the start and
        end (exclusive) values.

    Examples
    --------
    >>> import numpy as np
    >>> from emcfile._indexing import contiguous_ranges

    >>> arr = np.array([0, 1, 3, 4, 6])
    >>> contiguous_ranges(arr)
    array([[0, 2],
           [3, 5],
           [6, 7]])

    >>> arr2 = np.array([0, 1, 2, 5, 6, 8])
    >>> contiguous_ranges(arr2)
    array([[0, 3],
           [5, 7],
           [8, 9]])
    """
    if len(a) == 0:
        return np.zeros((0, 2), np.uint64)
    # b = np.abs(a[1:] - a[:-1])
    b = a[1:] - a[:-1]
    i = np.where(b != 1)[0]
    ans = np.empty((len(i) + 1, 2), np.uint64)
    ans[1:, 0] = a[i + 1]
    ans[:-1, 1] = a[i] + 1
    ans[0, 0] = a[0]
    ans[-1, -1] = a[-1] + 1
    return ans


@deprecated("Use contiguous_ranges() instead.")
def concat_continous(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return contiguous_ranges(a)


def split_range(start: int, stop: int, num_chunks: int) -> List[Tuple[int, int]]:
    """Divide a numerical range into nearly equal half-open ranges.

    Examples
    --------
    >>> split_range(0, 10, 3)
    [(0, 4), (4, 7), (7, 10)]
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks(={num_chunks}) should be positive")
    base = (stop - start) // num_chunks
    remainder = (stop - start) % num_chunks
    ranges = []
    for _ in range(remainder):
        ranges.append((start, start + base + 1))
        start += base + 1
    if base == 0:
        return ranges
    for _ in range(remainder, num_chunks):
        ranges.append((start, start + base))
        start += base
    return ranges


@deprecated("Use split_range() instead.")
def divide_range(s: int, e: int, n: int) -> List[Tuple[int, int]]:
    return split_range(s, e, n)
