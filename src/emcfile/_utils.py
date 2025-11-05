from typing import Any

import numpy as np
import numpy.typing as npt


def concat_continous(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Concatenates consecutive numbers in an array into start and end pairs.

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
    >>> from emcfile._utils import concat_continous

    >>> arr = np.array([0, 1, 3, 4, 6])
    >>> concat_continous(arr)
    array([[0, 2],
           [3, 5],
           [6, 7]])

    >>> arr2 = np.array([0, 1, 2, 5, 6, 8])
    >>> concat_continous(arr2)
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
