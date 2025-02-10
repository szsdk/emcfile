from typing import Any

import numpy as np
import numpy.typing as npt


def concat_continous(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Example
        input [0, 1, 3, 4, 6]
        output [[0, 2], [3, 5], [6, 7]]
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
