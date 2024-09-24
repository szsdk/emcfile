from typing import List, Tuple

_units = ["B", "KB", "MB", "GB", "TB", "PB"]


def pretty_size(s: float) -> str:
    """
    The function convert a size, a number with unit byte, to a string with suitable unit.

    Parameters
    ----------
    s : float
        The input size in bytes.

    Returns
    -------
    str:
        The return.
    """
    unit = 0
    while s >= 1024:
        s /= 1024
        unit += 1
    if unit == 0:
        return f"{s} B"
    else:
        return f"{s:.2f} {_units[unit]}"


def divide_range(s: int, e: int, n: int) -> List[Tuple[int, int]]:
    """
    This function divides a range `range(s, e)` into `n` pieces nearly equivalently.

    Parameters
    ----------
    s : int
        The start.
    e : int
        The end.
    n : int
        The number of pieces.

    Returns
    -------
    List[Tuple[int, int]]:
        The result

    Raises
    ------
    ValueError:
        `n` should be a integer larger than 0.

    See Also
    --------
    tests.test_utils.test_divide_range
    """
    if n <= 0:
        raise ValueError(f"n(={n}) should be positive")
    base = (e - s) // n
    size = (e - s) % n
    ans = []
    for _ in range(size):
        ans.append((s, s + base + 1))
        s += base + 1
    if base == 0:
        return ans
    for _ in range(size, n):
        ans.append((s, s + base))
        s += base
    return ans
