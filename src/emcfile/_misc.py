from typing import List, Tuple

_units = ["B", "KB", "MB", "GB", "TB", "PB"]


def pretty_size(s: float) -> str:
    """
    Converts a size in bytes to a human-readable string with an appropriate unit.

    This function takes a size in bytes and formats it into a more readable
    string, using units such as KB, MB, GB, etc.

    Parameters
    ----------
    s
        The size in bytes.

    Returns
    -------
    str
        A human-readable string representing the size.

    Examples
    --------
    >>> from emcfile._misc import pretty_size

    >>> pretty_size(1024)
    '1.00 KB'

    >>> pretty_size(1234567)
    '1.18 MB'

    >>> pretty_size(500)
    '500.0 B'
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
    Divides a numerical range into a specified number of nearly equal sub-ranges.

    This function is useful for splitting a large range of numbers into smaller,
    more manageable chunks, for example, for parallel processing.

    Parameters
    ----------
    s
        The starting integer of the range.
    e
        The ending integer of the range.
    n
        The number of sub-ranges to divide the main range into.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples, where each tuple represents a sub-range.

    Raises
    ------
    ValueError
        If `n` is not a positive integer.

    Examples
    --------
    >>> from emcfile._misc import divide_range

    >>> divide_range(0, 10, 3)
    [(0, 4), (4, 7), (7, 10)]

    >>> divide_range(0, 10, 5)
    [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
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
