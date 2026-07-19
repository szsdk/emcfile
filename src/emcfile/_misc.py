from typing import List, Tuple

from typing_extensions import deprecated

_units = ["B", "KB", "MB", "GB", "TB", "PB"]


def pretty_size(num_bytes: float) -> str:
    """
    Converts a size in bytes to a human-readable string with an appropriate unit.

    This function takes a size in bytes and formats it into a more readable
    string, using units such as KB, MB, GB, etc.

    Parameters
    ----------
    num_bytes
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
    unit_index = 0
    while num_bytes >= 1024:
        num_bytes /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{num_bytes} B"
    else:
        return f"{num_bytes:.2f} {_units[unit_index]}"


def split_range(start: int, stop: int, num_chunks: int) -> List[Tuple[int, int]]:
    """
    Divides a numerical range into a specified number of nearly equal sub-ranges.

    This function is useful for splitting a large range of numbers into smaller,
    more manageable chunks, for example, for parallel processing.

    Parameters
    ----------
    start
        The starting integer of the range.
    stop
        The ending integer of the range.
    num_chunks
        The number of sub-ranges to divide the main range into.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples, where each tuple represents a sub-range.

    Raises
    ------
    ValueError
        If `num_chunks` is not a positive integer.

    Examples
    --------
    >>> from emcfile._misc import split_range

    >>> split_range(0, 10, 3)
    [(0, 4), (4, 7), (7, 10)]

    >>> split_range(0, 10, 5)
    [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
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
