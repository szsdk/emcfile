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
