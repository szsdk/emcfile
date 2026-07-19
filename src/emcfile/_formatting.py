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
    >>> from emcfile._formatting import pretty_size

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
