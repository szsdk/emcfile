"""Compatibility exports for the former mixed-purpose utility module."""

import warnings

from ._formatting import pretty_size
from ._indexing import divide_range, split_range

warnings.warn(
    "emcfile._misc is deprecated; import emcfile._formatting or "
    "emcfile._indexing instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["divide_range", "pretty_size", "split_range"]
