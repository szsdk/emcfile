"""Compatibility alias for :mod:`emcfile._indexing`."""

import sys
import warnings

from . import _indexing as _canonical_module

warnings.warn(
    "emcfile._utils is deprecated; import emcfile._indexing instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _canonical_module
