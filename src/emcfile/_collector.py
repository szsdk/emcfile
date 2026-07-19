"""Compatibility alias for :mod:`emcfile._pattern_collector`."""

import sys
import warnings

from . import _pattern_collector as _canonical_module

warnings.warn(
    "emcfile._collector is deprecated; import emcfile._pattern_collector instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _canonical_module
