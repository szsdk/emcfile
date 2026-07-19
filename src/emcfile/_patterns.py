"""Compatibility alias for :mod:`emcfile._pattern_factory`."""

import sys
import warnings

from . import _pattern_factory as _canonical_module

warnings.warn(
    "emcfile._patterns is deprecated; import emcfile._pattern_factory instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _canonical_module
