"""Compatibility alias for :mod:`emcfile._pattern_files`."""

import sys
import warnings

from . import _pattern_files as _canonical_module

warnings.warn(
    "emcfile._pattern_sone_file is deprecated; import emcfile._pattern_files instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _canonical_module
