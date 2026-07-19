"""Compatibility alias for :mod:`emcfile._emc_patterns`."""

import sys
import warnings

from . import _emc_patterns as _canonical_module

warnings.warn(
    "emcfile._pattern_sone is deprecated; import emcfile._emc_patterns instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _canonical_module
