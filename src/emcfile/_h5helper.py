"""Compatibility alias for :mod:`emcfile._hdf5`."""

import sys
import warnings

from . import _hdf5 as _canonical_module

warnings.warn(
    "emcfile._h5helper is deprecated; import emcfile._hdf5 instead",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _canonical_module
