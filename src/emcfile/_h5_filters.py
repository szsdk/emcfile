"""Lazy helpers for optional standard HDF5 filter plugins."""

from __future__ import annotations

from typing import Any


FAST_EXTRA_MESSAGE = "HDF5 Zstd support requires the optional 'hdf5-fast' dependencies. Install with: pip install 'emcfile[hdf5-fast]'"


def hdf5plugin() -> Any:
    try:
        import hdf5plugin as plugin
    except ModuleNotFoundError as exc:
        raise RuntimeError(FAST_EXTRA_MESSAGE) from exc
    return plugin


def needs_zstd_filter(dataset: Any) -> bool:
    plist = dataset.id.get_create_plist()
    return any(plist.get_filter(index)[0] == 32015 for index in range(plist.get_nfilters()))


def ensure_filter_for(dataset: Any) -> None:
    """Register Zstd before h5py attempts to decode a matching dataset."""
    if needs_zstd_filter(dataset):
        hdf5plugin()


def ensure_group_filters(group: Any) -> None:
    for value in group.values():
        if hasattr(value, "id"):
            ensure_filter_for(value)
