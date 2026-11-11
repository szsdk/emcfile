"""Virtual-dataset composition for homogeneous EMC HDF5 v2 files."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np

from ._h5_accel import invalidate_cache_for_path
from ._h5_filters import needs_zstd_filter
from ._hdf5 import H5Path, PATH_TYPE, atomic_write_path, h5path

_NAMES = ("ones", "multi", "place_ones", "place_multi", "count_multi")
_DTYPES: dict[str, tuple[np.dtype[Any], ...]] = {
    "ones": (np.dtype("i4"), np.dtype("u4")),
    "multi": (np.dtype("i4"), np.dtype("u4")),
    "place_ones": (np.dtype("u4"),),
    "place_multi": (np.dtype("u4"),),
    "count_multi": (np.dtype("i4"),),
}


def _dataset_path(group: str, name: str) -> str:
    return f"{group.rstrip('/')}/{name}" if group != "/" else f"/{name}"


def create_vds(output_file: PATH_TYPE, source_files: Sequence[PATH_TYPE], *, overwrite: bool = False, relative_paths: bool = True) -> H5Path:
    """Concatenate HDF5 v2 sources using virtual datasets.

    Parameters
    ----------
    output_file : path-like
        Destination HDF5 file or ``"file.h5::/group"`` path. Must not
        refer to a source file.
    source_files : sequence of paths
        Nonempty sequence of HDF5 v2 pattern groups in concatenation
        order. All must have equal pixel counts and position encodings.
    overwrite : bool, default False
        Permit an existing output file. A root destination replaces the
        entire file; a group destination replaces only that group.
    relative_paths : bool, default True
        Store source filenames relative to the output file's directory.
        If False, store absolute source filenames.

    Returns
    -------
    H5Path
        Resolved destination path, suitable for ``open_patterns``.

    Raises
    ------
    ValueError
        If sources are empty, incompatible, or include the output file.
    FileExistsError
        If the destination file exists and overwriting is disabled.

    Notes
    -----
    Event data is referenced rather than copied. Source files must remain
    available; relative paths require preserving their locations relative
    to the output file. VDS reads use the generic HDF5 reader.
    """
    if not source_files:
        raise ValueError("source_files must not be empty")
    output, sources = h5path(output_file).resolve(), [h5path(path).resolve() for path in source_files]
    info: list[tuple[H5Path, int, int, str, dict[str, int]]] = []
    source_dtypes: dict[str, np.dtype[Any]] = {}
    uses_zstd = False
    for source in sources:
        if source.fn == output.fn:
            raise ValueError("the VDS output cannot also be a source file")
        with source.open_group() as (_, group):
            if str(group.attrs.get("version", "1")) != "2":
                raise ValueError(f"{source}: expected HDF5 EMC version 2")
            if any(
                name not in group
                or group[name].ndim != 1
                or group[name].dtype not in _DTYPES[name]
                for name in _NAMES
            ):
                raise ValueError(f"{source}: incompatible v2 datasets")
            for name in _NAMES:
                dt = group[name].dtype
                if name not in source_dtypes:
                    source_dtypes[name] = dt
                elif source_dtypes[name] != dt:
                    raise ValueError(f"{source}: inconsistent dtype for {name}")
                if needs_zstd_filter(group[name]):
                    uses_zstd = True
            if bool(group.attrs.get("uses_zstd", False)):
                uses_zstd = True
            size = int(group.attrs["num_data"])
            if len(group["ones"]) != size or len(group["multi"]) != size or len(group["place_multi"]) != len(group["count_multi"]):
                raise ValueError(f"{source}: inconsistent v2 dataset lengths")
            info.append((source, int(group.attrs["num_pix"]), size, str(group.attrs.get("position_encoding", "absolute")), {name: len(group[name]) for name in _NAMES}))
    _, pixels, _, encoding, _ = info[0]
    if any(item[1] != pixels or item[3] != encoding for item in info[1:]):
        raise ValueError("VDS sources must share num_pix and position_encoding")
    if output.gn == "/" and output.fn.exists() and not overwrite:
        raise FileExistsError(f"{output.fn} exists")

    def _populate(file: h5py.File) -> None:
        if output.gn == "/":
            group: h5py.File | h5py.Group = file
        else:
            if output.gn in file:
                if not overwrite:
                    raise FileExistsError(f"{output} exists")
                del file[output.gn]
            group = file.create_group(output.gn)
        for name in _NAMES:
            layout = h5py.VirtualLayout(shape=(sum(item[4][name] for item in info),), dtype=source_dtypes[name])
            start = 0
            for source, _, _, _, lengths in info:
                length = lengths[name]
                filename = os.path.relpath(source.fn, output.fn.parent) if relative_paths else str(source.fn)
                layout[start : start + length] = h5py.VirtualSource(filename, _dataset_path(source.gn, name), shape=(length,))
                start += length
            group.create_virtual_dataset(name, layout)
        group.attrs.update(
            version="2",
            num_pix=pixels,
            num_data=sum(item[2] for item in info),
            position_encoding=encoding,
            is_vds=True,
            uses_zstd=uses_zstd,
        )

    invalidate_cache_for_path(output.fn)
    if output.gn == "/" or not output.fn.exists():
        with atomic_write_path(output.fn) as tmp_fn:
            with h5py.File(tmp_fn, "w") as file:
                _populate(file)
    else:
        with h5py.File(output.fn, "a") as file:
            _populate(file)
    invalidate_cache_for_path(output.fn)
    return output
