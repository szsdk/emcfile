"""Virtual-dataset composition for homogeneous EMC HDF5 v2 files."""

from __future__ import annotations

import os
from collections.abc import Sequence

import h5py
import numpy as np

from ._hdf5 import H5Path, PATH_TYPE, h5path

_NAMES = ("ones", "multi", "place_ones", "place_multi", "count_multi")
_DTYPES = {"ones": np.dtype("i4"), "multi": np.dtype("i4"), "place_ones": np.dtype("u4"), "place_multi": np.dtype("u4"), "count_multi": np.dtype("i4")}


def _dataset_path(group: str, name: str) -> str:
    return f"{group.rstrip('/')}/{name}" if group != "/" else f"/{name}"


def create_vds(output_file: PATH_TYPE, source_files: Sequence[PATH_TYPE], *, overwrite: bool = False, relative_paths: bool = True) -> H5Path:
    """Concatenate compatible v2 files without copying their payload bytes."""
    if not source_files:
        raise ValueError("source_files must not be empty")
    output, sources = h5path(output_file).resolve(), [h5path(path).resolve() for path in source_files]
    info: list[tuple[H5Path, int, int, str, dict[str, int]]] = []
    for source in sources:
        if source.fn == output.fn:
            raise ValueError("the VDS output cannot also be a source file")
        with source.open_group() as (_, group):
            if str(group.attrs.get("version", "1")) != "2":
                raise ValueError(f"{source}: expected HDF5 EMC version 2")
            if any(name not in group or group[name].ndim != 1 or group[name].dtype != _DTYPES[name] for name in _NAMES):
                raise ValueError(f"{source}: incompatible v2 datasets")
            size = int(group.attrs["num_data"])
            if len(group["ones"]) != size or len(group["multi"]) != size or len(group["place_multi"]) != len(group["count_multi"]):
                raise ValueError(f"{source}: inconsistent v2 dataset lengths")
            info.append((source, int(group.attrs["num_pix"]), size, str(group.attrs.get("position_encoding", "absolute")), {name: len(group[name]) for name in _NAMES}))
    _, pixels, _, encoding, _ = info[0]
    if any(item[1] != pixels or item[3] != encoding for item in info[1:]):
        raise ValueError("VDS sources must share num_pix and position_encoding")
    if output.fn.exists() and not overwrite:
        raise FileExistsError(f"{output.fn} exists")
    with h5py.File(output.fn, "w" if output.gn == "/" else "a") as file:
        if output.gn == "/":
            group = file
        else:
            if output.gn in file:
                if not overwrite:
                    raise FileExistsError(f"{output} exists")
                del file[output.gn]
            group = file.create_group(output.gn)
        for name in _NAMES:
            layout = h5py.VirtualLayout(shape=(sum(item[4][name] for item in info),), dtype=_DTYPES[name])
            start = 0
            for source, _, _, _, lengths in info:
                length = lengths[name]
                filename = os.path.relpath(source.fn, output.fn.parent) if relative_paths else str(source.fn)
                layout[start : start + length] = h5py.VirtualSource(filename, _dataset_path(source.gn, name), shape=(length,))
                start += length
            group.create_virtual_dataset(name, layout)
        group.attrs.update(version="2", num_pix=pixels, num_data=sum(item[2] for item in info), position_encoding=encoding, is_vds=True)
    return output
