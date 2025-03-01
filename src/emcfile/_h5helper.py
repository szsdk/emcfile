from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union, cast

import h5py
import numpy as np
import numpy.typing as npt

__all__ = [
    "H5Path",
    "PATH_TYPE",
    "check_h5path",
    "check_remove_groups",
    "h5group",
    "write_array",
    "read_array",
    "h5path",
    "make_path",
    "write_obj_h5",
    "read_obj_h5",
]

_log = logging.getLogger(__name__)


def parse_h5path(fname: "str | Path") -> tuple[Path, str]:
    if not check_h5path(fname):
        raise ValueError(f"{fname} is not a valid h5path")

    fn_gn = str(fname).split("::")  # file name, group name
    if len(fn_gn) == 1:
        fn_gn.append("/")
    fn, gn = fn_gn
    gn = "/" if gn == "" else gn
    return Path(fn), gn


@contextmanager
def h5group(
    fname: str, *args: Any, add: bool = True, **kargs: Any
) -> Iterator[tuple[h5py.File, h5py.Group]]:
    """
    Parse a string to a tuple (fptr:h5py.File, group:h5py.Group)


    Parameters
    ----------
    fname : str
        fname = "example_file.h5" or "example_file.h5::some_group"
    add : bool
        Flag for whether create the group if it does not exsist.

    Raises
    ------
    Exception:
        When the fname cannot be correctly parsed.
    """
    fn, gn = parse_h5path(fname)
    with h5py.File(fn, *args, **kargs) as fptr:
        if (gn not in fptr) and add:
            fptr.create_group(gn)
        g = fptr[gn]
        if not isinstance(g, h5py.Group):
            raise Exception(f"{gn} is not a group.")
        yield fptr, g


class H5Path:
    def __init__(self, fn: "str | Path", gn: str):
        """
        H5Path

        Parameters
        ----------
        fn : Union[str, Path]
            The filename of the hdf5 file.
        gn : Union[str, Path]
            The group/datset name.
        """
        self.fn = Path(fn)
        self.gn = str(gn)

    def __truediv__(self, a: "str | Path") -> H5Path:
        return H5Path(self.fn, str(Path(self.gn) / a))

    def __iter__(self) -> Iterator["Path | str"]:
        yield self.fn
        yield self.gn

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, H5Path):
            return False
        return self.fn == other.fn and self.gn == other.gn

    def exists(self) -> bool:
        if not self.fn.exists():
            return False
        with h5py.File(self.fn, "r") as fp:
            return self.gn in fp

    def resolve(self):
        return H5Path(self.fn.resolve(), self.gn)

    @contextmanager
    def open(self, *args: Any, **kargs: Any) -> Iterator[h5py.File]:
        with h5py.File(self.fn, *args, **kargs) as fptr:
            yield fptr

    @contextmanager
    def open_group(
        self,
        mode: str = "r",
        group_mode: str = "r",
        track_order: Optional[bool] = None,
        *args: Any,
        **kargs: Any,
    ) -> Iterator[tuple[h5py.File, Union[h5py.Dataset, h5py.Group]]]:
        """
        group_mode : [r|a]
        """
        with h5py.File(self.fn, mode, *args, **kargs) as fptr:
            if group_mode == "r":
                pass
            elif group_mode == "a":
                if self.gn not in fptr:
                    fptr.create_group(self.gn, track_order=track_order)
            else:
                raise ValueError(group_mode)
            g = fptr[self.gn]
            if not isinstance(g, (h5py.Dataset, h5py.Group)):
                raise Exception(f"{self.gn} is not a group.")
            yield fptr, g

    def __str__(self) -> str:
        return f"{self.fn}::{self.gn}"

    def __repr__(self) -> str:
        return f"H5Path({self.fn}, {self.gn})"

    @classmethod  # type: ignore
    def __get_validators__(cls):
        yield lambda v, _: h5path(v)


PATH_TYPE = Union[str, H5Path, os.PathLike[str]]


def read_array(
    fname: PATH_TYPE,
    /,
    *,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[Any]:
    """
    Read an array from file. The file name (`fname`) could be a `str`, a `Path`.
    """
    f = make_path(fname)
    if not f.exists():
        raise FileNotFoundError(f)
    if isinstance(f, H5Path):
        with f.open_group("r", "r") as (_, dataptr):
            return cast(npt.NDArray[Any], dataptr[...])
    elif f.suffix == ".npy":
        ans = np.load(f)
    elif f.suffix in [".bin", ".emc"]:
        ans = np.fromfile(f, dtype)
    else:
        raise Exception(f"Cannot identify file({fname}) suffix.")
    return cast(npt.NDArray[Any], ans)


def write_array(
    fname: PATH_TYPE,
    arr: npt.NDArray[Any],
    /,
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Union[None, str, int] = None,
) -> None:
    f = make_path(fname)
    if f.exists() and not overwrite:
        raise FileExistsError(f"{f} exists.")
    if isinstance(f, H5Path):
        with f.open("a") as fp:
            if f.gn in fp:
                del fp[f.gn]
            fp.create_dataset(
                f.gn,
                data=arr,
                compression=compression,
                compression_opts=compression_opts,
            )
        return
    else:
        if compression is not None:
            _log.warning(
                "The compression parameter is ignored. This is used only for HDF5 files."
            )  # pragma: no cover
        if compression_opts is not None:
            _log.warning(
                "The compression_opts parameter is ignored. This is used only for HDF5 files."
            )  # pragma: no cover
        if f.suffix == ".npy":
            np.save(f, arr)
            return
        if f.suffix in [".bin", ".emc"]:
            arr.ravel().tofile(str(f))
            return
    raise ValueError(f"Cannot identify file({fname}) suffix.")


def check_h5path(s: PATH_TYPE) -> bool:
    if isinstance(s, H5Path):
        return True
    fn_gn = str(s).split("::")  # file name, group name
    lf = len(fn_gn)
    if lf == 0 or lf > 2:
        return False
    if lf == 2:
        return True
    return (Path(fn_gn[0]).suffix.lower() == ".h5") or h5py.is_hdf5(fn_gn[0])


def h5path(src: PATH_TYPE, group: Optional[str] = None) -> H5Path:
    """
    Convert string / path to `H5Path`.
    Example: example.h5::h5_group

    Parameters
    ----------
    src : Union[Path, str]
        The input.

    group : Optional[str]

    Returns
    -------
    H5Path:
        The return
    """
    if group is not None:
        if isinstance(src, H5Path):
            raise TypeError()
        return H5Path(str(src), group)
    if isinstance(src, (Path, str)):
        return H5Path(*parse_h5path(src))
    if isinstance(src, H5Path):
        return src
    raise TypeError()


def make_path(s: PATH_TYPE) -> Union[Path, H5Path]:
    """
    The make_path function creates a path object based on the input value. It accepts a string or
    an existing Path object and returns either a `Path` object or an `H5Path` object, depending on the input.
    """
    if check_h5path(s):
        return h5path(s)
    return Path(cast(str, s))


def check_remove_groups(
    fp: Union[h5py.Group, h5py.File], groups: Iterable[str], overwrite: bool
) -> None:
    attrs = fp.attrs.keys()
    for g in groups:
        if g in fp:
            if overwrite:
                del fp[g]
            else:
                raise ValueError(f"{g} exists")
        elif g in attrs and not overwrite:
            raise ValueError(f"{g} exists in attrs")


def _check_exists(
    group_name: str, fp: Union[h5py.Group, h5py.File], overwrite: bool, verbose: bool
) -> bool:
    if group_name not in fp:
        return False
    if verbose:
        _log.info(f"{group_name} already exists.")
    if not overwrite:
        raise Exception(f"{group_name} already exists.")
    return True


_T = Union[Mapping[str, "_T"], npt.NDArray[Any], str, int, float, bool]


def _write_single(
    group: Union[h5py.File, h5py.Group, h5py.Dataset],
    k: str,
    v: _T,
    overwrite: bool,
    verbose: bool,
    compression: Optional[str],
    compression_opts: Union[None, str, int],
) -> None:
    if isinstance(v, np.ndarray):
        if not isinstance(group, (h5py.File, h5py.Group)):
            raise Exception(f"Cannot write type {type(v)} to {type(group)}")
        if _check_exists(k, group, overwrite, verbose):
            del group[k]
        group.create_dataset(
            k, data=v, compression=compression, compression_opts=compression_opts
        )
    elif isinstance(v, dict):
        if not isinstance(group, (h5py.File, h5py.Group)):
            raise Exception(f"Cannot write type {type(v)} to {type(group)}")
        _write_group(group, k, v, overwrite, verbose, compression, compression_opts)
    else:
        group.attrs[k] = v


def _write_group(
    fp: Union[h5py.File, h5py.Group],
    group_name: str,
    obj: _T,
    overwrite: bool,
    verbose: bool,
    compression: Optional[str],
    compression_opts: Union[None, str, int],
) -> None:
    if not isinstance(obj, dict):
        raise Exception(f"Cannot write type {type(obj)}")

    obj_dot = obj.pop(".", None)
    if obj_dot is not None:
        if _check_exists(group_name, fp, overwrite, verbose):
            del fp[group_name]
        fp.create_dataset(group_name, data=obj_dot)
    elif not _check_exists(group_name, fp, overwrite, verbose):
        fp.create_group(group_name)

    g = fp[group_name]
    if isinstance(g, h5py.Datatype):
        raise NotImplementedError("The support for h5py.Datatype is not implemented.")
    for k, v in obj.items():
        _write_single(g, k, v, overwrite, verbose, compression, compression_opts)
    if obj_dot is not None:
        obj["."] = obj_dot


def write_obj_h5(
    fn: Union[str, H5Path],
    obj: _T,
    overwrite: bool = False,
    verbose: bool = False,
    compression: Optional[str] = None,
    compression_opts: Union[None, str, int] = None,
) -> None:
    """Save dict `obj` to a `h5py.File`. The `np.ndarray` values are saved as
    h5 datasets. Others are saved in attributes of h5 group `group_name`.
    Parameters
    ----------
    fn : h5py.File
        The handle of an h5file.
    group_name : str
    obj : dict
    """
    f = h5path(fn)
    with f.open("a") as fp:
        _write_group(fp, f.gn, obj, overwrite, verbose, compression, compression_opts)


def _read_group(g: Union[h5py.File, h5py.Group]) -> dict[str, _T]:
    ans = dict()
    for k, v in g.attrs.items():
        ans[k] = v
    if isinstance(g, (h5py.Group, h5py.File)):
        for k in g.keys():
            ans[k] = _read_group(g[k]) if isinstance(g[k], h5py.Group) else g[k][...]
    else:
        ans["."] = g[...]
    return ans


def read_obj_h5(fn: Union[str, H5Path]) -> dict[str, Any]:
    """
    The inverse operation of `save_obj`. Read a dictionary from a h5 group.
    Parameters
    ----------
    fn : h5py.File
        The input h5 file handler.
    group_name : str
        The group name stores the object.
    """
    f = h5path(fn)
    with f.open_group() as (_, gp):
        return _read_group(gp)
