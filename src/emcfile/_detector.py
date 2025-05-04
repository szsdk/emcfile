from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from enum import IntEnum
from functools import reduce
from pathlib import Path
from typing import Any, Literal, Optional, Type, TypeVar, Union, cast

import h5py
import numpy as np
import numpy.typing as npt
from numpy import ma
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import binned_statistic

from ._h5helper import PATH_TYPE, H5Path, make_path

__all__ = [
    "det_render",
    "Detector",
    "DetRender",
    "detector",
    "get_2ddet",
    "get_3ddet_from_shape",
    "det_isclose",
    "get_ewald_vec",
    "PixelType",
]

_log = logging.getLogger(__name__)

T1 = TypeVar("T1", bound=npt.NBitBase)
T2 = TypeVar("T2", bound=npt.NBitBase)


class PixelType(IntEnum):
    GOOD = 0  # This is not recommended to be used in a bitwise mask.
    CORNER = 0b00000001
    BAD = 0b00000010


_BITMAP = Union[Sequence[int], int]


def _bitmap_to_int(bm: _BITMAP) -> np.uint8:
    if isinstance(bm, int):
        return np.uint8(bm)
    ans = 0
    for i in bm:
        ans |= 1 << i
    return np.uint8(ans)


class Detector:
    dtype = np.dtype(
        [
            ("coor", "f8", 3),
            ("factor", "f4"),
            ("mask", "i4"),
        ]
    )

    def __init__(
        self,
        coor: npt.NDArray[np.float64],
        factor: npt.NDArray[np.float64],
        mask: npt.NDArray[np.int32],
        detd: float,
        ewald_rad: float,
    ) -> None:
        self.num_pix = len(factor)
        self.coor: npt.NDArray[np.float64] = coor.copy()
        self.mask: npt.NDArray[np.int32] = mask.copy()
        self.factor: npt.NDArray[np.float64] = factor.copy()
        self.detd = detd
        self.ewald_rad = ewald_rad
        self._norm_flag: Optional[bool] = None

    @property
    def pixel_size(self) -> float:
        return self.detd / self.ewald_rad

    @property
    def coor_factor(self) -> npt.NDArray[np.float64]:
        return np.concatenate([self.coor, self.factor[:, None]], axis=1)

    @property
    def den_size(self) -> int:
        return 2 * int(np.max(np.linalg.norm(self.coor, axis=1))) + 3

    def norm(self) -> None:
        self.factor /= self.factor.mean()
        self._norm_flag = None

    def __repr__(self) -> str:
        r = np.linalg.norm(self.coor, axis=1)
        ans = f"""Detector <{hex(id(self))}>
  Dimension: {self.ndim}
  Number of pixels: {len(self.coor)}
  Detector distance: {self.detd:.3f} mm
  Ewald Radius: {self.ewald_rad:.3f} pixel
  q_max : {r.max():.3f} pixel
  q_min : {r.min():.3f} pixel
  Normalized: {self.norm_flag}
"""
        for i, pt in enumerate(PixelType):
            n = np.sum(self.mask == pt)
            ans = ans + (f"  Mask: {pt} - {n}\n" if i == 0 else f"        {pt} - {n}\n")
        return ans

    @property
    def ndim(self) -> int:
        return 2 if np.isinf(self.ewald_rad) else 3
        # if np.all(self.coor[:, -1] == 0):
        #     return 2
        # return 3

    @property
    def norm_flag(self) -> bool:
        if self._norm_flag is not None:
            return self._norm_flag
        return bool(np.isclose(self.factor.mean(), 1.0))

    def write(self, fname: PATH_TYPE, overwrite: bool = False) -> None:
        fn = make_path(fname)
        if fn.exists() and not overwrite:
            raise FileExistsError(f"{fn} exists.")
        if isinstance(fn, H5Path):
            return _write_h5det(self, fn)
        elif isinstance(fn, Path) and fn.suffix in [".dat"]:
            return _write_asciidet(self, fn)
        else:
            raise ValueError(f"Cannot parse {fname}")

    def __getitem__(
        self,
        index: slice
        | Sequence[PixelType]
        | npt.NDArray[np.bool_]
        | npt.NDArray[np.integer[Any]],
    ) -> Detector:
        idx = index
        if isinstance(index, Sequence):
            if len(index) == 0:
                raise ValueError("0-length input")
            if isinstance(index[0], PixelType):
                idx = reduce(np.logical_or, [self.mask == pt for pt in index])
        return Detector(
            self.coor[idx],
            self.factor[idx],
            self.mask[idx],
            self.detd,
            self.ewald_rad,
        )

    def __array__(self) -> npt.NDArray[Any]:
        ans = np.empty(self.num_pix, dtype=Detector.dtype)
        ans["coor"] = self.coor
        ans["factor"] = self.factor
        ans["mask"] = self.mask
        return ans

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[Type[object]],
        args: Iterable[object],
        kwargs: Mapping[str, object],
    ) -> object:
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override

        # __array_function__ to handle Detector objects.

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def check_ewald_rad(self, rtol: float = 1e-04, atol: float = 1e-06) -> bool:
        coor = self.coor
        if coor.shape[0] >= 128:
            coor = coor[:: coor.shape[0] // 67]
        vec = get_ewald_vec(coor)
        if vec[3] == 0 and np.isinf(self.ewald_rad):
            return True
        r = np.linalg.norm(vec[:3]) / vec[3]
        if not np.isclose(self.ewald_rad, r, rtol=rtol, atol=atol):
            logging.warning("det Ewald radius: %f, calculated: %f", self.ewald_rad, r)
            return False
        return True

    def mask_set(
        self,
        flags: _BITMAP,
        values: _BITMAP,
        pixel_indices: None
        | slice
        | npt.NDArray[np.bool_]
        | npt.NDArray[np.integer[Any]] = None,
    ) -> None:
        f = _bitmap_to_int(flags)
        v = _bitmap_to_int(values)
        self.mask[pixel_indices] = (self.mask[pixel_indices] & ~f) | (v & f)

    def mask_flip(
        self,
        flags: _BITMAP,
        pixel_indices: None
        | slice
        | npt.NDArray[np.bool_]
        | npt.NDArray[np.integer[Any]] = None,
    ) -> None:
        f = _bitmap_to_int(flags)
        self.mask[pixel_indices] ^= f

    def mask_select(self, flags: _BITMAP, values: _BITMAP) -> npt.NDArray[np.bool_]:
        f = _bitmap_to_int(flags)
        v = _bitmap_to_int(values)
        return cast(npt.NDArray[np.bool_], (self.mask & f) == v)


HANDLED_FUNCTIONS = {}

FT = TypeVar("FT", bound=Callable[..., Any])


def implements(np_function: Callable[..., Any]) -> Callable[[FT], FT]:
    "Register an __array_function__ implementation for PatternsSOne objects."

    def decorator(func: FT) -> FT:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate_Detector(dets: Sequence[Detector]) -> Detector:
    ewald_rad = dets[0].ewald_rad
    np.testing.assert_array_equal([d.ewald_rad for d in dets], dets[0].ewald_rad)
    detd = dets[0].detd
    np.testing.assert_array_equal([d.detd for d in dets], dets[0].detd)
    data = np.concatenate([np.asarray(d) for d in dets], axis=0)
    return Detector(data["coor"], data["factor"], data["mask"], detd, ewald_rad)


def _from_asciidet(fname: Path) -> Detector:
    with fname.open() as fp:
        num_pix, detd, ewald_rad = [
            f(d) for f, d in zip([int, float, float], fp.readline().strip().split(" "))
        ]
    det = np.genfromtxt(
        fname,
        skip_header=1,
        dtype=Detector.dtype,
    )
    if num_pix != det.shape[0]:
        raise ValueError(f"Number of pixels in {fname} does not match header.")
    return Detector(
        det["coor"],
        det["factor"],
        det["mask"],
        detd,
        ewald_rad,
    )


def _write_asciidet(det: Detector, fname: Path) -> None:
    _log.info(f"Writing ASCII detector to {fname}")
    qx, qy, qz = [det.coor[:, i].ravel() for i in range(3)]
    corr = det.factor.ravel()
    mask = det.mask.ravel().astype("u1")

    with fname.open("w") as fptr:
        fptr.write("%d %.6f %.6f\n" % (qx.size, det.detd, det.ewald_rad))
        for pars in zip(qx, qy, qz, corr, mask):
            fptr.write("%21.15e %21.15e %21.15e %21.15e %d\n" % pars)


def _write_h5det(det: Detector, fname: H5Path) -> None:
    _log.info(f"Writing HDF5 detector to {fname}")
    with fname.open_group("a", "a") as (_, fptr):
        assert isinstance(fptr, (h5py.Group, h5py.File))
        for s in ["qx", "qy", "qz", "corr", "mask", "detd", "ewald_rad"]:
            if s in fptr:
                del fptr[s]
        fptr["qx"] = det.coor[:, 0].ravel().astype("f8")
        fptr["qy"] = det.coor[:, 1].ravel().astype("f8")
        fptr["qz"] = det.coor[:, 2].ravel().astype("f8")
        fptr["corr"] = det.factor.ravel().astype("f8")
        fptr["mask"] = det.mask.ravel().astype("u1")
        fptr["detd"] = float(det.detd)
        fptr["ewald_rad"] = float(det.ewald_rad)


def _from_h5det(fname: H5Path) -> Detector:
    with fname.open_group("r", "r") as (_, fp):
        return Detector(
            np.array(
                [
                    cast(h5py.Dataset, fp["qx"])[:],
                    cast(h5py.Dataset, fp["qy"])[:],
                    cast(h5py.Dataset, fp["qz"])[:],
                ]
            )
            .reshape(3, -1)
            .T.copy(),
            cast(h5py.Dataset, fp["corr"])[:].ravel(),
            cast(h5py.Dataset, fp["mask"])[:].ravel(),
            cast(h5py.Dataset, fp["detd"])[...].item(),
            cast(h5py.Dataset, fp["ewald_rad"])[...].item(),
        )


def _from_file(fname: PATH_TYPE) -> Detector:
    f = make_path(fname)
    if not f.exists():
        raise FileNotFoundError(f"Detector file, {fname}, does not exist.")
    if isinstance(f, H5Path):
        return _from_h5det(f)
    if isinstance(f, Path) and (f.suffix == ".dat"):
        return _from_asciidet(f)
    raise ValueError(f"do not know how to handle {f} detector file")


def _init_detector(
    coor: npt.NDArray["np.floating[T1]"],
    mask: npt.NDArray["np.integer[T2]"],
    factor: npt.NDArray["np.floating[T1]"],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    if not (coor.shape[0] == mask.shape[0] == factor.shape[0]):
        raise ValueError("`coor`, `mask`, `factor` should have the same length.")
    if coor.shape[1] == 2:
        new_coor = np.zeros((coor.shape[0], 3), coor.dtype)
        new_coor[:, :2] = coor
    else:
        new_coor = coor
    return new_coor.astype(np.float64), mask.astype(np.int32), factor.astype(np.float64)


def detector(
    src: "Detector | PATH_TYPE | None" = None,
    *,
    coor: Optional[npt.NDArray["np.floating[T1]"]] = None,
    mask: Optional[npt.NDArray["np.integer[T2]"]] = None,
    factor: Optional[npt.NDArray["np.floating[T1]"]] = None,
    detd: "float | int | None" = None,
    ewald_rad: "float | int | None" = None,
    norm_flag: bool = True,
    check_consistency: bool = True,
) -> Detector:
    """
    This is the interface function for `Detector`. It create a Detector object from
    a source or from provided coordinates, mask, factor, and other parameters.

    Parameters
    ----------
    src : Union[Detector, PATH_TYPE, None]
        Source from which to create the Detector. Can be None, Detector object, or file
        path. If src is a Detector, this function return another copy of the input, the
        data of the output `Detector` could be rewritten by over given arguments, such
        as `coor`. If src is a file (a str or a Path), read the detector form it. The
        file format could be '.h5', '.dat', '.emc'. If src is None, all parameters for a
        Detector should be given. Default is None.

    coor : Optional[npt.NDArray]
        Coordinates of the detector.

    mask : Optional[npt.NDArray]
        Mask of the detector.

    factor : Optional[npt.NDArray]
        Factor of the detector.

    detd : Union[float, int, np.floating, np.integer, None]
        Detector distance.

    ewald_rad : Union[float, int, np.floating, np.integer, None]
        Ewald radius.

    norm_flag : bool
        Flag to indicate whether to normalize the detector. Default is True.

    Returns
    -------
    Detector: Created detector object.

    Raises
    -------
    Exception: If the source cannot be parsed.
    """

    det = None
    if src is None:
        if (
            (coor is not None)
            and (mask is not None)
            and (factor is not None)
            and (detd is not None)
            and (ewald_rad is not None)
        ):
            coor_, mask_, factor_ = _init_detector(coor, mask, factor)
            det = Detector(coor_, factor_, mask_, float(detd), float(ewald_rad))
    elif isinstance(src, Detector):
        det = Detector(
            src.coor if coor is None else coor.astype(np.float64),
            src.factor if factor is None else factor.astype(np.float64),
            src.mask if mask is None else mask.astype(np.int32),
            src.detd if detd is None else float(detd),
            src.ewald_rad if ewald_rad is None else float(ewald_rad),
        )
    elif isinstance(src, (str, Path, H5Path)):
        det = _from_file(src)

    if det is None:
        raise Exception(f"Can not parse {src}({type(src)}")
    if norm_flag:
        det.norm()
    if check_consistency and (not det.check_ewald_rad()):
        raise ValueError(
            "Ewald radius is not consistent with given detector coordinates."
        )
    return det


def get_2ddet(
    det: Detector,
    /,
    *,
    inplace: bool = False,
    pixel_space: Literal["real", "reciprocal"] = "reciprocal",
) -> Detector:
    """
    Get a detector for 2d clustering

    Parameters
    ----------
    det : Detector

    inplace : bool


    Returns
    -------
    Detector

    """
    if inplace:
        ans = det
    else:
        ans = deepcopy(det)
    if det.detd != det.ewald_rad:
        logging.warning(
            f"Detector distance is not equal to Ewald radius. {pixel_space} space is used"
        )
    ans.coor[:, :2] = xyz_to_cxy(
        ans.coor, det.ewald_rad, -int(np.sign(det.coor[:, 2].sum()))
    )
    if pixel_space == "real":
        ans.coor[:] *= det.detd / det.ewald_rad

    ans.coor[:, 2] = 0.0
    ans.ewald_rad = np.inf
    return ans


def det_isclose(
    det1: Detector, det2: Detector, /, *, rtol: float = 1e-6
) -> bool:  # pragma: no cover
    """
    Check whether two detectors are close to each other.

    The order of comparison is sorted by their computational difficulties.

    Parameters
    ----------
    det1 : Detector
        The first detector
    det2 : Detector
        The second detector
    rtol : float
        The relative tolerance controls all closeness comparison.

    Returns
    -------
    bool:
        Result
    """
    if not np.allclose(det1.detd, det2.detd, rtol=rtol):
        return False
    if not np.allclose(det1.ewald_rad, det2.ewald_rad, rtol=rtol):
        return False
    if not np.all(det1.mask == det2.mask):
        return False
    if not np.allclose(det1.coor, det2.coor, rtol=rtol):
        return False
    if not np.allclose(det1.factor, det2.factor, rtol=rtol):
        return False
    return True


def get_ewald_vec(coor: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """
    Calcualte the center of a sphere with a point cloud distributed on it.

    Parameters
    ----------
    coor : np.ndarray
        The point cloud whose shape is (number of points, 3)

    Returns
    -------
    np.ndarray:
        The projection coordinate of the center.
    """
    # if coor.shape[0] < 4:
    #     raise ValueError("At least 4 points are required.")

    # 2d detector test
    for ca, cb, cc in itertools.combinations(range(coor.shape[0]), 3):
        db = coor[cb] - coor[ca]
        db /= np.linalg.norm(db)
        dc = coor[cc] - coor[ca]
        dc /= np.linalg.norm(dc)
        n0 = np.cross(db, dc)
        if np.linalg.norm(n0) > 1e-1:
            break
    else:
        raise ValueError("The input coor is degenerated to a line.")
    # if np.linalg.norm(n0) < 1e-1:
    #     raise ValueError("The input coor is degenerated to a line.")

    coor_shift = coor - coor[ca]
    with np.errstate(divide="ignore", invalid="ignore"):
        coor_shift /= np.linalg.norm(coor_shift, axis=1, keepdims=True)
    coor_shift[ca] = 0
    is_2d = np.all(np.abs(np.dot(coor_shift, n0.reshape(3, 1))) < 1e-3)
    if is_2d:
        return np.array([n0[0], n0[1], n0[2], 0.0])

    def _f(r: npt.NDArray[Any], x: npt.NDArray[Any]) -> float:
        a = np.linalg.norm(x - r, axis=1)
        return float(a.std())

    res = minimize(
        _f,
        np.array([-0.0, 0.0, -1.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "disp": False},
        args=(coor if len(coor) < 32 else coor[:: len(coor) // 32],),
    )
    res = minimize(
        _f,
        res.x,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "disp": False},
        args=(coor,),
    )
    return np.concatenate([res.x, [1.0]])


def xyz_to_cxy(
    xyz: npt.NDArray[Any], ewald_rad: float, direction: int
) -> npt.NDArray[np.float64]:
    is_1d = xyz.ndim == 1
    if is_1d:
        xyz = xyz[None, :]
    ans = xyz[:, :2] * ewald_rad / (ewald_rad + direction * xyz[:, 2][:, None])
    return ans.ravel() if is_1d else ans


def cxy_to_xyz(
    cxy: npt.NDArray[Any], ewald_rad: float, direction: int
) -> npt.NDArray[np.float64]:
    is_1d = cxy.ndim == 1
    if is_1d:
        cxy = cxy[None, :]
    cxy = cxy / ewald_rad
    xyz = np.empty((cxy.shape[0], 3))
    cr = 1 / np.sqrt(1 + np.linalg.norm(cxy, axis=1) ** 2)
    xyz[:, :2] = ewald_rad * cxy * cr.reshape(-1, 1)
    xyz[:, 2] = direction * ewald_rad * (cr - 1)
    return xyz.ravel() if is_1d else xyz


class DetRender:
    """

    Attributes
    ----------
    direction : int
    cxy : np.ndarray(num_pix, 2)
        The detector pixel position
    xy : np.ndarray(num_pix, 2)
        The image pixel position
    frame_shape : (int, int)
        The shape of image (frame)
    """

    def __init__(self, det: Detector):
        self._det = det
        self.direction = -int(np.sign(det.coor[:, 2].sum()))
        # TODO decide the direction with `get_ewald_vec`
        self.cxy = self.to_cxy(self._det.coor)
        self.xy = np.round(self.cxy - self.cxy.min(axis=0)).astype(np.int32)
        self.frame_shape = self.xy.max(axis=0) + 1
        self._mask = np.zeros(self.frame_shape, dtype="u1")
        pix_idx = self._det.mask == 2
        self._mask[self.xy[pix_idx, 0], self.xy[pix_idx, 1]] = 1
        self._mask = self._mask.T.copy()
        self._count: npt.NDArray[np.float64] = ma.masked_array(
            np.zeros((self.frame_shape[1], self.frame_shape[0]), dtype="f8"),
            mask=self._mask,
        )  # type: ignore
        np.add.at(
            self._count, (self.xy[:, 1], self.xy[:, 0]), np.ones(self.xy.shape[0])
        )
        self._count /= cast(
            np.float64,
            self._count.mean(),  # type: ignore
        )

    def frame_pixels(self) -> list[npt.NDArray[np.float64]]:
        et = self.frame_extent()
        return cast(
            list[npt.NDArray[np.float64]],
            np.meshgrid(
                np.linspace(et[0], et[1], self.frame_shape[1]),
                np.linspace(et[2], et[3], self.frame_shape[0]),
            ),
        )

    def render(self, raw_img: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """
        The right way to visualize the `raw_img` is
        `plt.imshow(img, origin='lower', extent=self.frame_extent())`.
        """
        img = ma.masked_array(
            np.zeros((self.frame_shape[1], self.frame_shape[0]), dtype="f8"),
            mask=self._mask,
        )  # type: ignore
        np.add.at(img, (self.xy[:, 1], self.xy[:, 0]), raw_img)
        return cast(npt.NDArray[np.float64], img / self._count)

    def to_cxy(self, xyz: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        cxy = xyz_to_cxy(xyz, self._det.ewald_rad, self.direction)
        cxy *= self._det.detd / self._det.ewald_rad
        return cxy

    def to_xyz(self, cxy: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        cxy = cxy * self._det.ewald_rad / self._det.detd
        return cxy_to_xyz(cxy, self._det.ewald_rad, self.direction)

    def frame_extent(
        self, pixel_unit: bool = True
    ) -> tuple[float, float, float, float]:
        xmin, ymin = self.cxy.min(axis=0)
        dx, dy = self.frame_shape
        if pixel_unit:
            return xmin, xmin + dx, ymin, ymin + dy
        else:
            r = self._det.detd / self._det.ewald_rad
            return xmin * r, (xmin + dx) * r, ymin * r, (ymin + dy) * r


def det_render(det: Detector) -> DetRender:
    """
    The factory function of `DetRender`

    Parameters
    ----------
    det : Detector


    Returns
    -------
    DetRender

    """
    return DetRender(det)


def grid_position(shape: tuple[int, ...]) -> list[npt.NDArray[np.float64]]:
    """
    Parameters
    ----------
    shape : tuple[int]

    Returns
    -------
    positions: list
    """
    return cast(
        list[npt.NDArray[np.float64]],
        np.meshgrid(
            *[np.linspace(-(s - 1) / 2, (s - 1) / 2, s) for s in shape], indexing="ij"
        ),
    )


def get_3ddet_from_shape(
    shape: tuple[int, int], det: Detector, apply_mask: bool = True
) -> Detector:
    """
    This fucntion resample the given detector `det` into a new detector  whose pixels
    are aligned in a grid with shape `shape`.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of new detector

    det : Detector
        The original detector

    apply_mask : bool
        If it is true, a pixel in the new detector is marked as bad if it out of range
        (q.min(), q.max()) where q is the radii of pixels in `det`.

    Returns
    -------
    Detector

    """
    detd = det.detd
    ewald_rad = det.ewald_rad
    direction = -int(np.sign(det.coor[:, 2].sum()))
    if direction == 0:
        direction = 1
    coor = cxy_to_xyz(
        np.array(grid_position(shape)).reshape(2, -1).T, ewald_rad, direction
    )

    det_r = np.linalg.norm(det.coor[:, :2], axis=1)
    ds = binned_statistic(
        det_r, det.factor, statistic="mean", bins=int(det_r.max() - det_r.min()) + 1
    )
    y = ds.statistic
    x = (ds.bin_edges[:-1] + ds.bin_edges[1:]) / 2
    f = interp1d(
        x,
        y,
        assume_sorted=False,
        fill_value=cast(float, (y.max(), y.min())),  # only make pyright happy
        bounds_error=False,
    )
    r = np.linalg.norm(coor[:, :2], axis=1)
    factor = f(r)
    mask = np.full_like(factor, PixelType.GOOD, int)
    if apply_mask:
        mask[r < det_r.min()] = PixelType.BAD
        mask[r > det_r.max()] = PixelType.BAD
    return Detector(coor, factor, mask, detd, ewald_rad)
