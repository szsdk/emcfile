from __future__ import annotations

import logging
from enum import IntEnum
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import numpy.typing as npt
from beartype import beartype
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


class PixelType(IntEnum):
    GOOD = 0
    CORNER = 1
    BAD = 2


class Detector:
    def __init__(
        self,
        coor: npt.NDArray[np.float64],
        factor: npt.NDArray[np.float64],
        mask: npt.NDArray[np.int16],
        detd: float,
        ewald_rad: float,
        *,
        writeable: bool = False,
    ) -> None:
        self.num_pix = len(factor)
        self.coor: npt.NDArray[np.float64] = coor.copy()
        self.mask: npt.NDArray[np.int16] = mask.copy()
        self.factor: npt.NDArray[np.float64] = factor.copy()
        self.detd = detd
        self.ewald_rad = ewald_rad
        self.setflags(writeable=writeable)
        self._norm_flag: Optional[bool] = None

    @property
    def pixel_size(self) -> float:
        return self.detd / self.ewald_rad

    @property
    def coor_factor(self) -> npt.NDArray[np.float64]:
        return cast(
            npt.NDArray[np.float64],
            np.concatenate([self.coor, self.factor[:, None]], axis=1),
        )

    @property
    def den_size(self) -> int:
        return 2 * int(np.max(np.linalg.norm(self.coor, axis=1))) + 3

    def norm(self) -> None:
        self.factor.setflags(write=True)
        self.factor /= self.factor.mean()
        self.factor.setflags(write=self.flags["writeable"])
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
        if np.all(self.coor[:, -1] == 0):
            return 2
        return 3

    @property
    def norm_flag(self) -> bool:
        if self._norm_flag is not None:
            return self._norm_flag
        nf = bool(np.isclose(self.factor.mean(), 1.0))
        if not self.flags["writeable"]:
            self._norm_flag = nf
        return nf

    @property
    def flags(self) -> Dict[str, Any]:
        return {"writeable": self._writeable}

    def setflags(self, writeable: Optional[bool] = None) -> None:
        """
        Set the writeablility

        Parameters
        ----------
        writeable : Optional[bool]

        """
        if writeable is None:
            return
        if not isinstance(writeable, bool):
            raise ValueError("Wrong type of writeable")
        for arr in cast(list[npt.NDArray[Any]], [self.coor, self.mask, self.factor]):
            arr.setflags(write=writeable)
        self._writeable: bool = writeable
        if writeable:
            self._norm_flag = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the detecotr to a dictionary

        Returns
        -------
        Dict[str, Any]

        """
        return {
            "coor": self.coor.copy(),
            "factor": self.factor.copy(),
            "mask": self.mask.copy(),
            "detd": self.detd,
            "ewald_rad": self.ewald_rad,
        }

    def copy(self) -> Detector:
        return Detector(
            coor=self.coor,
            factor=self.factor,
            mask=self.mask,
            detd=self.detd,
            ewald_rad=self.ewald_rad,
        )

    def deepcopy(self) -> Detector:
        return Detector(**self.to_dict())

    def write(self, fname: Union[str, Path, H5Path], overwrite: bool = False) -> None:
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
        index: Union[
            slice, list[PixelType], npt.NDArray[np.bool_], npt.NDArray[np.integer[Any]]
        ],
    ) -> Detector:
        if isinstance(index, list):
            if len(index) == 0:
                raise ValueError("0-length input")
            if isinstance(index[0], PixelType):
                idx = reduce(np.logical_or, [self.mask == pt for pt in index])
        else:
            idx = index
        return Detector(
            self.coor[idx],
            self.factor[idx],
            self.mask[idx],
            self.detd,
            self.ewald_rad,
        )


def _from_asciidet(fname: Path) -> Detector:
    with fname.open() as fp:
        num_pix, detd, ewald_rad = [
            f(d) for f, d in zip([int, float, float], fp.readline().strip().split(" "))
        ]
    det = np.genfromtxt(
        fname,
        skip_header=1,
        dtype=[
            ("qx", "f8"),
            ("qy", "f8"),
            ("qz", "f8"),
            ("factor", "f4"),
            ("mask", "i4"),
        ],
    )
    return Detector(
        np.array([det["qx"], det["qy"], det["qz"]]).T,
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
            np.array([fp["qx"][:], fp["qy"][:], fp["qz"][:]]).T.copy(),
            fp["corr"][:],
            fp["mask"][:],
            fp["detd"][...].item(),
            fp["ewald_rad"][...].item(),
        )


def _from_file(fname: PATH_TYPE) -> Detector:
    f = make_path(fname)
    if not f.exists():
        raise FileNotFoundError(f"Detector file, {fname}, does not exist.")
    if isinstance(f, H5Path):
        return _from_h5det(f)
    if isinstance(f, Path):
        if f.suffix == ".dat":
            return _from_asciidet(f)
        else:
            raise NotImplementedError()
    raise ValueError(f"do not know how to handle {f} detector file")


def _init_detector(
    coor: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.integer[Any]],
    factor: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.float64]]:
    if not (coor.shape[0] == mask.shape[0] == factor.shape[0]):
        raise ValueError("`coor`, `mask`, `factor` should have the same length.")
    if coor.shape[1] == 2:
        new_coor = np.zeros((coor.shape[0], 3), coor.dtype)
        new_coor[:, :2] = coor
    else:
        new_coor = coor
    return new_coor.astype(np.float64), mask.astype(np.int16), factor.astype(np.float64)


@beartype
def detector(
    src: Union[Detector, PATH_TYPE, None] = None,
    *,
    coor: Optional[npt.NDArray] = None,
    mask: Optional[npt.NDArray] = None,
    factor: Optional[npt.NDArray] = None,
    detd: Union[float, int, np.floating, np.integer, None] = None,
    ewald_rad: Union[float, int, np.floating, np.integer, None] = None,
    writeable: bool = False,
    norm_flag: bool = True,
) -> Detector:
    """
    This is the interface function for Detector.

    Parameters
    ----------
    src : Union[Detector, str, Path, None]
        If src is a Detector, this function return another copy of the input.
        If src is a file(a str or a Path), read the detector form it.
        The file format could be '.h5', '.dat', '.ini', '.cfg'.
        If src is None, all parameters for a Detector should be given.
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
            coor, mask, factor = _init_detector(coor, mask, factor)
            det = Detector(
                coor, factor, mask, float(detd), float(ewald_rad), writeable=writeable
            )
    elif isinstance(src, Detector):
        det = Detector(
            src.coor if coor is None else coor.astype(np.float64),
            src.factor if factor is None else factor.astype(np.float64),
            src.mask if mask is None else mask.astype(np.int16),
            src.detd if detd is None else float(detd),
            src.ewald_rad if ewald_rad is None else float(ewald_rad),
            writeable=writeable,
        )
    elif isinstance(src, (str, Path, H5Path)):
        det = _from_file(src)
        det.setflags(writeable=writeable)

    if det is None:
        raise Exception(f"Can not parse {src}({type(src)}")
    if norm_flag:
        det.norm()
    return det


@beartype
def get_2ddet(det: Detector, /, *, inplace: bool = False) -> Detector:
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
        ans = det.deepcopy()
    ans.setflags(writeable=True)
    ans.coor[:, :2] = xyz_to_cxy(
        ans.coor, det.ewald_rad, det.detd, -int(np.sign(det.coor[:, 2].sum()))
    )

    ans.coor[:, 2] = 0.0
    ans.setflags(writeable=False)
    return ans


@beartype
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
        The coordinate of the center.
    """

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
    return np.array(res.x)


def xyz_to_cxy(
    xyz: npt.NDArray[Any], ewald_rad: float, detd: float, direction: int
) -> npt.NDArray[np.float64]:
    is_1d = xyz.ndim == 1
    if is_1d:
        xyz = xyz[None, :]
    ans = xyz[:, :2] * detd / (ewald_rad + direction * xyz[:, 2][:, None])
    return ans.ravel() if is_1d else ans


def cxy_to_xyz(
    cxy: npt.NDArray[Any], ewald_rad: float, detd: float, direction: int
) -> npt.NDArray[np.float64]:
    is_1d = cxy.ndim == 1
    if is_1d:
        cxy = cxy[None, :]
    xyz = np.empty((cxy.shape[0], 3))
    cxy = cxy / detd
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
        The detecotr pixel position
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
        self.xy = np.round(self.cxy - self.cxy.min(axis=0)).astype("i4")
        self.frame_shape = self.xy.max(axis=0) + 1
        self._mask = np.zeros(self.frame_shape, dtype="u1")
        pix_idx = self._det.mask == 2
        self._mask[self.xy[pix_idx, 0], self.xy[pix_idx, 1]] = 1
        self._mask = self._mask.T.copy()
        self._count: ma.MaskedArray = ma.masked_array(
            np.zeros((self.frame_shape[1], self.frame_shape[0]), dtype="f8"),
            mask=self._mask,
        )
        np.add.at(
            self._count, (self.xy[:, 1], self.xy[:, 0]), np.ones(self.xy.shape[0])
        )
        self._count /= self._count.mean()

    def frame_pixels(self) -> list[npt.NDArray[np.float64]]:
        et = self.frame_extent()
        return np.meshgrid(
            np.linspace(et[0], et[1], self.frame_shape[1]),
            np.linspace(et[2], et[3], self.frame_shape[0]),
        )

    def render(self, raw_img: npt.NDArray[Any]) -> ma.MaskedArray[Any, Any]:
        """
        The right way to visualize the `raw_img` is
        `plt.imshow(img, origin='lower', extent=self.frame_extent())`.
        """
        img: ma.MaskedArray[Any, Any] = ma.masked_array(
            np.zeros((self.frame_shape[1], self.frame_shape[0]), dtype="f8"),
            mask=self._mask,
        )
        np.add.at(img, (self.xy[:, 1], self.xy[:, 0]), raw_img)
        return img / self._count

    def to_cxy(self, xyz: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        return xyz_to_cxy(xyz, self._det.ewald_rad, self._det.detd, self.direction)

    def to_xyz(self, cxy: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        return cxy_to_xyz(cxy, self._det.ewald_rad, self._det.detd, self.direction)

    def frame_extent(self) -> tuple[float, float, float, float]:
        xmin, ymin = self.cxy.min(axis=0)
        dx, dy = self.frame_shape
        return xmin, xmin + dx, ymin, ymin + dy


@beartype
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


@beartype
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
        np.array(grid_position(shape)).reshape(2, -1).T, ewald_rad, detd, direction
    )

    det_r = np.linalg.norm(det.coor[:, :2], axis=1)
    ds = binned_statistic(
        det_r, det.factor, statistic="mean", bins=int(det_r.max() - det_r.min()) + 1
    )
    y = ds.statistic
    x = (ds.bin_edges[:-1] + ds.bin_edges[1:]) / 2
    f = interp1d(
        x, y, assume_sorted=False, fill_value=(y.max(), y.min()), bounds_error=False
    )
    r = np.linalg.norm(coor[:, :2], axis=1)
    factor = f(r)
    mask = np.full_like(factor, PixelType.GOOD, int)
    if apply_mask:
        mask[r < det_r.min()] = PixelType.BAD
        mask[r > det_r.max()] = PixelType.BAD
    return Detector(coor, factor, mask, detd, ewald_rad)
