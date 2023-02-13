from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import emcfile as ef


@pytest.fixture()
def det():
    coor2d = np.mgrid[-32:33, -32:33].reshape(2, -1).T.astype(np.float64)
    num_pix = len(coor2d)
    coor = np.zeros((num_pix, 3))
    coor[:, :2] = coor2d
    ewald_rad = 128
    r2d = np.linalg.norm(coor2d, axis=1)
    coor[:, 2] = np.sqrt(ewald_rad**2 - r2d**2) - ewald_rad
    factor = np.random.uniform(3e-4, 4e-4, num_pix)
    mask = np.zeros(num_pix, np.int16)
    mask[r2d < 10] = ef.PixelType.BAD
    mask[r2d > 32] = ef.PixelType.CORNER
    det = ef.detector(
        coor=coor,
        mask=mask,
        factor=factor,
        detd=ewald_rad * 0.1,
        ewald_rad=ewald_rad,
        norm_flag=False,
    )
    return det


@pytest.fixture()
def det_file(tmp_path_factory, det):
    fn = tmp_path_factory.mktemp("data") / "det.dat"
    det.write(fn)
    return fn


def test_det_operation(det):
    np.array(det)


def test_det_read(det_file):
    ef.detector(det_file, norm_flag=False)
    with pytest.raises(Exception):
        ef.detector("data/det_sim.foo", norm_flag=False)
    with pytest.raises(Exception):
        ef.detector(12, norm_flag=False)


def test_det_write(det):
    det1 = det
    with TemporaryDirectory() as f:
        f = Path(f)
        for suffix in [".dat", ".h5"]:
            det1.write(f / f"det{suffix}")
            det1.write(f / f"det{suffix}", overwrite=True)
            with pytest.raises(FileExistsError):
                det1.write(f / f"det{suffix}")
            det2 = ef.detector(f / f"det{suffix}", norm_flag=False)
            ef.detector(det2)
        assert ef.det_isclose(det1, det2)


def test_get_ewald_vec(det):
    np.testing.assert_almost_equal(det.pixel_size, 0.1)
    x = ef.get_ewald_vec(det.coor)
    np.testing.assert_allclose(np.linalg.norm(x), det.ewald_rad, rtol=1e-4)


def test_repr(det_file):
    det = ef.detector(det_file, norm_flag=True)
    ans = f"""Detector <{hex(id(det))}>
  Dimension: 3
  Number of pixels: 4225
  Detector distance: 12.800 mm
  Ewald Radius: 128.000 pixel
  q_max : 46.004 pixel
  q_min : 0.000 pixel
  Normalized: True
  Mask: 0 - 2904
        1 - 1016
        2 - 305
"""
    assert repr(det) == ans


def test_getitem(det):
    det1 = det[[ef.PixelType.GOOD]]
    det2 = det[det.mask == ef.PixelType.GOOD]
    assert ef.det_isclose(det1, det2)


def test_get_2ddet(det):
    det_2d = ef.get_2ddet(det)
    ef.get_2ddet(det, inplace=True)
    assert ef.det_isclose(det, det_2d)
    assert det_2d.ndim == 2


def test_get_3ddet_from_shape(det):
    ef.get_3ddet_from_shape((100, 100), det)


def test_cxy_xyz_conversion(det):
    from emcfile._detector import cxy_to_xyz, xyz_to_cxy

    direction = 1 if det.coor[:, 2].sum() < 0 else -1
    cxy = xyz_to_cxy(det.coor, det.ewald_rad, direction)
    xyz = cxy_to_xyz(cxy, det.ewald_rad, direction)
    np.testing.assert_almost_equal(xyz, det.coor, decimal=4)


def test_det_render(det):
    detr = ef.det_render(det)
    np.testing.assert_almost_equal(
        detr.to_xyz(detr.to_cxy(det.coor)), det.coor, decimal=4
    )

    np.testing.assert_almost_equal(
        detr.to_xyz(detr.to_cxy(det.coor[0])), det.coor[0], decimal=4
    )
    detr.frame_extent()
    detr.frame_pixels()


def test_concatenate(det):
    det_sym = deepcopy(det)
    det_sym.coor *= np.array([-1, -1, 1])
    assert np.concatenate([det, det_sym]).num_pix == 2 * det.num_pix
