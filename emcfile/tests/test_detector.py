from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

try:
    import matplotlib.pylab as plt

    PLT_IMPORTED = True
except ModuleNotFoundError:
    PLT_IMPORTED = False

import emcfile as ef


def test_det_read():
    ef.detector("data/det_sim.dat", norm_flag=False)
    with pytest.raises(Exception):
        ef.detector("data/det_sim.foo", norm_flag=False)
    with pytest.raises(Exception):
        ef.detector(12, norm_flag=False)


def test_det_write():
    det1 = ef.detector("data/det_sim.dat", norm_flag=False)
    with TemporaryDirectory() as f:
        f = Path(f)
        for suffix in [".dat", ".h5"]:
            det1.write(f / f"det{suffix}")
            det1.write(f / f"det{suffix}", overwrite=True)
            with pytest.raises(FileExistsError):
                det1.write(f / f"det{suffix}")
            det2 = ef.detector(f / f"det{suffix}", norm_flag=False)
            ef.detector(**det2.to_dict())
        assert ef.det_isclose(det1.copy(), det2)


def test_get_ewald_vec():
    det = ef.detector("data/det_sim.dat", norm_flag=False)
    x = ef.get_ewald_vec(det.coor)
    np.testing.assert_allclose(np.linalg.norm(x), det.ewald_rad, rtol=1e-4)


def test_repr():
    det = ef.detector("data/det_sim.dat", norm_flag=True)
    ans = f"""Detector <{hex(id(det))}>
  Dimension: 3
  Number of pixels: 4096
  Detector distance: 136.364 mm
  Ewald Radius: 136.364 pixel
  q_max : 42.880 pixel
  q_min : 0.707 pixel
  Normalized: True
  Mask: 0 - 3044
        1 - 1000
        2 - 52
"""
    assert repr(det) == ans


def test_getitem():
    det = ef.detector("data/det_sim.dat", norm_flag=True)
    det1 = det[[ef.PixelType.GOOD]]
    det2 = det[det.mask == ef.PixelType.GOOD]
    assert ef.det_isclose(det1, det2)


def test_get_2ddet():
    det = ef.detector("data/det_sim.dat", norm_flag=True)
    det_2d = ef.get_2ddet(det)
    ef.get_2ddet(det, inplace=True)
    assert ef.det_isclose(det, det_2d)
    assert det_2d.ndim == 2


def test_get_3ddet_from_shape():
    det = ef.detector("data/det_sim.dat", norm_flag=True)
    ef.get_3ddet_from_shape((100, 100), det)


def test_set_flags():
    det = ef.detector("data/det_sim.dat", norm_flag=True, writeable=False)
    old_flags = det.flags
    det.setflags()
    assert old_flags == det.flags
    assert not det.flags["writeable"]
    assert det.norm_flag
    assert det.norm_flag  # Check ndim cache is set correctly.


def test_cxy_xyz_conversion():
    from emcfile._detector import cxy_to_xyz, xyz_to_cxy

    det = ef.detector("data/det_sim.dat")
    direction = 1 if det.coor[:, 2].sum() < 0 else -1
    cxy = xyz_to_cxy(det.coor, det.ewald_rad, det.detd * 2, direction)
    xyz = cxy_to_xyz(cxy, det.ewald_rad, det.detd * 2, direction)
    np.testing.assert_almost_equal(xyz, det.coor, decimal=4)


def test_det_render():
    det = ef.detector("data/det_sim.dat")
    detr = ef.det_render(det)
    np.testing.assert_almost_equal(
        detr.to_xyz(detr.to_cxy(det.coor)), det.coor, decimal=4
    )

    np.testing.assert_almost_equal(
        detr.to_xyz(detr.to_cxy(det.coor[0])), det.coor[0], decimal=4
    )
    detr.frame_extent()
    detr.frame_pixels()


@pytest.mark.skipif(not PLT_IMPORTED, reason="Cannot import matplotlib")
def test_plot_rings():
    det = ef.detector("data/det_sim.dat")
    detr = ef.det_render(det)
    fig, ax = plt.subplots()
    detr.plot_rings(ax)
