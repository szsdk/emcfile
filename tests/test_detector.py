from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import emcfile as ef

DATA_DIR = Path(__file__).resolve().parent.parent / "tmp"


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


@pytest.fixture(scope="session")
def real_det():
    return ef.detector(DATA_DIR / "det_10482_v04_streak_lowq_bin4.h5")


@pytest.fixture(scope="session")
def real_patterns():
    return ef.patterns(DATA_DIR / "test.emc")


@pytest.fixture()
def det_file(tmp_path_factory, det):
    fn = tmp_path_factory.mktemp("data") / "det.dat"
    det.write(fn)
    return fn


def test_det_operation(det):
    np.array(det)
    np.testing.assert_array_equal(
        det.coor_factor, np.concatenate([det.coor, det.factor[:, None]], axis=1)
    )


def test_simple_detector():
    ef.detector(coor=(100, 100), detd=2000)


def test_det_read(det_file):
    ef.detector(det_file, norm_flag=False)
    with pytest.raises(FileNotFoundError):
        ef.detector("data/det_sim.foo", norm_flag=False)
    with pytest.raises(TypeError, match="Unsupported detector source type"):
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
    det.check_ewald_rad()
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


def reference_render(detr, raw_img):
    img = np.ma.masked_array(
        np.zeros((detr.frame_shape[1], detr.frame_shape[0]), dtype="f8"),
        mask=detr._mask,
    )
    np.add.at(img, (detr.xy[:, 1], detr.xy[:, 0]), raw_img)
    return img / detr._count


def test_render(det):
    detr = ef.det_render(det)
    raw_1d = det.coor[:, 0]
    raw_2d = np.tile(raw_1d, (5, 1))

    result = detr.render(raw_2d)
    assert result.shape == (5, detr._render_H, detr._render_W)

    single = detr.render(raw_1d)
    np.testing.assert_array_almost_equal(single, reference_render(detr, raw_1d))
    np.testing.assert_array_almost_equal(result[0], single)

    result_1d = detr.render(raw_1d)
    np.testing.assert_array_almost_equal(result_1d, result[0])
    np.testing.assert_array_almost_equal(result[1], result[0])


def test_render_uncovered_pixels_are_nan(det):
    sparse_det = det[np.arange(0, det.num_pix, 2)]
    detr = ef.det_render(sparse_det)
    raw = np.ones(sparse_det.num_pix)
    uncovered = np.asarray(detr._count.filled(0)) == 0

    assert uncovered.any()
    assert np.isnan(detr.render(raw).data[uncovered]).all()
    assert np.isnan(detr.render(raw[None, :]).data[0, uncovered]).all()


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (2, 3, 4),
        (10,),
        (2, 10),
    ],
)
def test_render_validates_input_shape(det, shape):
    with pytest.raises(ValueError):
        ef.det_render(det).render(np.zeros(shape))


def test_render_real(real_det, real_patterns):
    detr = ef.det_render(real_det)
    dense = real_patterns.todense()
    raw = dense[0].astype(np.float64)

    img = detr.render(raw)
    np.testing.assert_array_almost_equal(img, reference_render(detr, raw))


def test_render_2d_real(real_det, real_patterns):
    detr = ef.det_render(real_det)
    dense = real_patterns.todense().astype(np.float64)

    result = detr.render(dense)
    assert result.shape == (real_patterns.num_data, detr._render_H, detr._render_W)

    single = detr.render(dense[0])
    np.testing.assert_array_almost_equal(result[0], single, decimal=4)
    np.testing.assert_array_almost_equal(result[-1], detr.render(dense[-1]), decimal=4)


def test_display(det):
    html = det._repr_html_()
    assert isinstance(html, str)
    assert "Detector" in html or "detector" in html.lower()

    render_html = ef.det_render(det)._repr_html_()
    assert isinstance(render_html, str)


def test_display_marimo(det):
    # _repr_html_() always returns HTML string
    html = det._repr_html_()
    assert isinstance(html, str)
    assert "Detector" in html or "detector" in html.lower()


def test_concatenate(det):
    det_sym = deepcopy(det)
    det_sym.coor *= np.array([-1, -1, 1])
    assert np.concatenate([det, det_sym]).num_pix == 2 * det.num_pix


def test_2ddet():
    l_half = int(10 * 0.5)
    s = np.linspace(-l_half, l_half, 20)
    xy = np.meshgrid(s, s)
    coor = np.array([i.flatten() for i in xy]).T
    num_pix = coor.shape[0]
    mask = np.zeros(num_pix, "i4")
    factor = np.ones(num_pix)
    ef.detector(coor=coor, mask=mask, factor=factor, detd=-1, ewald_rad=np.inf)


def test_mask_functions(det):
    det = deepcopy(det)
    np.testing.assert_equal(det.mask_select([1], [1]), det.mask == ef.PixelType.BAD)
    np.testing.assert_equal(
        det.mask_select([1, 0], [1, 0]), det.mask_select(0b11, 0b11)
    )
    det1 = deepcopy(det)
    det1.mask_flip(0b10)
    np.testing.assert_equal(det1.mask, det.mask ^ 0b10)

    det1 = deepcopy(det)
    det1.mask_set(0b11, 0b00)
    np.testing.assert_equal(det1.mask, ef.PixelType.GOOD)

    det1 = deepcopy(det)
    det1.mask_set(0b11, 0b01)
    np.testing.assert_equal(det1.mask, ef.PixelType.CORNER)


def test_detector_not_equal(det):
    """
    This tests covers the case where two detectors are not equal.
    """
    a = det[: det.num_pix - 1]
    assert a != det

    a = deepcopy(det)
    a.coor[0, 0] = -1000
    assert a != det

    a = deepcopy(det)
    a.factor[0] = -1000
    assert a != det
