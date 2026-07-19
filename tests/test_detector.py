from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import emcfile as ef

DATA_DIR = Path(__file__).resolve().parent.parent / "tmp"


def _require_real_data(filename):
    path = DATA_DIR / filename
    if not path.is_file():
        pytest.skip(f"Optional real-data fixture is unavailable: {path}")
    return path


@pytest.fixture()
def det():
    planar_coordinates = np.mgrid[-32:33, -32:33].reshape(2, -1).T.astype(np.float64)
    num_pixels = len(planar_coordinates)
    coordinates = np.zeros((num_pixels, 3))
    coordinates[:, :2] = planar_coordinates
    ewald_radius = 128
    planar_radii = np.linalg.norm(planar_coordinates, axis=1)
    coordinates[:, 2] = np.sqrt(ewald_radius**2 - planar_radii**2) - ewald_radius
    correction_factors = np.random.uniform(3e-4, 4e-4, num_pixels)
    mask = np.zeros(num_pixels, np.int16)
    mask[planar_radii < 10] = ef.PixelType.BAD
    mask[planar_radii > 32] = ef.PixelType.CORNER
    return ef.detector(
        coordinates=coordinates,
        mask=mask,
        correction_factors=correction_factors,
        detector_distance=ewald_radius * 0.1,
        ewald_radius=ewald_radius,
        normalize=False,
    )


@pytest.fixture(scope="session")
def real_detector():
    return ef.detector(_require_real_data("det_10482_v04_streak_lowq_bin4.h5"))


@pytest.fixture(scope="session")
def real_patterns():
    return ef.patterns(_require_real_data("test.emc"))


@pytest.fixture()
def det_file(tmp_path_factory, det):
    path = tmp_path_factory.mktemp("data") / "det.dat"
    det.write(path)
    return path


def test_detector_array_conversion(det):
    np.array(det)
    np.testing.assert_array_equal(
        det.geometry_array,
        np.concatenate(
            [
                det.coordinates,
                det.correction_factors[:, None],
            ],
            axis=1,
        ),
    )


def test_simple_detector():
    ef.detector(coordinates=(100, 100), detector_distance=2000)


def test_detector_read(det_file):
    ef.detector(det_file, normalize=False)
    with pytest.raises(FileNotFoundError):
        ef.detector("data/det_sim.foo", normalize=False)
    with pytest.raises(TypeError, match="Unsupported detector source type"):
        ef.detector(12, normalize=False)


def test_detector_write(det):
    with TemporaryDirectory() as directory:
        directory_path = Path(directory)
        for suffix in [".dat", ".h5"]:
            path = directory_path / f"detector{suffix}"
            det.write(path)
            det.write(path, overwrite=True)
            with pytest.raises(FileExistsError):
                det.write(path)
            restored_detector = ef.detector(path, normalize=False)
            ef.detector(restored_detector)
            assert ef.detectors_allclose(det, restored_detector)


def test_fit_ewald_sphere_center(det):
    np.testing.assert_almost_equal(det.pixel_size, 0.1)
    center = ef.fit_ewald_sphere_center(det.coordinates)
    np.testing.assert_allclose(np.linalg.norm(center), det.ewald_radius, rtol=1e-4)


def test_repr(det_file):
    det = ef.detector(det_file, normalize=True)
    expected = f"""Detector <{hex(id(det))}>
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
    assert repr(det) == expected


def test_getitem(det):
    selected_by_type = det[[ef.PixelType.GOOD]]
    selected_by_mask = det[det.mask == ef.PixelType.GOOD]
    assert ef.detectors_allclose(selected_by_type, selected_by_mask)


def test_project_detector_to_2d(det):
    projected = ef.project_detector_to_2d(det)
    ef.project_detector_to_2d(det, inplace=True)
    det.check_ewald_rad()
    assert ef.detectors_allclose(det, projected)
    assert projected.ndim == 2


def test_resample_detector(det):
    ef.resample_detector((100, 100), det)


def test_cxy_xyz_conversion(det):
    from emcfile._detector import cxy_to_xyz, xyz_to_cxy

    direction = 1 if det.coordinates[:, 2].sum() < 0 else -1
    projected_coordinates = xyz_to_cxy(det.coordinates, det.ewald_radius, direction)
    coordinates = cxy_to_xyz(projected_coordinates, det.ewald_radius, direction)
    np.testing.assert_almost_equal(coordinates, det.coordinates, decimal=4)


def test_detector_renderer(det):
    renderer = ef.detector_renderer(det)
    np.testing.assert_almost_equal(
        renderer.to_xyz(renderer.to_cxy(det.coordinates)),
        det.coordinates,
        decimal=4,
    )

    np.testing.assert_almost_equal(
        renderer.to_xyz(renderer.to_cxy(det.coordinates[0])),
        det.coordinates[0],
        decimal=4,
    )
    renderer.frame_extent()
    renderer.frame_pixels()


def _reference_render(renderer, image):
    rendered = np.ma.masked_array(
        np.zeros((renderer.frame_shape[1], renderer.frame_shape[0]), dtype="f8"),
        mask=renderer._mask,
    )
    np.add.at(
        rendered,
        (renderer.pixel_coordinates[:, 1], renderer.pixel_coordinates[:, 0]),
        image,
    )
    return rendered / renderer._count


def test_render(det):
    renderer = ef.detector_renderer(det)
    image = det.coordinates[:, 0]
    image_batch = np.tile(image, (5, 1))

    result = renderer.render(image_batch)
    assert result.shape == (5, renderer._render_height, renderer._render_width)

    single = renderer.render(image)
    np.testing.assert_array_almost_equal(single, _reference_render(renderer, image))
    np.testing.assert_array_almost_equal(result[0], single)

    result_1d = renderer.render(image)
    np.testing.assert_array_almost_equal(result_1d, result[0])
    np.testing.assert_array_almost_equal(result[1], result[0])


def test_render_uncovered_pixels_are_nan(det):
    sparse_detector = det[np.arange(0, det.num_pix, 2)]
    renderer = ef.detector_renderer(sparse_detector)
    image = np.ones(sparse_detector.num_pix)
    uncovered = np.asarray(renderer._count.filled(0)) == 0

    assert uncovered.any()
    assert np.isnan(renderer.render(image).data[uncovered]).all()
    assert np.isnan(renderer.render(image[None, :]).data[0, uncovered]).all()


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
        ef.detector_renderer(det).render(np.zeros(shape))


def test_render_real(real_detector, real_patterns):
    renderer = ef.detector_renderer(real_detector)
    dense = real_patterns.todense()
    raw = dense[0].astype(np.float64)

    rendered = renderer.render(raw)
    np.testing.assert_array_almost_equal(rendered, _reference_render(renderer, raw))


def test_render_2d_real(real_detector, real_patterns):
    renderer = ef.detector_renderer(real_detector)
    dense = real_patterns.todense().astype(np.float64)

    result = renderer.render(dense)
    assert result.shape == (
        real_patterns.num_patterns,
        renderer._render_height,
        renderer._render_width,
    )

    single = renderer.render(dense[0])
    np.testing.assert_array_almost_equal(result[0], single, decimal=4)
    np.testing.assert_array_almost_equal(
        result[-1], renderer.render(dense[-1]), decimal=4
    )


def test_display(det):
    html = det._repr_html_()
    assert isinstance(html, str)
    assert "Detector" in html or "detector" in html.lower()

    render_html = ef.detector_renderer(det)._repr_html_()
    assert isinstance(render_html, str)


def test_display_marimo(det):
    # _repr_html_() always returns HTML string
    html = det._repr_html_()
    assert isinstance(html, str)
    assert "Detector" in html or "detector" in html.lower()


def test_concatenate(det):
    mirrored_det = deepcopy(det)
    mirrored_det.coordinates *= np.array([-1, -1, 1])
    assert np.concatenate([det, mirrored_det]).num_pix == 2 * det.num_pix


def test_create_2d_detector():
    half_width = int(10 * 0.5)
    positions = np.linspace(-half_width, half_width, 20)
    grid = np.meshgrid(positions, positions)
    coordinates = np.array([axis.flatten() for axis in grid]).T
    num_pixels = coordinates.shape[0]
    mask = np.zeros(num_pixels, "i4")
    correction_factors = np.ones(num_pixels)
    ef.detector(
        coordinates=coordinates,
        mask=mask,
        correction_factors=correction_factors,
        detector_distance=-1,
        ewald_radius=np.inf,
    )


def test_mask_functions(det):
    det = deepcopy(det)
    np.testing.assert_equal(
        det.select_mask_bits([1], [1]), det.mask == ef.PixelType.BAD
    )
    np.testing.assert_equal(
        det.select_mask_bits([1, 0], [1, 0]),
        det.select_mask_bits(0b11, 0b11),
    )
    modified = deepcopy(det)
    modified.toggle_mask_bits(0b10)
    np.testing.assert_equal(modified.mask, det.mask ^ 0b10)

    modified = deepcopy(det)
    modified.set_mask_bits(0b11, 0b00)
    np.testing.assert_equal(modified.mask, ef.PixelType.GOOD)

    modified = deepcopy(det)
    modified.set_mask_bits(0b11, 0b01)
    np.testing.assert_equal(modified.mask, ef.PixelType.CORNER)


def test_detector_not_equal(det):
    """
    This tests covers the case where two detectors are not equal.
    """
    modified = det[: det.num_pix - 1]
    assert modified != det

    modified = deepcopy(det)
    modified.coordinates[0, 0] = -1000
    assert modified != det

    modified = deepcopy(det)
    modified.correction_factors[0] = -1000
    assert modified != det
