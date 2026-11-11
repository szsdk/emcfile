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


def test_2d_detector_rendering_resampling_and_combined_bitmasks(tmp_path, det):
    det_2d = ef.project_detector_to_2d(det)
    assert det_2d.ndim == 2
    assert np.isfinite(det_2d.coordinates).all()

    det_2d_real = ef.project_detector_to_2d(det_2d, pixel_space="real")
    np.testing.assert_allclose(det_2d_real.coordinates, det_2d.coordinates)

    renderer_2d = ef.detector_renderer(det_2d)
    assert renderer_2d.frame_shape[0] > 1 and renderer_2d.frame_shape[1] > 1
    roundtrip_xyz = renderer_2d.to_xyz(renderer_2d.to_cxy(det_2d.coordinates))
    np.testing.assert_allclose(roundtrip_xyz, det_2d.coordinates)
    img_2d = renderer_2d.render(np.ones(det_2d.num_pix))
    covered = np.asarray(renderer_2d._count.filled(0)) > 0
    assert covered.any()
    assert not np.isnan(img_2d.data[covered]).any()

    resampled_2d = ef.resample_detector((16, 16), det_2d)
    assert np.isfinite(resampled_2d.coordinates).all()

    combo = deepcopy(det)
    combo.mask[:] = ef.PixelType.GOOD
    combo.mask[0] = int(ef.PixelType.CORNER) | int(ef.PixelType.BAD)
    combo.mask[1] = 1 << 10
    assert combo[[ef.PixelType.BAD]].num_pix == 1
    assert combo[[ef.PixelType.CORNER]].num_pix == 1
    assert combo[[ef.PixelType.GOOD]].num_pix == combo.num_pix - 2

    combo_renderer = ef.detector_renderer(combo)
    x0, y0 = combo_renderer.pixel_coordinates[0]
    assert bool(combo_renderer._mask[y0, x0])

    for suffix in (".dat", ".h5"):
        out = tmp_path / f"combo{suffix}"
        combo.write(out)
        restored = ef.detector(out, normalize=False, check_consistency=False)
        np.testing.assert_array_equal(restored.mask, combo.mask)

    coords_2d = det_2d.coordinates[:, :2]
    overridden = ef.detector(
        det_2d, coordinates=coords_2d, normalize=False, check_consistency=False
    )
    assert overridden.coordinates.shape == (det_2d.num_pix, 3)
    with pytest.raises(ValueError, match="same length"):
        ef.detector(det_2d, mask=np.zeros(3, dtype=np.int32), check_consistency=False)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.int16,
        np.uint16,
        np.int8,
        np.uint8,
        np.float32,
        np.float64,
    ],
)
def test_renderer_dense_dtypes_and_fallback_parity(monkeypatch, det, dtype):
    rng = np.random.default_rng(42)
    raw_1d = rng.integers(0, 50, size=det.num_pix).astype(dtype)
    raw_2d = rng.integers(0, 50, size=(4, det.num_pix)).astype(dtype)

    ren_fast = ef.detector_renderer(det)
    out_fast_1d = ren_fast.render(raw_1d)
    out_fast_2d = ren_fast.render(raw_2d)

    monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", "1")
    ren_py = ef.detector_renderer(det)
    out_py_1d = ren_py.render(raw_1d)
    out_py_2d = ren_py.render(raw_2d)

    np.testing.assert_allclose(ren_fast.cxy, ren_py.cxy, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(ren_fast.xy, ren_py.xy)
    np.testing.assert_array_equal(ren_fast._mask, ren_py._mask)
    np.testing.assert_allclose(
        ren_fast._count.data, ren_py._count.data, rtol=1e-14, atol=1e-14
    )

    np.testing.assert_array_equal(out_fast_1d.mask, out_py_1d.mask)
    np.testing.assert_allclose(
        out_fast_1d.data, out_py_1d.data, rtol=1e-14, atol=1e-14, equal_nan=True
    )
    np.testing.assert_array_equal(out_fast_2d.mask, out_py_2d.mask)
    np.testing.assert_allclose(
        out_fast_2d.data, out_py_2d.data, rtol=1e-14, atol=1e-14, equal_nan=True
    )


def test_renderer_strided_deepcopy_and_grid_layouts(monkeypatch, det):
    import pickle

    rng = np.random.default_rng(7)
    base_2d = rng.integers(0, 100, size=(6, det.num_pix * 2), dtype=np.int32)
    strided_1d = base_2d[1, ::2]
    strided_2d = np.asfortranarray(base_2d[::2, ::2])
    assert not strided_1d.flags.c_contiguous
    assert not strided_2d.flags.c_contiguous

    ren = ef.detector_renderer(det)
    res_1d = ren.render(strided_1d)
    res_2d = ren.render(strided_2d)
    np.testing.assert_allclose(
        res_1d.data,
        ren.render(np.ascontiguousarray(strided_1d)).data,
        rtol=1e-14,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        res_2d.data,
        ren.render(np.ascontiguousarray(strided_2d)).data,
        rtol=1e-14,
        equal_nan=True,
    )

    for cloned in (deepcopy(ren), pickle.loads(pickle.dumps(ren))):
        cloned_res = cloned.render(strided_2d)
        np.testing.assert_array_equal(cloned_res.mask, res_2d.mask)
        np.testing.assert_allclose(
            cloned_res.data, res_2d.data, rtol=1e-14, equal_nan=True
        )

    # Verify both column-major (simple_detector) and row-major (exact identity) grids
    det_col = ef.detector(
        coordinates=(32, 32), detector_distance=500, check_consistency=False
    )
    coords_row = (
        det_col.coordinates.reshape(32, 32, 3).transpose(1, 0, 2).reshape(-1, 3).copy()
    )
    det_row = ef.detector(
        coordinates=coords_row,
        mask=np.zeros(32 * 32, dtype=np.uint8),
        correction_factors=det_col.correction_factors.copy(),
        detector_distance=det_col.detector_distance,
        ewald_radius=det_col.ewald_radius,
        check_consistency=False,
    )
    batch = rng.integers(0, 20, size=(3, 32 * 32), dtype=np.int32)
    for grid_det in (det_col, det_row):
        r_fast = ef.detector_renderer(grid_det).render(batch)
        monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", "1")
        r_py = ef.detector_renderer(grid_det).render(batch)
        monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)
        np.testing.assert_array_equal(r_fast.mask, r_py.mask)
        np.testing.assert_allclose(
            r_fast.data, r_py.data, rtol=1e-14, atol=1e-14, equal_nan=True
        )


def test_renderer_direct_sparse_patterns(tmp_path, monkeypatch, det):
    rng = np.random.default_rng(123)
    dense = rng.choice(
        [0, 1, 2, 5], size=(6, det.num_pix), p=[0.85, 0.10, 0.03, 0.02]
    ).astype(np.int32)
    dense[2] = 0  # include an all-zero pattern
    pats = ef.patterns(dense)
    sparse_0 = pats.sparse_pattern(0)

    h5_path = tmp_path / "sparse_render.h5"
    emc_path = tmp_path / "sparse_render.emc"
    pats.write(h5_path)
    pats.write(emc_path)
    f_h5 = ef.open_patterns(h5_path)
    f_emc = ef.open_patterns(emc_path)

    sparse_det = det[np.arange(0, det.num_pix, 2)]
    dense_sub = dense[:, : sparse_det.num_pix]
    pats_sub = ef.patterns(dense_sub)

    for target_det, target_dense, target_pats in (
        (det, dense, pats),
        (sparse_det, dense_sub, pats_sub),
    ):
        ren_fast = ef.detector_renderer(target_det)
        ref_1d = ren_fast.render(target_dense[0])
        ref_2d = ren_fast.render(target_dense)

        sp_1d_fast = ren_fast.render(target_pats.sparse_pattern(0))
        sp_2d_fast = ren_fast.render(target_pats)
        np.testing.assert_array_equal(sp_1d_fast.mask, ref_1d.mask)
        np.testing.assert_allclose(
            sp_1d_fast.data, ref_1d.data, rtol=1e-14, atol=1e-14, equal_nan=True
        )
        np.testing.assert_array_equal(sp_2d_fast.mask, ref_2d.mask)
        np.testing.assert_allclose(
            sp_2d_fast.data, ref_2d.data, rtol=1e-14, atol=1e-14, equal_nan=True
        )

        # Empty pattern array
        empty_res = ren_fast.render(target_pats[:0])
        assert empty_res.shape == (0, ren_fast._render_height, ren_fast._render_width)

        monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", "1")
        ren_py = ef.detector_renderer(target_det)
        sp_1d_py = ren_py.render(target_pats.sparse_pattern(0))
        sp_2d_py = ren_py.render(target_pats)
        monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)

        np.testing.assert_array_equal(sp_1d_py.mask, ref_1d.mask)
        np.testing.assert_allclose(
            sp_1d_py.data, ref_1d.data, rtol=1e-14, atol=1e-14, equal_nan=True
        )
        np.testing.assert_array_equal(sp_2d_py.mask, ref_2d.mask)
        np.testing.assert_allclose(
            sp_2d_py.data, ref_2d.data, rtol=1e-14, atol=1e-14, equal_nan=True
        )

    ren_main = ef.detector_renderer(det)
    ref_main = ren_main.render(dense)
    for f_src in (f_h5, f_emc):
        out_f = ren_main.render(f_src)
        np.testing.assert_array_equal(out_f.mask, ref_main.mask)
        np.testing.assert_allclose(
            out_f.data, ref_main.data, rtol=1e-14, atol=1e-14, equal_nan=True
        )

    wrong_sp = ef.SPARSE_PATTERN(
        det.num_pix + 1, sparse_0.place_ones, sparse_0.place_multi, sparse_0.count_multi
    )
    with pytest.raises(ValueError, match="pixels"):
        ren_main.render(wrong_sp)
    with pytest.raises(ValueError, match="pixels"):
        ren_main.render(pats_sub)


def test_fit_ewald_sphere_center_parity_and_edge_cases(monkeypatch, det):
    fast = pytest.importorskip("emcfile_fast")
    if not hasattr(fast.detector, "fit_ewald_sphere_center_rs"):
        pytest.skip("emcfile-fast does not provide sphere fitting")
    monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)
    # 3D sphere center parity
    v_fast = ef.fit_ewald_sphere_center(det.coordinates)
    monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", "1")
    v_py = ef.fit_ewald_sphere_center(det.coordinates)
    monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)
    np.testing.assert_allclose(v_fast, v_py, rtol=1e-5, atol=1e-5)

    # 2D coplanar detector parity
    det_2d = ef.project_detector_to_2d(det)
    v2_fast = ef.fit_ewald_sphere_center(det_2d.coordinates)
    monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", "1")
    v2_py = ef.fit_ewald_sphere_center(det_2d.coordinates)
    monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)
    np.testing.assert_allclose(v2_fast, v2_py, rtol=1e-12, atol=1e-12)
    assert v2_fast[3] == 0.0

    # Degenerate collinear coordinates raise ValueError
    line_coords = np.column_stack(
        [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)]
    )
    with pytest.raises(ValueError, match="degenerated to a line"):
        ef.fit_ewald_sphere_center(line_coords)
    with pytest.raises(ValueError, match="degenerated to a line"):
        ef.fit_ewald_sphere_center(line_coords[:2])


@pytest.mark.parametrize("size", [4, 31, 32, 1025])
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_fit_sphere_known_center(size, scale, monkeypatch):
    fast = pytest.importorskip("emcfile_fast")
    if not hasattr(fast.detector, "fit_ewald_sphere_center_rs"):
        pytest.skip("emcfile-fast does not provide sphere fitting")
    monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)
    rng = np.random.default_rng(17)
    center = np.array([0.3, -0.2, -1.0]) * scale
    directions = rng.normal(size=(size, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coordinates = center + scale * directions
    original = coordinates.copy()
    result = ef.fit_ewald_sphere_center(coordinates)
    np.testing.assert_allclose(result[:3] / scale, center / scale, atol=1e-7)
    assert result[3] == 1.0
    np.testing.assert_array_equal(coordinates, original)


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("layout", ["float32", "strided", "fortran", "integer"])
def test_fit_sphere_planar_input_layouts(monkeypatch, disabled, layout):
    monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", str(int(disabled)))
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    if layout == "float32":
        coordinates = coordinates.astype(np.float32)
    elif layout == "strided":
        coordinates = np.repeat(coordinates.astype(float), 2, axis=0)[::2]
    elif layout == "fortran":
        coordinates = np.asfortranarray(coordinates, dtype=float)
    original = coordinates.copy()
    np.testing.assert_array_equal(ef.fit_ewald_sphere_center(coordinates), [0, 0, 1, 0])
    np.testing.assert_array_equal(coordinates, original)


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize(
    "coordinates,match",
    [
        (np.zeros(3), "shape"),
        (np.zeros((3, 2)), "shape"),
        (np.zeros((0, 3)), "degenerated to a line"),
        (np.zeros((2, 3)), "degenerated to a line"),
        (np.full((4, 3), np.nan), "finite"),
        (np.full((4, 3), np.inf), "finite"),
    ],
)
def test_fit_sphere_invalid_inputs(monkeypatch, disabled, coordinates, match):
    monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", str(int(disabled)))
    with pytest.raises(ValueError, match=match):
        ef.fit_ewald_sphere_center(coordinates)


def test_fit_sphere_older_extension_fallback(monkeypatch, det):
    from types import SimpleNamespace

    from emcfile import _detector

    monkeypatch.setattr(_detector, "_get_det_rs", lambda: SimpleNamespace())
    result = ef.fit_ewald_sphere_center(det.coordinates)
    np.testing.assert_allclose(result[:3], [0, 0, -128], atol=1e-5)


def test_fit_sphere_noisy_data(monkeypatch):
    fast = pytest.importorskip("emcfile_fast")
    if not hasattr(fast.detector, "fit_ewald_sphere_center_rs"):
        pytest.skip("emcfile-fast does not provide sphere fitting")
    rng = np.random.default_rng(73)
    directions = rng.normal(size=(2000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coordinates = np.array([2.0, -3.0, -20.0]) + 20 * directions
    coordinates += rng.normal(scale=0.01, size=coordinates.shape)
    monkeypatch.delenv("EMCFILE_DISABLE_DET_ACCEL", raising=False)
    actual = ef.fit_ewald_sphere_center(coordinates)
    monkeypatch.setenv("EMCFILE_DISABLE_DET_ACCEL", "1")
    expected = ef.fit_ewald_sphere_center(coordinates)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual[:3], [2, -3, -20], atol=0.002)


def test_fit_sphere_rust_input_validation():
    fast = pytest.importorskip("emcfile_fast")
    fit = getattr(fast.detector, "fit_ewald_sphere_center_rs", None)
    if fit is None:
        pytest.skip("emcfile-fast does not provide sphere fitting")
    with pytest.raises(ValueError, match="finite"):
        fit(np.full((4, 3), np.nan))
    with pytest.raises(ValueError, match="shape"):
        fit(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="contiguous"):
        fit(np.ones((8, 3))[::2])
