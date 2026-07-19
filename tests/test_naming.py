import importlib
import sys

import numpy as np
import pytest

import emcfile as ef


def test_emc_pattern_names_preserve_runtime_identity():
    patterns = ef.patterns(np.array([[0, 1, 2]], dtype=np.int32))

    assert ef.EMCPatternArray is ef.PatternsSOne
    assert isinstance(patterns, ef.EMCPatternArray)
    assert patterns.num_patterns == patterns.num_data == 1
    assert patterns.num_pixels == patterns.num_pix == 3
    assert patterns.ones_offsets is patterns.ones_idx
    assert patterns.multi_offsets is patterns.multi_idx
    assert patterns.ones.tolist() == [1]
    assert patterns.multi.tolist() == [1]


def test_pattern_file_names_preserve_runtime_identity(tmp_path):
    path = tmp_path / "patterns.emc"
    ef.patterns(np.array([[0, 1, 2]], dtype=np.int32)).write(path)

    source = ef.open_patterns(path)
    assert ef.EMCBinaryPatternFile is ef.PatternsSOneEMC
    assert isinstance(source, ef.EMCBinaryPatternFile)
    assert source.num_patterns == source.num_data
    assert source.num_pixels == source.num_pix


def test_descriptive_writer_and_collector_keywords(tmp_path):
    collector = ef.EMCPatternCollector(batch_size=1)
    collector.append(np.array([0, 1, 2], dtype=np.int32))
    assert collector.batch_size == collector.max_buffer_size == 1

    path = ef.as_hdf5_path(tmp_path / "patterns.h5", "/patterns")
    collector.to_patterns().write(path, hdf5_version="2")
    source = ef.open_patterns(path)
    assert isinstance(source, ef.HDF5PatternFile)


def test_hdf5_path_field_alternatives(tmp_path):
    path = ef.as_hdf5_path(tmp_path / "data.h5", "/data")

    assert path.file_path == path.fn
    assert path.object_path == path.gn
    assert ef.as_path(str(path)) == path
    assert ef.is_hdf5_path(path)


def test_detector_field_alternatives_are_synchronized():
    detector = ef.detector(coordinates=(3, 4), detector_distance=10, normalize=False)

    assert detector.coordinates is detector.coor
    assert detector.correction_factors is detector.factor
    assert detector.detector_distance == detector.detd
    assert detector.ewald_radius == detector.ewald_rad
    assert detector.is_normalized == detector.norm_flag
    np.testing.assert_array_equal(detector.geometry_array, detector.coor_factor)

    replacement = detector.coordinates.copy()
    detector.coordinates = replacement
    assert detector.coor is replacement


def test_renderer_field_alternatives_preserve_original_fields():
    detector = ef.detector(coordinates=(3, 4), detector_distance=10)
    renderer = ef.DetectorRenderer(detector)

    assert ef.DetectorRenderer is ef.DetRender
    assert renderer.detector is detector
    assert renderer.projected_coordinates is renderer.cxy
    assert renderer.pixel_coordinates is renderer.xy


def test_superseded_callables_have_pep_702_deprecation_markers():
    deprecated_callables = [
        ef.det_render,
        ef.det_isclose,
        ef.get_2ddet,
        ef.get_3ddet_from_shape,
        ef.get_ewald_vec,
        ef.PatternsSOne.get_mean_count,
        ef.PatternsSOne.check_indices_ordered,
        ef.PatternsSOne.ensure_indices_ordered,
    ]

    assert all(
        hasattr(callable_, "__deprecated__") for callable_ in deprecated_callables
    )


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("_collector", "_pattern_collector"),
        ("_h5helper", "_hdf5"),
        ("_pattern_sone", "_emc_patterns"),
        ("_pattern_sone_file", "_pattern_files"),
        ("_patterns", "_pattern_factory"),
        ("_utils", "_indexing"),
    ],
)
def test_deprecated_modules_are_exact_aliases(old_name, new_name):
    old_qualified_name = f"emcfile.{old_name}"
    sys.modules.pop(old_qualified_name, None)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        old_module = importlib.import_module(old_qualified_name)

    new_module = importlib.import_module(f"emcfile.{new_name}")
    assert old_module is new_module


def test_deprecated_misc_module_preserves_both_utility_groups():
    sys.modules.pop("emcfile._misc", None)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        old_module = importlib.import_module("emcfile._misc")

    assert old_module.pretty_size(1024) == "1.00 KB"
    assert old_module.split_range(0, 4, 2) == [(0, 2), (2, 4)]
