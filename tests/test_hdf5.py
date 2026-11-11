from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest

import emcfile as ef


def test_hdf5_group(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test.h5"
    expected = np.random.rand(222)
    with ef.h5group(f"{path}::group", "a") as (_, group):
        group.create_dataset("data", data=expected)
    with ef.h5group(f"{path}::group", "r") as (_, group):
        np.testing.assert_array_equal(cast(h5py.Dataset, group["data"])[...], expected)


@pytest.mark.parametrize(
    "path_string, expected",
    [
        ("/tmp/tmpl6ppiovx.h5::inten", (Path("/tmp/tmpl6ppiovx.h5"), "inten")),
        ("few.h5", (Path("few.h5"), "/")),
        ("few.h5::", (Path("few.h5"), "/")),
        ("few.h5::/", (Path("few.h5"), "/")),
        ("few.txt::", (Path("few.txt"), "/")),
        ("fwef.txt", None),
        ("fewf.txt::32::33", None),
    ],
)
def test_as_hdf5_path(path_string, expected):
    if expected is None:
        with pytest.raises(ValueError):
            ef.as_hdf5_path(path_string)
    else:
        assert tuple(ef.as_hdf5_path(path_string)) == expected


@pytest.mark.parametrize(
    "path_string, expected",
    [
        ("/tmp/vx.h5::inten", ef.as_hdf5_path("/tmp/vx.h5", "inten")),
        ("/tmp/vx.h5", ef.as_hdf5_path("/tmp/vx.h5", "/")),
        ("/tmp/vx.txt", Path("/tmp/vx.txt")),
    ],
)
def test_as_path(path_string, expected):
    assert ef.as_path(path_string) == expected


def test_hdf5_path_display(tmp_path):
    path = ef.as_hdf5_path(tmp_path / "test.h5", "group")
    html = path._repr_html_()
    assert isinstance(html, str)
    assert "test.h5" in html or "HDF5" in html


def _objects_equal(actual, expected):
    if actual.keys() != expected.keys():
        return False
    for key, value in actual.items():
        if isinstance(value, np.ndarray):
            if not np.all(value == expected[key]):
                return False
        elif isinstance(value, dict):
            if not _objects_equal(value, expected[key]):
                return False
        else:
            if value != expected[key]:
                return False
    return True


def test_hdf5_object_roundtrip(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test.h5"
    value = {
        "name": "sz",
        "age": 27,
        "data": {"test": np.random.rand(3, 5)},
        "datatype": np.dtype([("a", int), ("b", float)]),
    }
    object_path = f"{path}::person"
    ef.write_hdf5_object(object_path, value, overwrite=False)
    with pytest.raises(FileExistsError):
        ef.write_hdf5_object(object_path, value, overwrite=False)
    ef.write_hdf5_object(object_path, value, overwrite=True)
    assert _objects_equal(ef.read_hdf5_object(object_path), value)

    value = {"name": "sz", "age": 27, ".": np.random.rand(3, 5)}
    ef.write_hdf5_object(object_path, value, overwrite=True, verbose=True)
    assert _objects_equal(ef.read_hdf5_object(object_path), value)


def test_write_hdf5_object_does_not_mutate_input_on_error(tmp_path):
    value = {".": np.arange(3), "invalid": object()}

    with pytest.raises(TypeError):
        ef.write_hdf5_object(f"{tmp_path / 'test.h5'}::data", value)

    np.testing.assert_array_equal(value["."], np.arange(3))


@pytest.fixture(
    params=[
        ("numpy.npy", np.random.rand(10)),
        ("g.h5::group", np.random.rand(10)),
        ("g.bin", np.random.rand(32)),
        ("g.h5::gro", None),
        ("g.npy", None),
    ]
)
def array_file_case(request, tmp_path):
    path = tmp_path / request.param[0]
    expected = request.param[1]
    if request.param[1] is None:
        return path, expected, pytest.raises(FileNotFoundError)
    ef.write_array(path, expected)
    return path, expected, does_not_raise()


def test_read_array(array_file_case):
    path, expected, expectation = array_file_case
    with expectation:
        np.testing.assert_array_equal(ef.read_array(path), expected)


def test_h5path_truediv_uses_posix_separators():
    root = ef.as_hdf5_path("sample.h5", "/")
    sub = root / "entry" / "data"
    assert sub.gn == "/entry/data"
    nested = ef.as_hdf5_path("sample.h5", "entry") / "sub/group"
    assert nested.gn == "entry/sub/group"


def test_write_hdf5_object_root_and_overwrite_removes_stale_keys(tmp_path):
    root_h5 = tmp_path / "root_obj.h5"
    first = {"alpha": 1, "nested": {"old_key": np.arange(3)}}
    ef.write_hdf5_object(f"{root_h5}::/", first)
    assert _objects_equal(ef.read_hdf5_object(f"{root_h5}::/"), first)

    with pytest.raises(FileExistsError):
        ef.write_hdf5_object(f"{root_h5}::/", {"beta": 2}, overwrite=False)

    second = {"beta": 2, "nested": {"new_key": np.arange(2)}}
    ef.write_hdf5_object(f"{root_h5}::/", second, overwrite=True)
    restored = ef.read_hdf5_object(f"{root_h5}::/")
    assert _objects_equal(restored, second)
    assert "alpha" not in restored
    assert "old_key" not in restored["nested"]

    bad_h5 = tmp_path / "failed_obj.h5"
    with pytest.raises(TypeError):
        ef.write_hdf5_object(f"{bad_h5}::/sub", {"bad": object()})
    assert not bad_h5.exists()


def test_h5path_exists_on_corrupt_file_and_read_array_dtype(tmp_path):
    not_h5 = tmp_path / "empty_or_corrupt.h5"
    not_h5.write_bytes(b"")
    assert not ef.as_hdf5_path(not_h5, "/").exists()
    not_h5.write_bytes(b"not an hdf5 file")
    assert not ef.as_hdf5_path(not_h5, "/group").exists()

    arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    for name in ("arr.npy", "arr.h5::/vec"):
        p = tmp_path / name
        ef.write_array(p, arr)
        got_f32 = ef.read_array(p, dtype=np.float32)
        assert got_f32.dtype == np.float32
        np.testing.assert_allclose(got_f32, arr.astype(np.float32))

    p_bin = tmp_path / "arr_f32.bin"
    ef.write_array(p_bin, arr.astype(np.float32))
    got_bin_f32 = ef.read_array(p_bin, dtype=np.float32)
    assert got_bin_f32.dtype == np.float32
    np.testing.assert_allclose(got_bin_f32, arr.astype(np.float32))


