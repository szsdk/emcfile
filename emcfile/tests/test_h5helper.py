import tempfile
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pytest

import emcfile as ef


def test_h5group():
    rand_data = np.random.rand(222)
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        with ef.h5group(f"{f.name}::group", "a") as (_, g):
            g.create_dataset("data", data=rand_data)
        with ef.h5group(f"{f.name}::group", "r") as (_, g):
            np.testing.assert_array_equal(g["data"][...], rand_data)


@pytest.mark.parametrize(
    "fname, result",
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
def test_h5path(fname, result):
    if result is None:
        with pytest.raises(ValueError):
            ef.h5path(fname)
    else:
        assert tuple(ef.h5path(fname)) == result


@pytest.mark.parametrize(
    "fname, result",
    [
        ("/tmp/vx.h5::inten", ef.h5path("/tmp/vx.h5", "inten")),
        ("/tmp/vx.h5", ef.h5path("/tmp/vx.h5", "/")),
        ("/tmp/vx.txt", Path("/tmp/vx.txt")),
    ],
)
def test_make_path(fname, result):
    assert ef.make_path(fname) == result


def _compare_dict(d1, d2):
    if d1.keys() != d2.keys():
        return False
    for k, v in d1.items():
        if isinstance(v, np.ndarray):
            if not np.all(v == d2[k]):
                return False
        elif isinstance(v, dict):
            if not _compare_dict(v, d2[k]):
                return False
        else:
            if not v == d2[k]:
                return False
    return True


def test_obj_h5():
    with tempfile.NamedTemporaryFile(suffix=".h5") as fth5:
        obj = {"name": "sz", "age": 27, "data": {"test": np.random.rand(3, 5)}}
        obj_path = f"{fth5.name}::person"
        ef.write_obj_h5(obj_path, obj, overwrite=False)
        with pytest.raises(Exception):
            ef.write_obj_h5(obj_path, obj, overwrite=False)
        ef.write_obj_h5(obj_path, obj, overwrite=True)
        assert _compare_dict(ef.read_obj_h5(obj_path), obj)

        obj = {"name": "sz", "age": 27, ".": np.random.rand(3, 5)}
        ef.write_obj_h5(obj_path, obj, overwrite=True, verbose=True)
        assert _compare_dict(ef.read_obj_h5(obj_path), obj)


@pytest.fixture(
    params=[
        ("numpy.npy", np.random.rand(10)),
        ("g.h5::group", np.random.rand(10)),
        ("g.bin", np.random.rand(32)),
        ("g.h5::gro", None),
        ("g.npy", None),
    ]
)
def cases_read_array(request, tmp_path):
    fn = tmp_path / request.param[0]
    result = request.param[1]
    if request.param[1] is None:
        return fn, result, pytest.raises(FileNotFoundError)
    ef.write_array(fn, result)
    return fn, result, does_not_raise()


def test_read_array(cases_read_array):
    filename, result, expectation = cases_read_array
    with expectation:
        np.testing.assert_array_equal(ef.read_array(filename), result)
