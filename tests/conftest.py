import pytest


@pytest.fixture
def hdf5_fast():
    for module in ("hdf5plugin", "numba", "zstandard"):
        pytest.importorskip(module)
