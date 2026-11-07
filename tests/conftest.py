import pytest


@pytest.fixture
def hdf5_fast():
    for module in ("hdf5plugin", "emcfile_fast"):
        pytest.importorskip(module)
