import runpy
from pathlib import Path

import pytest

scripts = Path(__file__).parent.glob("tutorial*.py")


@pytest.mark.parametrize("script", scripts)
def test_script_execution(script):
    try:
        runpy.run_path(str(script))
    except ModuleNotFoundError as err:
        if "cupy" not in str(err):
            raise err
