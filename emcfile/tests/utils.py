import contextlib

import numpy as np


@contextlib.contextmanager
def temp_seed(seed):
    """
    keep the global random state in a temporary variable and reset it once the function is done
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
