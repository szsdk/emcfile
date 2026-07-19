import contextlib
from typing import Iterator

import numpy as np


@contextlib.contextmanager
def temporary_random_seed(seed: int) -> Iterator[None]:
    """Temporarily set NumPy's global random seed, then restore its state."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class FakeMarimoHTML:
    def __init__(self, text: str):
        self.text = text

    def callout(self, kind: str = "neutral"):
        return self


class FakeMarimo:
    Html = FakeMarimoHTML

    @staticmethod
    def as_html(value: object) -> str:
        return f"<pre>{type(value).__name__}</pre>"
