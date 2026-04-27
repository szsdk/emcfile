import contextlib
from typing import Iterator

import numpy as np


@contextlib.contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    """
    keep the global random state in a temporary variable and reset it once the function is done
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class FakeMarimoHtml:
    def __init__(self, text: str):
        self.text = text

    def callout(self, kind: str = "neutral"):
        return self


class FakeMarimo:
    Html = FakeMarimoHtml

    @staticmethod
    def as_html(value: object) -> str:
        return f"<pre>{type(value).__name__}</pre>"
