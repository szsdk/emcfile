import numpy as np
import pytest

import emcfile as ef


def test_collector():
    cl = ef.PatternsSOneCollector(16)
    imgs = []
    for _ in range(30):
        img = np.random.randint(32, size=34)
        cl.append(img)
        imgs.append(img)
    assert ef.patterns(np.array(imgs)) == cl.patterns()

    with pytest.raises(ValueError):
        cl.append(np.random.randint(32, size=35))
