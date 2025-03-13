import numpy as np
import pytest

import emcfile as ef


def test_collector(tmp_path):
    cl = ef.PatternsSOneCollector(16)
    imgs = []
    for _ in range(30):
        img = np.random.randint(32, size=34)
        cl.append(img)
        imgs.append(img)
    ref = ef.patterns(np.array(imgs))
    assert ref == cl.patterns()

    # test write
    cl.write(tmp_path / "test.emc")
    assert ref == ef.patterns(tmp_path / "test.emc")

    # test append with wrong size
    with pytest.raises(ValueError):
        cl.append(np.random.randint(32, size=35))

    cl.extend(imgs)
    cl.extend(np.array(imgs))
    cl.extend(ef.patterns(np.array(imgs)))
    assert np.concatenate(cl.pattern_list()) == ef.patterns(np.array(imgs * 4))
