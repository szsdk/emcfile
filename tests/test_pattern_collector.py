import numpy as np
import pytest

import emcfile as ef


def test_collector(tmp_path):
    collector = ef.EMCPatternCollector(batch_size=16)
    html = collector._repr_html_()
    assert isinstance(html, str)
    assert "Pattern collector" in html or "collector" in html.lower()

    images = []
    for _ in range(30):
        image = np.random.randint(32, size=34)
        collector.append(image)
        images.append(image)
    expected = ef.patterns(np.array(images))
    assert expected == collector.to_patterns()[:]

    # test write
    collector.write(tmp_path / "test.emc")
    assert expected == ef.patterns(tmp_path / "test.emc")

    # test append with wrong size
    with pytest.raises(ValueError):
        collector.append(np.random.randint(32, size=35))

    collector.extend(images)
    collector.extend(tuple(images))
    collector.extend(np.array(images))
    collector.extend(ef.patterns(np.array(images)))
    assert np.concatenate(collector.pattern_batches()) == ef.patterns(
        np.array(images * 5)
    )

    with pytest.raises(ValueError):
        collector.extend([np.random.randint(32, size=35)])
    with pytest.raises(ValueError):
        collector.extend(ef.patterns(np.zeros((1, 35), dtype=int)))
    with pytest.raises(TypeError):
        collector.extend([object()])
    with pytest.raises(TypeError):
        collector.extend(iter(images))

    html = collector._repr_html_()
    assert isinstance(html, str)
