import numpy as np

from emcfile._indexing import contiguous_ranges


def test_contiguous_ranges():
    assert np.all(
        contiguous_ranges(np.array([1, 2, 3, 2, 5, 6, 7, 8]))
        == np.array([[1, 4], [2, 3], [5, 9]])
    )

    assert len(contiguous_ranges(np.array([]))) == 0
    assert np.all(contiguous_ranges(np.array([0])) == np.array([[0, 1]]))
    assert np.all(contiguous_ranges(np.array([0, 1])) == np.array([[0, 2]]))
    assert np.all(contiguous_ranges(np.arange(10)) == np.array([[0, 10]]))
    assert np.all(
        contiguous_ranges(
            np.concatenate(
                [
                    np.arange(10),
                    np.arange(10, 20),
                ]
            )
        )
        == np.array([[0, 20]])
    )
    assert np.all(
        contiguous_ranges(
            np.concatenate(
                [
                    np.arange(5, 10),
                    np.arange(20),
                ]
            )
        )
        == np.array([[5, 10], [0, 20]])
    )
