import numpy as np

from emcfile._utils import concat_continous


def test_concat_continous():
    assert np.all(
        concat_continous(np.array([1, 2, 3, 2, 5, 6, 7, 8]))
        == np.array([[1, 4], [2, 3], [5, 9]])
    )

    assert len(concat_continous(np.array([]))) == 0
    assert np.all(concat_continous(np.array([0])) == np.array([[0, 1]]))
    assert np.all(concat_continous(np.array([0, 1])) == np.array([[0, 2]]))
    assert np.all(concat_continous(np.arange(10)) == np.array([[0, 10]]))
    assert np.all(
        concat_continous(
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
        concat_continous(
            np.concatenate(
                [
                    np.arange(5, 10),
                    np.arange(20),
                ]
            )
        )
        == np.array([[5, 10], [0, 20]])
    )
