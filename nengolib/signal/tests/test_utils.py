import numpy as np

from nengolib.signal.utils import nrmse, shift
from nengolib.stats import sphere


def test_nrmse():
    assert np.allclose(nrmse(np.ones(100), np.ones(100)), 0)
    assert np.allclose(nrmse([0, 0, 0], [42, 9000, 0]), 1)

    ideal = sphere.sample(100, 3)
    r = np.asarray([1, 2, 3])
    assert np.allclose(nrmse(ideal, r*ideal, axis=0), 1-1./r)


def test_shift():
    a = [1, 2, 3]
    assert np.allclose(shift(a), [0, 1, 2])

    assert np.allclose(shift(a, 2, 42), [42, 42, 1])

    assert np.allclose(shift(a, 1000), [0, 0, 0])

    a = [[1, 2], [3, 4], [5, 6]]
    assert np.allclose(shift(a), [[0, 0], [1, 2], [3, 4]])
    assert np.allclose(shift(a, 2, [0, 1]), [[0, 1], [0, 1], [1, 2]])
    assert np.allclose(shift(a, 2, 1), [[1, 1], [1, 1], [1, 2]])
