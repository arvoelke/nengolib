import numpy as np

from nengolib.signal.utils import nrmse
from nengolib.stats import sphere


def test_nrmse():
    assert np.allclose(nrmse(np.ones(100), np.ones(100)), 0)
    assert np.allclose(nrmse([0, 0, 0], [42, 9000, 0]), 1)

    ideal = sphere.sample(100, 3)
    r = np.asarray([1, 2, 3])
    assert np.allclose(nrmse(ideal, r*ideal, axis=0), 1-1./r)
