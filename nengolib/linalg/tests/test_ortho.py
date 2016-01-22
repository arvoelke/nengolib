import numpy as np
import pytest

from nengolib.linalg import random_orthogonal


@pytest.mark.parametrize("d", [1, 2, 3, 50, 100])
def test_random_orthogonal(d, rng):
    # TODO: could also verify that the coefficients are distributed
    # according to nengo.dists.CosineSimilarity (has been checked manually)
    x = random_orthogonal(d, rng)
    assert np.allclose(np.dot(x, x.T), np.eye(d))
