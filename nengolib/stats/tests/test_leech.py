import os

import numpy as np

from nengo.utils.numpy import norm

from nengolib.stats.leech import leech_kissing, _leech_kissing_cache_file


def test_leech_kissing():
    x = leech_kissing()

    assert x.shape == (196560, 24)
    assert np.allclose(norm(x, axis=1, keepdims=True), 1)
    assert len(set(map(tuple, x))) == len(x)  # no duplicates
    assert os.path.exists(_leech_kissing_cache_file)

    x_cached = leech_kissing()
    assert np.allclose(x, x_cached)

    s = np.random.choice(len(x), 1000, replace=False)
    # The only vectors in x that have dot product with s > 0.5 are those
    # from s themselves, and so this is exactly 1 (per vector in s)
    assert np.all(np.sum(np.dot(x, x[s].T) > 0.5, axis=0) == 1)
