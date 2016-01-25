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
