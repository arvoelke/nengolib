import os
from itertools import product, combinations
from collections import Counter

import numpy as np
from nengo.utils.paths import cache_dir

__all__ = ['leech_kissing']

_leech_kissing_cache_file = os.path.join(cache_dir, "leech_kissing.npy")


def leech_kissing():
    if os.path.exists(_leech_kissing_cache_file):
        return np.load(_leech_kissing_cache_file)

    # https://en.wikipedia.org/wiki/Golay_Code
    # http://www.markronan.com/mathematics/symmetry-corner/the-golay-code/
    # http://www.markronan.com/mathematics/symmetry-corner/leech-lattice/

    A = [
        [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]]

    G = np.append(np.eye(12), A, axis=1)
    assert G.shape == (12, 24)

    golay_code = np.empty((2**12, 24), dtype=int)
    for i, w in enumerate(product((0, 1), repeat=12)):
        golay_code[i] = np.dot(w, G) % 2

    weights = np.sum(golay_code, axis=1, dtype=int)
    assert sorted(Counter(weights).items()) == [
        (0, 1), (8, 759), (12, 2576), (16, 759), (24, 1)]

    octets = golay_code[weights == 8]
    assert len(octets) == 759

    witt_group = []
    for signs in product((-1, 1), repeat=8):
        # check even number of minus signs
        if np.prod(signs, dtype=int) == 1:
            v = 2*np.asarray(signs)
            for octet in octets:
                u = octet.copy()
                u[u == 1] *= v
                witt_group.append(u)
    assert len(witt_group) == 97152

    corner_group = []
    for corner in combinations(range(24), 2):
        corner = list(corner)  # for numpy slicing
        for signs in product((-1, 1), repeat=2):
            u = np.zeros(24)
            u[corner] = 4*np.asarray(signs)
            corner_group.append(u)
    assert len(corner_group) == 1104

    golay_group = []
    for i in range(24):
        for signs in golay_code:
            u = np.ones(24)
            u[i] = -3
            u[signs == 1] *= -1
            golay_group.append(u)
    assert len(golay_group) == 98304

    kissing = np.asarray(
        witt_group + corner_group + golay_group, dtype=np.float64)
    kissing /= np.sqrt(32)

    np.save(_leech_kissing_cache_file, kissing)
    return kissing
