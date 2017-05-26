import os
from itertools import product, combinations
from collections import Counter

import numpy as np
from nengo.utils.paths import cache_dir

default_cache_file = os.path.join(cache_dir, "leech_kissing.npy")

__all__ = ['leech_kissing', 'default_cache_file']


def leech_kissing(cache_file=None):
    """Generates the 196,560 "kissing points" in 24 dimensions.

    These are the points that are unit distance from the origin in the
    24--dimensional Leech lattice. [#]_ Such points give the optimal
    configuration for sphere-packing in 24 dimensions. [#]_

    Parameters
    ----------
    cache_file : ``string``, optional
        Name of file to cache/retrieve the results of this function call.
        Defaults to Nengo's cache directory + ``"leech_kissing.npy"``.

    Returns
    -------
    ``(196560, 24) np.array``
        The kissing points in 24 dimensions.

    See Also
    --------
    :class:`nengo.dists.CosineSimilarity`

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Leech_lattice
    .. [#] https://en.wikipedia.org/wiki/Kissing_number_problem

    Examples
    --------
    >>> from nengolib.stats import leech_kissing
    >>> pts = leech_kissing()

    We can visualize some of the lattice structure by projections into two
    dimensions. This scatter plot will look the same regardless of which
    two coordinates are chosen.

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(pts[:, 0], pts[:, 1], s=50)
    >>> plt.show()
    """

    # It was surprisingly hard to find any implementations of this online.
    # I instead gathered the details from reading:
    # - https://en.wikipedia.org/wiki/Golay_Code
    # - http://www.markronan.com/mathematics/symmetry-corner/the-golay-code/
    # - http://www.markronan.com/mathematics/symmetry-corner/leech-lattice/

    if cache_file is None:  # pragma: no cover
        cache_file = default_cache_file

    if os.path.exists(cache_file):
        return np.load(cache_file)

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

    np.save(cache_file, kissing)
    return kissing
