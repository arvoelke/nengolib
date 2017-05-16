import numpy as np

import nengo
from nengo.dists import Choice

from nengolib.stats.ntmdists import cube

__all__ = ['EnsembleCube']


class EnsembleCube(nengo.Ensemble):
    """An EnsembleArray flattened into a single Ensemble.

    This is necessary any time a desired function is of the form:
    ``f(x_1, ..., x_D) = f_1(x_1) + ... + f_D(x_D)``
    for some unknown functions ``f_1``, ..., ``f_D`` and a ``D``-dimensional
    representation. Usually each "sub-function" is a low-order polynomial.

    For example, the optimized Product network is an instance of this
    network after a change of basis: ``[[1, 1], [1, -1]]``, because of
    the equivalence ``xy = 1/4 ((x + y)^2 - (x - y)^2)``.

    This network is useful whenever ``x_1``, ..., ``x_D`` are given / fixed,
    but the optimal corresponding sub-functions are not explicitly known.
    """

    # TODO: support transparent change of basis?

    # TODO: In theory linear transformations coming from the EnsembleCube
    # can be solved more efficiently. The obvious solution here is to use
    # an EnsembleArray, but then you lose the above benefits.

    def __init__(self, n_neurons, dimensions, half_width=1, eval_points=cube,
                 **ens_kwargs):
        self.half_width = half_width

        for illegal in ('radius', 'encoders'):
            if illegal in ens_kwargs:
                raise ValueError("do not supply '%s' keyword argument" %
                                 illegal)

        I = np.eye(dimensions)
        encoders = Choice(np.vstack([I, -I]))  # gain scaled by half_width

        super(EnsembleCube, self).__init__(
            n_neurons,
            dimensions,
            radius=half_width,
            encoders=encoders,
            eval_points=eval_points,
            **ens_kwargs)
