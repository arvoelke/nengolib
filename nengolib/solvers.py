import warnings

import numpy as np

from nengo.solvers import Solver, LstsqL2


class BiasedSolver(Solver):
    """Wraps a solver with a bias neuron, and extracts its weights.

    This is setup correctly by nengolib.Connection; not to be used directly.
    """

    def __init__(self, solver=LstsqL2()):
        self.solver = solver
        self.bias = None
        try:
            # parent class changed in Nengo 2.1.1
            # need to do this because self.weights is read-only
            super(BiasedSolver, self).__init__(weights=solver.weights)
        except TypeError:  # pragma: no cover
            super(BiasedSolver, self).__init__()
            self.weights = solver.weights

    def __call__(self, A, Y, rng=None, E=None):
        if self.bias is not None:
            # this is okay if due to multiple builds of the same network (#99)
            warnings.warn("%s called twice; ensure not being shared between "
                          "multiple connections" % type(self).__name__,
                          UserWarning)
        scale = A.max()  # to make regularization consistent
        AB = np.empty((A.shape[0], A.shape[1] + 1))
        AB[:, :-1] = A
        AB[:, -1] = scale
        XB, solver_info = self.solver.__call__(AB, Y, rng=rng, E=E)
        solver_info['bias'] = self.bias = XB[-1, :] * scale
        return XB[:-1, :], solver_info

    def bias_function(self, size):
        """Returns the function for the pre-synaptic bias node."""
        return lambda _: np.zeros(size) if self.bias is None else self.bias
