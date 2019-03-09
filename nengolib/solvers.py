import warnings

import numpy as np

from nengo.solvers import Solver, LstsqL2
from nengo.version import version_info

if version_info < (2, 5, 0):  # pragma: no cover
    Temporal = None  # not supported (requires nengo PR #1313)
else:
    from nengolib.temporal import Temporal

__all__ = ['Temporal']


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

    def __call__(self, A, Y, __hack__=None, **kwargs):
        assert __hack__ is None
        # __hack__ is necessary prior to nengo PR #1359 (<2.6.1)
        # and following nengo PR #1507 (>2.8.0)

        if self.bias is not None:
            # this is okay if due to multiple builds of the same network (#99)
            warnings.warn("%s called twice; ensure not being shared between "
                          "multiple connections" % type(self).__name__,
                          UserWarning)
        scale = A.max()  # to make regularization consistent
        AB = np.empty((A.shape[0], A.shape[1] + 1))
        AB[:, :-1] = A
        AB[:, -1] = scale
        XB, solver_info = self.solver.__call__(AB, Y, **kwargs)
        solver_info['bias'] = self.bias = XB[-1, :] * scale
        return XB[:-1, :], solver_info

    def bias_function(self, size):
        """Returns the function for the pre-synaptic bias node."""
        return lambda _: np.zeros(size) if self.bias is None else self.bias
