import numpy as np

from nengo.exceptions import NengoException
from nengo.solvers import Solver, LstsqL2

__all__ = ['BiasedSolver']


class BiasedSolver(Solver):
    """Wraps a solver with a bias neuron, and extracts its weights."""

    def __init__(self, solver=LstsqL2(), magnitude=1.0):
        self.magnitude = magnitude
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
            raise NengoException("can only use %s once; create a new instance "
                                 "per connection" % self.__class__.__name__)
        scale = A.max()  # to make regularization consistent
        AB = np.empty((A.shape[0], A.shape[1] + 1))
        AB[:, :-1] = A
        AB[:, -1] = scale * self.magnitude
        XB, solver_info = self.solver.__call__(AB, Y, rng=rng, E=E)
        solver_info['bias'] = self.bias = XB[-1, :] * scale
        return XB[:-1, :], solver_info

    def bias_function(self, size):
        """Returns the function for the pre-synaptic bias node."""
        return lambda x: np.zeros(size) if self.bias is None else x * self.bias
