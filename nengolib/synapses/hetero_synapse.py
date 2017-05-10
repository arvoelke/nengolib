import numpy as np
from scipy.linalg import block_diag

from nengo.utils.compat import is_iterable

from nengolib.signal.discrete import cont2discrete
from nengolib.signal.system import LinearSystem

__all__ = ['HeteroSynapse']


class HeteroSynapse(object):
    """Callable class for applying different synapses to a vector.

    If `elementwise == False` (default), each synapse is applied to every
        dimension, and so `size_out == size_in * len(synapses)`. The
        output dimensions are ordered by input dimension, such that
        index `i*len(synapses) + j` is the `i`'th input dimension convolved
        with the `j`'th filter.

    If `elementwise == True`, `len(synapses)` must match `size_in`, in
        which case each synapse is applied separately to each dimension,
        and so `size_out == size_in`.

    The latter can be used to connect to a population of neurons with a
    different synapse for each neuron.
    """

    def __init__(self, systems, dt=None, elementwise=False, method='zoh'):
        if not is_iterable(systems) or isinstance(systems, LinearSystem):
            systems = [systems]
        self.systems = systems
        self.dt = dt
        self.elementwise = elementwise

        self.A = []
        self.B = []
        self.C = []
        self.D = []
        for sys in systems:
            sys = LinearSystem(sys)
            if dt is not None:
                sys = cont2discrete(sys, dt, method=method)
            elif sys.analog:
                raise ValueError(
                    "system (%s) must be digital if not given dt" % sys)

            A, B, C, D = sys.ss
            self.A.append(A)
            self.B.append(B)
            self.C.append(C)
            self.D.append(D)

        # TODO: If all of the synapses are single order, than A is diagonal
        # and so np.dot(self.A, self._x) is trivial. But perhaps
        # block_diag is already optimized for this.

        # Note: ideally we could put this into CCF to reduce the A mapping
        # to a single dot product and a shift operation. But in general
        # since this is MIMO it is not controllable from a single input.
        # Instead we might want to consider balanced reduction to
        # improve efficiency.
        self.A = block_diag(*self.A)
        self.B = block_diag(*self.B) if elementwise else np.vstack(self.B)
        self.C = block_diag(*self.C)
        self.D = block_diag(*self.D) if elementwise else np.vstack(self.D)
        # TODO: shape validation

        self._x = np.zeros(len(self.A))[:, None]

    def __call__(self, t, u):
        u = u[:, None] if self.elementwise else u[None, :]
        y = np.dot(self.C, self._x) + np.dot(self.D, u)
        self._x = np.dot(self.A, self._x) + np.dot(self.B, u)
        return self.to_vector(y)

    def to_vector(self, y):
        return y.flatten(order='C')

    def from_vector(self, x):
        return x.reshape(*self._x.shape, order='C')
