import numpy as np
from scipy.linalg import block_diag
from scipy.signal import cont2discrete, tf2ss

from nengo.synapses import Synapse

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

    def __init__(self, synapses, dt, elementwise=False, method='zoh'):
        if isinstance(synapses, Synapse):
            synapses = [synapses]
        self.synapses = synapses
        self.dt = dt
        self.elementwise = elementwise

        self.A = []
        self.B = []
        self.C = []
        self.D = []
        for synapse in synapses:
            A, B, C, D, _ = cont2discrete(
                tf2ss(synapse.num, synapse.den), dt, method=method)
            self.A.append(A)
            self.B.append(B)
            self.C.append(C)
            self.D.append(D)

        self.A = block_diag(*self.A)
        self.B = block_diag(*self.B) if elementwise else np.vstack(self.B)
        self.C = block_diag(*self.C)
        self.D = block_diag(*self.D) if elementwise else np.vstack(self.D)

        self._x = None  # set by __call__ once size of u is known

    def __call__(self, t, u):
        # TODO: shape validation
        u = u[:, None] if self.elementwise else u[None, :]
        if self._x is None:
            self._x = np.zeros((len(self.A), u.shape[1]))
        y = np.dot(self.C, self._x) + np.dot(self.D, u)
        self._x = np.dot(self.A, self._x) + np.dot(self.B, u)
        return self.to_vector(y)

    def to_vector(self, y):
        return y.flatten(order='F')

    def from_vector(self, x):
        return x.reshape(*self._x.shape, order='F')
