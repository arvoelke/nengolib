"""Synapse objects that are more efficient for higher orders."""

import numpy as np
from scipy.signal import cont2discrete

from nengo.synapses import LinearFilter as BaseLinearFilter
from nengo.synapses import Lowpass as BaseLowpass
from nengo.synapses import Alpha as BaseAlpha
from nengo.synapses import Triangle as BaseTriangle

from nengolib.signal.system import sys2ss

__all__ = ['LinearFilter', 'Lowpass', 'Alpha', 'Triangle']


class StateSpaceStep(BaseLinearFilter.Step):

    def __init__(self, ss, output):
        self.output = output
        self._A, self._B, self._C, self._D = ss
        self._x = np.zeros(len(self._A))[:, None]

    def __call__(self, signal):
        u = signal[None, :]
        self._x = np.dot(self._A, self._x) + np.dot(self._B, u)
        self.output[...] = np.dot(self._C, self._x) + np.dot(self._D, u)


class SimulatorMixin(object):

    def make_step(self, dt, output, method='zoh'):
        if len(self.den) <= 2:  # fall back to reference implementation
            # note: bug in nengo where subclasses don't pass method=method
            return super(SimulatorMixin, self).make_step(dt, output)
        A, B, C, D = sys2ss(self)
        if self.analog:
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt, method=method)
        return StateSpaceStep((A, B, C, D), output)


def mixer(cls, mixin=SimulatorMixin):
    """Creates a class that inherits from cls and adds a mixin."""
    class Mixed(mixin, cls):  # order determines method resolution (MRO)
        pass
    return Mixed


# This implementation of the LinearFilter is more efficient when the order
# is greater than 2.
LinearFilter = mixer(BaseLinearFilter)

# The following will (currently) always redirect to the underlying base
# object, since their orders are <= 2.
Lowpass = mixer(BaseLowpass)
Alpha = mixer(BaseAlpha)
Triangle = mixer(BaseTriangle)
