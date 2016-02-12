import numpy as np
from scipy.signal import cont2discrete

from nengolib.signal.system import sys2ss, LinearSystem

__all__ = ['ss2sim']


def ss2sim(sys, synapse, dt=None):
    """Maps an LTI system to the synaptic dynamics in state-space."""
    A, B, C, D = sys2ss(sys)
    synapse = LinearSystem(synapse)
    if len(synapse) != 1 or not synapse.proper:
        raise TypeError("synapse (%s) must be first-order and proper" % (
            synapse,))

    # TODO: make sure synapse is analog, or handle the discrete synapse case
    # TODO: put derivations into a notebook

    a, = synapse.num
    b1, b2 = synapse.den
    if np.allclose(b2, 0):  # scaled integrator
        # put system into form: gain / s, and handle gain at the end
        gain = a / b1
        if dt is not None:
            # discretized integrator is dt / (z - 1)
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt=dt)
            A = 1./dt * (A - np.eye(len(A)))
            B = 1./dt * B

    else:  # scaled lowpass
        # put system into form: gain / (tau*s + 1), and handle gain at the end
        gain, tau = a / b2, b1 / b2  # divide both polynomials by b2

        if dt is None:
            # Analog case (normal principle 3)
            A = tau * A + np.eye(len(A))
            B = tau * B
        else:
            # Discretized case (derived from generalized principle 3)
            # discretized lowpass is (1 - a) / (z - a)
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt=dt)
            a = np.exp(-dt/tau)
            A = 1./(1 - a) * (A - a * np.eye(len(A)))
            B = 1./(1 - a) * B

    return (A / gain, B / gain, C, D)
