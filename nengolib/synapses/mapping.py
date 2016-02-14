import numpy as np

from nengolib.signal.system import LinearSystem
from nengolib.signal.discrete import cont2discrete

__all__ = ['ss2sim']


def ss2sim(sys, synapse, dt=None):
    """Maps an LTI system to the synaptic dynamics in state-space."""
    synapse = LinearSystem(synapse)
    if len(synapse) != 1 or not synapse.proper or not synapse.analog:
        raise ValueError("synapse (%s) must be first-order, proper, and "
                         "analog" % synapse)

    sys = LinearSystem(sys)
    if not sys.analog:
        raise ValueError("system (%s) must be analog" % sys)
    A, B, C, D = sys.ss

    # TODO: put derivations into a notebook
    a, = synapse.num
    b1, b2 = synapse.den
    if np.allclose(b2, 0):  # scaled integrator
        # put synapse into form: gain / s, and handle gain at the end
        gain = a / b1
        if dt is not None:
            # discretized integrator is dt / (z - 1)
            A, B, C, D = cont2discrete(sys, dt=dt).ss
            A = 1./dt * (A - np.eye(len(A)))
            B = 1./dt * B

    else:  # scaled lowpass
        # put synapse into form: gain / (tau*s + 1), and handle gain at the end
        gain, tau = a / b2, b1 / b2  # divide both polynomials by b2

        if dt is None:
            # Analog case (normal principle 3)
            A = tau * A + np.eye(len(A))
            B = tau * B
        else:
            # Discretized case (derived from generalized principle 3)
            # discretized lowpass is (1 - a) / (z - a)
            A, B, C, D = cont2discrete(sys, dt=dt).ss
            a = np.exp(-dt/tau)
            A = 1./(1 - a) * (A - a * np.eye(len(A)))
            B = 1./(1 - a) * B

    return (A / gain, B / gain, C, D)
