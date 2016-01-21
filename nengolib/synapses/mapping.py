import numpy as np
from scipy.signal import cont2discrete

from nengo.synapses import Lowpass

from nengolib.signal.system import sys2ss


def ss2sim(sys, synapse, dt=0):
    """Maps an LTI system to the synaptic dynamics in state-space."""
    A, B, C, D = sys2ss(sys)
    if not isinstance(synapse, Lowpass):
        # TODO: support other synapses
        raise TypeError("synapse (%s) must be Lowpass" % (synapse,))
    if dt == 0:
        # Analog case (principle 3)
        A = synapse.tau * A + np.eye(len(A))
        B = synapse.tau * B
    else:
        # Discretized case (generalized principle 3)
        A, B, C, D, _ = cont2discrete((A, B, C, D), dt=dt)
        a = np.exp(-dt/synapse.tau)
        A = 1./(1 - a) * (A - a * np.eye(len(A)))
        B = 1./(1 - a) * B
    return (A, B, C, D)
