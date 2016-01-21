import numpy as np
from scipy.signal import cont2discrete, zpk2ss, tf2ss, ss2tf, zpk2tf, lfilter

from nengo.synapses import Synapse

__all__ = ['sys2ss', 'sys2tf', 'tfmul', 'impulse', 'is_exp_stable', 'ss_radii']


def sys2ss(sys):
    """Converts an LTI system in any form to state-space."""
    if isinstance(sys, Synapse):
        return tf2ss(sys.num, sys.den)
    elif len(sys) == 2:
        return tf2ss(*sys)
    elif len(sys) == 3:
        return zpk2ss(*sys)
    elif len(sys) == 4:
        return map(np.array, sys)
    else:
        raise ValueError(
            "sys must be an instance of Synapse, or a tuple of 2 (tf), "
            "3 (zpk), or 4 (ss) arrays.")


def sys2tf(sys):
    """Converts an LTI system in any form to a transfer function."""
    if isinstance(sys, Synapse):
        return (sys.num, sys.den)
    elif len(sys) == 2:
        return map(np.array, sys)
    elif len(sys) == 3:
        return zpk2tf(*sys)
    elif len(sys) == 4:
        return ss2tf(*sys)
    else:
        raise ValueError(
            "sys must be an instance of Synapse, or a tuple of 2 (tf), "
            "3 (zpk), or 4 (ss) arrays.")


def tfmul(sys1, sys2):
    """Multiplies together the transfer functions for sys1 and sys2."""
    # This represents the convolution of systems 1 and 2
    p1, q1 = sys2tf(sys1)
    p2, q2 = sys2tf(sys2)
    return (np.polymul(p1, p2), np.polymul(q1, q2))


def impulse(sys, dt, length, discretized=False):
    """Simulates sys on a delta impulse for length timesteps of width dt."""
    tf = sys2tf(sys)
    if discretized:
        num, den = tf
    else:
        (num,), den, _ = cont2discrete(tf, dt)
    impulse = np.zeros(length)
    impulse[0] = 1
    return lfilter(num, den, impulse)


def is_exp_stable(A):
    """Returns true iff system is exponentially stable."""
    w, v = np.linalg.eig(A)
    return (w.real < 0).all()  # <=> e^(At) goes to 0


def ss_radii(A, B, C, D, radii=1.0):
    """Scales the system to compensate for radii of the state."""
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim > 1:
        raise ValueError("radii (%s) must be a 1-dim array or scalar" % (
            radii,))
    elif r.ndim == 0:
        r = np.ones(len(A)) * r
    A = A / r[:, None] * r
    B = B / r[:, None]
    C = C * r
    return A, B, C, D
