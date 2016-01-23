import numpy as np
from scipy.signal import (
    cont2discrete, zpk2ss, ss2tf, tf2ss, zpk2tf, lfilter, normalize)

from nengo.synapses import Synapse

__all__ = [
    'sys2ss', 'sys2tf', 'tfmul', 'apply_filter', 'impulse', 'is_exp_stable',
    'scale_state']


def _raise_invalid_sys():
    raise ValueError(
        "sys must be an instance of Synapse, or a tuple of 2 (tf), "
        "3 (zpk), or 4 (ss) arrays.")


def sys2ss(sys):
    """Converts an LTI system in any form to state-space."""
    if isinstance(sys, Synapse):
        return tf2ss(sys.num, sys.den)
    elif len(sys) == 2:
        return tf2ss(*sys)
    elif len(sys) == 3:
        return zpk2ss(*sys)
    elif len(sys) == 4:
        return tuple(map(np.array, sys))
    else:
        _raise_invalid_sys()


def sys2tf(sys):
    """Converts an LTI system in any form to a transfer function."""
    if isinstance(sys, Synapse):
        return (sys.num, sys.den)
    elif len(sys) == 2:
        return tuple(map(np.array, sys))
    elif len(sys) == 3:
        return zpk2tf(*sys)
    elif len(sys) == 4:
        # TODO: properly handle SIMO systems
        # https://github.com/scipy/scipy/issues/5753
        (num,), den = ss2tf(*sys)
        return (num, den)
    else:
        _raise_invalid_sys()


def check_sys_equal(sys1, sys2):
    """Returns true iff sys1 and sys2 have the same transfer functions."""
    tf1 = normalize(*sys2tf(sys1))
    tf2 = normalize(*sys2tf(sys2))
    for t1, t2 in zip(tf1, tf2):
        if not np.allclose(t1, t2):
            return False
    return True


def tfmul(sys1, sys2):
    """Multiplies together the transfer functions for sys1 and sys2."""
    # This represents the convolution of systems 1 and 2
    p1, q1 = sys2tf(sys1)
    p2, q2 = sys2tf(sys2)
    return (np.polymul(p1, p2), np.polymul(q1, q2))


def apply_filter(u, sys, dt, axis=-1):
    """Simulates sys on u for length timesteps of width dt."""
    # TODO: properly handle SIMO systems
    # https://github.com/scipy/scipy/issues/5753
    num, den = sys2tf(sys)
    if dt is not None:
        (num,), den, _ = cont2discrete((num, den), dt)
    return lfilter(num, den, u, axis)


def impulse(sys, dt, length, axis=-1):
    """Simulates sys on a delta impulse for length timesteps of width dt."""
    impulse = np.zeros(length)
    impulse[0] = 1
    return apply_filter(impulse, sys, dt, axis)


def _is_exp_stable(A):
    w, v = np.linalg.eig(A)
    return (w.real < 0).all()  # <=> e^(At) goes to 0


def is_exp_stable(sys):
    """Returns true iff system is exponentially stable."""
    return _is_exp_stable(sys2ss(sys)[0])


def scale_state(A, B, C, D, radii=1.0):
    """Scales the system to compensate for radii of the state."""
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim > 1:
        raise ValueError("radii (%s) must be a 1-dim array or scalar" % (
            radii,))
    elif r.ndim == 0:
        r = np.ones(len(A)) * r
    elif len(r) != len(A):
        raise ValueError("radii (%s) length must match state dimension %d" % (
            radii, len(A)))
    A = A / r[:, None] * r
    B = B / r[:, None]
    C = C * r
    return A, B, C, D
