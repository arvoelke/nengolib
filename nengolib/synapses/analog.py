"""Classical analog linear filters."""

import numpy as np
import warnings
from scipy.misc import pade, factorial

from nengo.utils.compat import is_integer

from nengolib.signal.system import LinearSystem

__all__ = [
    'LinearFilter', 'Lowpass', 'Alpha', 'DoubleExp', 'Bandpass', 'Highpass',
    'PureDelay']


def LinearFilter(num, den, analog=True):
    # don't inherit from LinearSystem, otherwise we get its metaclass too
    # what we really want to have is something that returns a LinearSystem
    return LinearSystem((num, den), analog=analog)


def Lowpass(tau):
    return LinearFilter([1], [tau, 1])


def Alpha(tau):
    return LinearFilter([1], [tau**2, 2*tau, 1])


def DoubleExp(tau1, tau2):
    return LinearFilter([1], [tau1*tau2, tau1 + tau2, 1])


def Bandpass(freq, Q):
    # http://www.analog.com/library/analogDialogue/archives/43-09/EDCh%208%20filter.pdf  # noqa: E501
    w_0 = freq * (2*np.pi)
    return LinearFilter([1], [1./w_0**2, 1./(w_0*Q), 1])


def Highpass(tau, order=1):
    """Differentiated lowpass, raised to a given power."""
    if order < 1 or not is_integer(order):
        raise ValueError("order (%s) must be integer >= 1" % order)
    num, den = map(np.poly1d, ([tau, 0], [tau, 1]))
    return LinearFilter(num**order, den**order)


def _passthrough_delay(m, c):
    """Analytically derived state-space when p = q = m.

    We use this because it is numerically stable for high m.
    """
    j = np.arange(1, m+1, dtype=np.float64)
    u = (m + j) * (m - j + 1) / (c * j)

    A = np.zeros((m, m))
    B = np.zeros((m, 1))
    C = np.zeros((1, m))
    D = np.zeros((1,))

    A[0, :] = B[0, 0] = -u[0]
    A[1:, :-1][np.diag_indices(m-1)] = u[1:]
    D[0] = (-1)**m
    C[0, np.arange(m) % 2 == 0] = 2*D[0]
    return LinearSystem((A, B, C, D), analog=True)


def _proper_delay(q, c):
    """Analytically derived state-space when p = q - 1.

    We use this because it is numerically stable for high q
    and doesn't have a passthrough.
    """
    j = np.arange(q, dtype=np.float64)
    u = (q + j) * (q - j) / (c * (j + 1))

    A = np.zeros((q, q))
    B = np.zeros((q, 1))
    C = np.zeros((1, q))
    D = np.zeros((1,))

    A[0, :] = -u[0]
    B[0, 0] = u[0]
    A[1:, :-1][np.diag_indices(q-1)] = u[1:]
    C[0, :] = (j + 1) / float(q) * (-1) ** (q - 1 - j)
    return LinearSystem((A, B, C, D), analog=True)


def _pade_delay(p, q, c):
    """Numerically evaluated state-space using Pade approximants.

    This may have numerical issues for large values of p or q.
    """
    i = np.arange(1, p+q+1, dtype=np.float64)
    taylor = np.append([1.0], (-c)**i / factorial(i))
    num, den = pade(taylor, q)
    return LinearFilter(num, den)


def PureDelay(c, order, p=None):
    """Linear filter for a continuous pure delay of c seconds.

    The order is the degree of the denominator in the transfer function, also
    known as q in the Pade approximation. Parameter p is the degree of the
    numerator. If p is equal to q, then the system will have a passthrough.
    If None (default), then p = q - 1. Either of these choices will be
    significantly more accurate than other choices of p when q is large.
    """
    q = order
    if p is None:
        p = q - 1

    if order < 1 or not is_integer(order):
        raise ValueError("order (%s) must be integer >= 1" % order)
    if p < 1 or not is_integer(p):
        raise ValueError("p (%s) must be integer >= 1" % p)

    if p == q:
        return _passthrough_delay(p, c)
    elif p == q - 1:
        return _proper_delay(q, c)
    else:
        if q >= 10:
            warnings.warn("For large values of q (>= 10), p should either be "
                          "None, q - 1, or q.")
        return _pade_delay(p, q, c)
