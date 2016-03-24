"""Classical analog linear filters."""

import numpy as np
from scipy.misc import pade, factorial

from nengo.utils.compat import is_integer

from nengolib.signal.system import LinearSystem

__all__ = [
    'LinearFilter', 'Lowpass', 'Alpha', 'DoubleExp', 'Bandpass', 'Highpass',
    'PadeDelay']


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


def PadeDelay(p, q, c=1.0):
    i = np.arange(1, p+q+1)
    taylor = np.append([1.0], (-c)**i / factorial(i))
    num, den = pade(taylor, q)
    return LinearFilter(num, den)
