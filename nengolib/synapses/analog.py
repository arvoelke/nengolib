"""Classical analog linear filters."""

import numpy as np

from nengo import LinearFilter
from nengo.utils.compat import is_number


def bandpass(freq, Q):
    """Q-bandpass filter."""
    # http://www.analog.com/library/analogDialogue/archives/43-09/EDCh%208%20filter.pdf  # noqa: E501
    w_0 = freq * (2*np.pi)
    return LinearFilter([1], [1./w_0**2, 1./(w_0*Q), 1])


def notch(freq, Q):
    """Notch (band-reject) filter."""
    # http://www.analog.com/library/analogDialogue/archives/43-09/EDCh%208%20filter.pdf  # noqa: E501
    w_0 = freq * (2*np.pi)
    return LinearFilter([1./w_0**2, 0, 1], [1./w_0**2, 1./(w_0*Q), 1])


def highpass(tau, order=1):
    """Differentiated lowpass, raised to a given power."""
    if order < 1 or not is_number(order):
        raise ValueError("order (%s) must be integer >= 1" % order)
    num, den = map(np.poly1d, ([tau, 0], [tau, 1]))
    return LinearFilter(num**order, den**order)
