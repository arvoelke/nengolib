"""Classical analog linear filters."""

import numpy as np
from scipy.misc import pade, factorial

from nengo.utils.compat import is_integer

from nengolib.signal.system import LinearSystem

__all__ = [
    'LinearFilter', 'Lowpass', 'Alpha', 'Bandpass', 'Highpass', 'PadeDelay']


class LinearFilter(LinearSystem):

    def __init__(self, num, den):
        # Note: we don't use self.analog because that information will get
        # lost if the instance uses operations inherited from LinearSystem
        super(LinearFilter, self).__init__((num, den))


class Lowpass(LinearFilter):

    def __init__(self, tau):
        super(Lowpass, self).__init__([1], [tau, 1])
        self.tau = tau


class Alpha(LinearFilter):

    def __init__(self, tau):
        super(Alpha, self).__init__([1], [tau**2, 2*tau, 1])
        self.tau = tau


class Bandpass(LinearFilter):
    """Q-bandpass filter."""

    def __init__(self, freq, Q):
        # http://www.analog.com/library/analogDialogue/archives/43-09/EDCh%208%20filter.pdf  # noqa: E501
        w_0 = freq * (2*np.pi)
        super(Bandpass, self).__init__([1], [1./w_0**2, 1./(w_0*Q), 1])


class Highpass(LinearFilter):
    """Differentiated lowpass, raised to a given power."""

    def __init__(self, tau, order=1):
        if order < 1 or not is_integer(order):
            raise ValueError("order (%s) must be integer >= 1" % order)
        num, den = map(np.poly1d, ([tau, 0], [tau, 1]))
        super(Highpass, self).__init__(num**order, den**order)


class PadeDelay(LinearFilter):

    def __init__(self, p, q, c=1.0):
        i = np.arange(1, p+q+1)
        taylor = np.append([1.0], (-c)**i / factorial(i))
        num, den = pade(taylor, q)
        super(PadeDelay, self).__init__(num, den)
