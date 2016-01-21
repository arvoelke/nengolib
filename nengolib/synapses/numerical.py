import numpy as np

from scipy.misc import pade, factorial

__all__ = ['pure_delay']


def pure_delay(p, q, c=1.0):
    """Returns the transfer function F(s) ~ e^(-sc) of order [p/q]."""
    i = np.arange(1, p+q+1)
    taylor = np.append([1.0], (-c)**i / factorial(i))
    return pade(taylor, q)
