import numpy as np
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov

from nengolib.signal.system import sys2ss

__all__ = ['state_norm']


def _H2P(A, B, analog):
    """Computes the positive-definite P matrix for determining the H2 norm."""
    if analog:
        P = solve_lyapunov(A, -np.dot(B, B.T))  # AP + PA^T = -BB^T
    else:
        # Note: discretization is not performed for the user
        P = solve_discrete_lyapunov(A, np.dot(B, B.T))  # APA^T - P = -BB^T
    return P


def state_norm(sys, analog=True, method='H2'):
    """Returns the H2-norm of each component of x in the state-space.

    This gives the power of each component of x in response to white-noise
    input with uniform power, or equivalently the power of each component of
    x in response to a delta impulse.

    Reference:
        http://www.maplesoft.com/support/help/maple/view.aspx?path=DynamicSystems%2FNormH2  # noqa: E501
    """
    if method != 'H2':
        raise ValueError("Only method=='H2' is currently supported")
    # TODO: accept an additional sys describing the filtering on the input
    # so that we can get the norm in response to different input spectra.
    A, B, C, D = sys2ss(sys)
    P = _H2P(A, B, analog)
    return np.sqrt(P[np.diag_indices(len(P))])
