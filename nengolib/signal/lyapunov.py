import numpy as np
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
from scipy.signal import cont2discrete

from nengolib.signal.system import sys2ss

__all__ = ['stateH2']


def _H2P(A, B, discrete=False):
    """Computes the positive-definite P matrix for determining the H2 norm."""
    if discrete:
        P = solve_discrete_lyapunov(A, np.dot(B, B.T))  # APA^T - P = -BB^T
        assert np.allclose(np.dot(A, np.dot(P, A.T)) - P + np.dot(B, B.T), 0)
    else:
        P = solve_lyapunov(A, -np.dot(B, B.T))  # AP + PA^T = -BB^T
        assert np.allclose(np.dot(A, P) + np.dot(P, A.T) + np.dot(B, B.T), 0)
    return P


def stateH2(sys, dt=None):
    """Returns the H2-norm of each component of x in the state-space.

    This gives the power of each component of x in response to white-noise
    input with uniform power.

    Reference:
        http://www.maplesoft.com/support/help/maple/view.aspx?path=DynamicSystems%2FNormH2  # noqa: E501
    """
    # TODO: accept an additional sys describing the filtering on the input
    # so that we can get the norm in response to different input spectra.
    A, B, C, D = sys2ss(sys)
    if dt is None:
        P = _H2P(A, B, discrete=False)
    else:
        A, B, C, D, dt = cont2discrete((A, B, C, D), dt)
        P = _H2P(A, B, discrete=True)
    return np.sqrt(P[np.diag_indices(len(P))])
