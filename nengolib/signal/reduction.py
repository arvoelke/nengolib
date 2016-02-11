import warnings

import numpy as np

from scipy.linalg import cholesky, svd, inv

from nengolib.signal.lyapunov import control_gram, observe_gram
from nengolib.signal.system import sys2zpk, sys2ss, LinearSystem

__all__ = ['minreal', 'similarity_transform', 'balreal', 'modred', 'balred']

# TODO: reference linear_model_reduction.ipynb in auto-generated docs


def minreal(sys, tol=1e-8):
    """Pole/zero cancellation within a given tolerance.

    References:
        http://www.mathworks.com/help/control/ref/minreal.html
    """
    z, p, k = sys2zpk(sys)
    mz = np.ones(len(z), dtype=bool)  # start with all zeros
    mp = np.zeros(len(p), dtype=bool)  # and no poles
    for i, pole in enumerate(p):
        # search among the remaining zeros
        bad = np.where((np.abs(pole - z) <= tol) & mz)[0]
        if len(bad):  # cancel this pole with one of the zeros
            mz[bad[0]] = False
        else:  # include this pole
            mp[i] = True
    return LinearSystem((z[mz], p[mp], k))


def similarity_transform(A, B, C, D, T, Tinv=None):
    """Changes basis of state-space (A, B, C, D) to T."""
    if Tinv is None:
        Tinv = inv(T)
    TA = np.dot(Tinv, np.dot(A, T))
    TB = np.dot(Tinv, B)
    TC = np.dot(C, T)
    TD = D
    return (TA, TB, TC, TD)


def balreal(sys):
    """Computes the balanced realization of sys and returns its eigenvalues.

    References:
        [1] http://www.mathworks.com/help/control/ref/balreal.html

        [2] Laub, A.J., M.T. Heath, C.C. Paige, and R.C. Ward, "Computation of
            System Balancing Transformations and Other Applications of
            Simultaneous Diagonalization Algorithms," *IEEE Trans. Automatic
            Control*, AC-32 (1987), pp. 115-122.
    """
    sys = LinearSystem(sys)  # cast first to memoize sys2ss

    R = control_gram(sys)
    O = observe_gram(sys)

    LR = cholesky(R, lower=True)
    LO = cholesky(O, lower=True)

    U, S, V = svd(np.dot(LO.T, LR))

    T = np.dot(LR, V.T) * S ** (-1. / 2)
    Tinv = (S ** (-1. / 2))[:, None] * np.dot(U.T, LO.T)

    A, B, C, D = sys2ss(sys)
    TA, TB, TC, TD = similarity_transform(A, B, C, D, T, Tinv)

    return LinearSystem((TA, TB, TC, TD)), S


def modred(sys, keep_states, method='del'):
    """Reduces model order by eliminating states.

    References:
        http://www.mathworks.com/help/control/ref/modred.html
    """
    A, B, C, D = sys2ss(sys)
    mask = np.zeros(len(A), dtype=bool)
    mask[np.asarray(keep_states)] = True

    grm = np.where(mask)[0]
    brm = np.where(~mask)[0]
    glm = grm[:, None]
    blm = brm[:, None]

    A11 = A[glm, grm]
    A12 = A[glm, brm]
    A21 = A[blm, grm]
    A22 = A[blm, brm]
    B1 = B[mask, :]
    B2 = B[~mask, :]
    C1 = C[:, mask]
    C2 = C[:, ~mask]

    if method == 'del':
        RA = A11
        RB = B1
        RC = C1
        RD = D

    elif method == 'dc':
        A22I = inv(A22)
        RA = A11 - np.dot(A12, np.dot(A22I, A21))
        RB = B1 - np.dot(A12, np.dot(A22I, B2))
        RC = C1 - np.dot(C2, np.dot(A22I, A21))
        RD = D - np.dot(C2, np.dot(A22I, B2))
        # TODO: for discrete case, simply replace (-A22I) with inv(I - A22)

    else:
        raise ValueError("invalid method: '%s'" % (method,))

    return LinearSystem((RA, RB, RC, RD))


def balred(sys, order, method='del'):
    """Reduces a LinearSystem to given order using balreal and modred."""
    sys, s = balreal(sys)
    if order < 1:
        raise ValueError("Invalid order (%s), must be at least 1" % (order,))
    if order >= len(sys):
        warnings.warn("Model is already of given order")
        return sys
    keep_states = np.arange(order)  # keep largest eigenvalues
    return modred(sys, keep_states, method)
