import warnings

import numpy as np

from scipy.linalg import inv

from nengolib.signal.lyapunov import balanced_transformation
from nengolib.signal.system import sys2zpk, LinearSystem

__all__ = ['pole_zero_cancel', 'modred', 'balred']

# TODO: reference linear_model_reduction.ipynb in auto-generated docs


def pole_zero_cancel(sys, tol=1e-8):
    """Pole/zero cancellation within a given tolerance.

    Sometimes referred to as the minimal realization in state-space.

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
    return LinearSystem((z[mz], p[mp], k), analog=sys.analog)


def modred(sys, keep_states, method='del'):
    """Reduces model order by eliminating states.

    References:
        http://www.mathworks.com/help/control/ref/modred.html
    """
    sys = LinearSystem(sys)
    A, B, C, D = sys.ss
    if not sys.analog:
        raise NotImplementedError("model reduction of digital filters not "
                                  "supported")

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

    return LinearSystem((RA, RB, RC, RD), analog=sys.analog)


def balred(sys, order, method='del'):
    """Reduces a LinearSystem to given order using balreal and modred."""
    sys = LinearSystem(sys)
    if order < 1:
        raise ValueError("Invalid order (%s), must be at least 1" % (order,))
    if order >= len(sys):
        warnings.warn("Model is already of given order")
        return sys
    T, Tinv, _ = balanced_transformation(sys)
    sys = sys.transform(T, Tinv=Tinv)
    keep_states = np.arange(order)  # keep largest eigenvalues
    return modred(sys, keep_states, method)
