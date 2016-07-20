import numpy as np
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov, eig
from scipy.optimize import fminbound

from nengolib.signal.discrete import impulse, cont2discrete
from nengolib.signal.system import LinearSystem


__all__ = ['state_norm', 'control_gram', 'observe_gram', 'l1_norm']


def _H2P(A, B, analog):
    """Computes the positive-definite P matrix for determining the H2-norm."""
    if analog:
        P = solve_lyapunov(A, -np.dot(B, B.T))  # AP + PA^T = -BB^T
    else:
        # Note: discretization is not performed for the user
        P = solve_discrete_lyapunov(A, np.dot(B, B.T))  # APA^T - P = -BB^T
    return P


def state_norm(sys, norm='H2'):
    """Returns the norm of each component of x in the state-space.

    The H2-norm gives the power of each component of x in response to
    white-noise input with uniform power, or equivalently the total power of
    each component of x in response to a delta impulse.

    Reference:
        http://www.maplesoft.com/support/help/maple/view.aspx?path=DynamicSystems%2FNormH2  # noqa: E501
    """
    if norm == 'H2':
        # TODO: accept an additional sys describing the filtering on the input
        # so that we can get the norm in response to different input spectra.
        sys = LinearSystem(sys)
        A, B, C, D = sys.ss
        P = _H2P(A, B, sys.analog)
        return np.sqrt(P[np.diag_indices(len(P))])
    else:
        raise NotImplementedError("norm must be one of: H2")


def control_gram(sys):
    """Computes the controllability/reachability gramiam of a linear system.

    Reference:
        https://en.wikibooks.org/wiki/Control_Systems/Controllability_and_Observability  # noqa: E501
    """
    sys = LinearSystem(sys)
    A, B, C, D = sys.ss
    return _H2P(A, B, sys.analog)


def observe_gram(sys):
    """Computes the observability gramiam of a linear system.

    Reference:
        https://en.wikibooks.org/wiki/Control_Systems/Controllability_and_Observability  # noqa: E501
    """
    sys = LinearSystem(sys)
    A, B, C, D = sys.ss
    return _H2P(A.T, C.T, sys.analog)


def _state_impulse(A, x0, k, delay=0):
    """Computes the states for discrete feedback A starting from x0."""
    x = np.zeros((k, len(A)))
    x[delay, :] = np.squeeze(x0)
    for i in range(delay+1, k):
        x[i, :] = np.dot(A, x[i-1, :])  # note: doesn't assume canonical form
    return x


def l1_norm(sys, rtol=1e-6, max_length=2**18):
    """Returns the L1-norm of a linear system within a relative tolerance.

    The L1-norm of a (BIBO stable) linear system is the integral of the
    absolute value of its impulse response. For unstable systems this will be
    infinite. The L1-norm is important because it bounds the worst-case
    output of the system for arbitrary inputs within [-1, 1]. In fact,
    this worst-case output is achieved by reversing the input which alternates
    between -1 and 1 during the intervals where the impulse response is
    negative or positive, respectively (in the limit as T -> infinity).

    Algorithm adapted from [1]_ following the methods of [2]_. This works by
    iteratively refining lower and upper bounds using progressively longer
    simulations and smaller timesteps. The lower bound is given by the
    absolute values of the discretized response. The upper bound is given by
    refining the time-step intervals where zero-crossings may have occurred.

    References:
        [1] http://www.mathworks.com/matlabcentral/fileexchange/41587-system-l1-norm/content/l1norm.m  # noqa: E501
            J.F. Whidborne (April 28, 1995).

        [2] Rutland, Neil K., and Paul G. Lane. "Computing the 1-norm of the
            impulse response of linear time-invariant systems."
            Systems & control letters 26.3 (1995): 211-221.
    """
    sys = LinearSystem(sys)
    if not sys.analog:
        raise ValueError("system (%s) must be analog" % sys)

    # Setup state-space system and check stability/conditioning
    A, B, C, D = sys.ss
    alpha = np.max(eig(A)[0].real)  # eq (28)
    if alpha >= 0:
        raise ValueError("system (%s) has unstable eigenvalue: %s" % (
            sys, alpha))

    # Compute a suitable lower-bound for the L1-norm
    # using the steady state response, which is equivalent to the
    # L1-norm without an absolute value (i.e. just an integral)
    G0 = sys.dcgain - sys.D  # -C.inv(A).B

    # Compute a suitable upper-bound for the L1-norm
    # Note this should be tighter than 2*sum(abs(hankel(sys)))
    def _normtail(sig, A, x, C):
        # observability gramiam when A perturbed by sig
        W = solve_lyapunov(A.T + sig*np.eye(len(A)), -C.T.dot(C))
        return np.sqrt(x.dot(W).dot(x.T) / 2 / sig)  # eq (39)

    xtol = -alpha * 1e-4
    _, fopt, _, _ = fminbound(
        _normtail, 0, -alpha, (A, B.T, C), xtol=xtol, full_output=True)

    # Setup parameters for iterative optimization
    L, U = abs(G0), fopt
    N = 2**4
    T = -1 / alpha

    while (N <= max_length and
           (np.allclose(L, 0) or .5 * (U - L) / L >= rtol)):  # eq (25)

        # Step 1. Improve the lower bound by simulating more.
        dt = T / N
        dsys = cont2discrete(sys, dt=dt)
        Phi = dsys.A

        y = impulse(dsys, dt=None, length=N)
        abs_y = abs(y[1:])
        L_impulse = np.sum(abs_y)

        # bound the missing response from t > T from below
        L_tail = abs(G0 - np.sum(y))  # eq (33)
        L = max(L, L_impulse + L_tail)

        # Step 2. Improve the upper bound using refined interval method.
        x = _state_impulse(Phi, x0=B, k=N, delay=0)  # eq (38)
        abs_e = np.squeeze(abs(C.dot(x.T)))
        x = x[:-1]  # compensate for computing where thresholds crossed

        # find intervals that could have zero-crossings and adjust their
        # upper bounds (the lower bound is exact for the other intervals)
        CTC = C.T.dot(C)
        W = solve_lyapunov(A.T, Phi.T.dot(CTC).dot(Phi) - CTC)  # eq (36)
        AWA = A.T.dot(W.dot(A))
        thresh = np.squeeze(  # eq (41)
            np.sqrt(dt * np.sum(x.dot(AWA) * x, axis=1)))
        cross = np.maximum(abs_e[:-1], abs_e[1:]) <= thresh  # eq (20)
        abs_y[cross] = np.sqrt(  # eq (22, 37)
            dt * np.sum(x[cross].dot(W) * x[cross], axis=1))

        # bound the missing response from t > T from above
        _, U_tail, _, _ = fminbound(
            _normtail, 0, -alpha, (A, x[-1], C), xtol=xtol, full_output=True)
        U_impulse = np.sum(abs_y)
        U = max(min(U, U_impulse + U_tail), L)

        N *= 2
        if U_impulse - L_impulse < U_tail - L_tail:  # eq (26)
            T *= 2

    return (U + L) / 2, .5 * (U - L) / L
