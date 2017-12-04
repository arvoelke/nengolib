import numpy as np
try:
    from scipy.linalg import solve_lyapunov as solve_continuous_lyapunov
except ImportError:  # pragma: no cover; github.com/scipy/scipy/pull/8082
    from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import solve_discrete_lyapunov, eig, cholesky, svd
from scipy.optimize import fminbound

from nengolib.signal.discrete import cont2discrete
from nengolib.signal.system import LinearSystem


__all__ = ['state_norm', 'control_gram', 'observe_gram',
           'balanced_transformation', 'hsvd', 'l1_norm']


def _H2P(A, B, analog):
    """Computes the positive-definite P matrix for determining the H2-norm."""
    if analog:
        P = solve_continuous_lyapunov(A, -np.dot(B, B.T))  # AP + PA^T = -BB^T
    else:
        # Note: discretization is not performed for the user
        P = solve_discrete_lyapunov(A, np.dot(B, B.T))  # APA^T - P = -BB^T
    return P


def state_norm(sys, norm='H2'):
    """Computes the norm of each dimension of ``x`` in the state-space.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    norm : ``string``, optional
       Defaults to ``'H2'``. Must be one of:

       * ``'H2'`` : The power of each dimension in response to white-noise
         input with uniform power, or equivalently in response to a delta
         impulse. [#]_

    Returns
    -------
    ``(len(sys),) np.array``
       The norm of each dimension in ``sys``.

    See Also
    --------
    :class:`.H2Norm`
    :func:`.l1_norm`

    References
    ----------
    .. [#] http://www.maplesoft.com/support/help/maple/view.aspx?path=DynamicSystems%2FNormH2

    Examples
    --------
    >>> from nengolib.signal import state_norm
    >>> from nengolib.synapses import PadeDelay
    >>> sys = PadeDelay(.1, order=4)
    >>> dt = 1e-4
    >>> y = sys.X.impulse(2500, dt=dt)

    Comparing the analytical H2-norm of the delay state to its simulated value:

    >>> assert np.allclose(np.linalg.norm(y, axis=0) * np.sqrt(dt),
    >>>                    state_norm(sys), atol=1e-4)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sys.ntrange(len(y), dt=dt), y)
    >>> plt.xlabel("Time (s)")
    >>> plt.show()
    """  # noqa: E501

    if norm == 'H2':
        # TODO: accept an additional sys describing the filtering on the input
        # so that we can get the norm in response to different input spectra.
        sys = LinearSystem(sys)
        A, B, C, D = sys.ss
        P = _H2P(A, B, sys.analog)
        return np.sqrt(P[np.diag_indices(len(P))])
    else:
        raise NotImplementedError("norm (%s) must be one of: 'H2'" % (norm,))


def control_gram(sys):
    """Computes the controllability/reachability gramiam of a linear system.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.

    Returns
    -------
    ``(len(sys), len(sys)) np.array``
       The controllability/reachability gramiam matrix.

    See Also
    --------
    :func:`.observe_gram`
    :func:`.balanced_transformation`

    References
    ----------
    .. [#] https://en.wikibooks.org/wiki/Control_Systems/Controllability_and_Observability
    """  # noqa: E501

    sys = LinearSystem(sys)
    A, B, C, D = sys.ss
    return _H2P(A, B, sys.analog)


def observe_gram(sys):
    """Computes the observability gramiam of a linear system.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.

    Returns
    -------
    ``(len(sys), len(sys)) np.array``
       The observability gramiam matrix.

    See Also
    --------
    :func:`.control_gram`
    :func:`.balanced_transformation`

    References
    ----------
    .. [#] https://en.wikibooks.org/wiki/Control_Systems/Controllability_and_Observability
    """  # noqa: E501

    sys = LinearSystem(sys)
    A, B, C, D = sys.ss
    return _H2P(A.T, C.T, sys.analog)


def balanced_transformation(sys):
    """Computes the balancing transformation, its inverse, and eigenvalues.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.

    Returns
    -------
    ``(len(sys), len(sys)) np.array``
       Similarity transformation matrix.
    ``(len(sys), len(sys)) np.array``
       Inverse of similarity transformation matrix.
    ``(len(sys),) np.array``
       Hankel singular values (see :func:`.hsvd`).

    See Also
    --------
    :func:`.balance`
    :class:`.Balanced`
    :func:`.hsvd`
    :func:`.LinearSystem.transform`

    References
    ----------
    .. [#] Laub, A.J., M.T. Heath, C.C. Paige, and R.C. Ward, "Computation of
       System Balancing Transformations and Other Applications of
       Simultaneous Diagonalization Algorithms," *IEEE Trans. Automatic
       Control*, AC-32 (1987), pp. 115-122.

    .. [#] http://www.mathworks.com/help/control/ref/balreal.html
    """

    sys = LinearSystem(sys)  # cast first to memoize sys2ss in control_gram
    if not sys.analog:
        raise NotImplementedError("balanced digital filters not supported")

    R = control_gram(sys)
    O = observe_gram(sys)

    LR = cholesky(R, lower=True)
    LO = cholesky(O, lower=True)

    U, S, V = svd(np.dot(LO.T, LR))

    T = np.dot(LR, V.T) * S ** (-1. / 2)
    Tinv = (S ** (-1. / 2))[:, None] * np.dot(U.T, LO.T)

    return T, Tinv, S


def hsvd(sys):
    """Compute Hankel singular values of a linear system.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.

    Returns
    -------
    ``(len(sys),) np.array``
       Hankel singular values.

    See Also
    --------
    :class:`.Hankel`
    :func:`.balanced_transformation`

    References
    ----------
    .. [#] Glover, Keith, and Jonathan R. Partington. "Bounds on the
       achievable accuracy in model reduction." Modelling, robustness and
       sensitivity reduction in control systems. Springer Berlin
       Heidelberg, 1987. 95-118.

    .. [#] https://www.mathworks.com/help/control/ref/hsvd.html
    """

    sys = LinearSystem(sys)
    R, O = control_gram(sys), observe_gram(sys)
    # sort needed to be consistent across different versions
    return np.sort(np.sqrt(abs(eig(np.dot(O, R))[0])))[::-1]


def _state_impulse(A, x0, k, delay=0):
    """Computes the states for discrete feedback A starting from x0."""
    x = np.zeros((k, len(A)))
    x[delay, :] = np.squeeze(x0)
    for i in range(delay+1, k):
        x[i, :] = np.dot(A, x[i-1, :])  # Note: doesn't assume canonical form
    return x


def l1_norm(sys, rtol=1e-6, max_length=2**18):
    """Computes the L1-norm of a linear system within a relative tolerance.

    The L1-norm of a (BIBO stable) linear system is the integral of the
    absolute value of its impulse response. For unstable systems this will be
    infinite. The L1-norm is important because it bounds the worst-case
    output of the system for arbitrary inputs within ``[-1, 1]``. In fact,
    this worst-case output is achieved by reversing the input which alternates
    between ``-1`` and ``1`` during the intervals where the impulse response is
    negative or positive, respectively (in the limit as ``T -> infinity``).

    Algorithm adapted from [#]_ following the methods of [#]_. This works by
    iteratively refining lower and upper bounds using progressively longer
    simulations and smaller timesteps. The lower bound is given by the
    absolute values of the discretized response. The upper bound is given by
    refining the time-step intervals where zero-crossings may have occurred.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    rtol : ``float``, optional
       Desired error (relative tolerance).
       Smaller tolerances require more compute time.
       Defaults to ``1e-6``.
    max_length : ``integer``, optional
       Maximum number of time-steps to simulate the system's impulse response.
       The simulation time-step is varied by the algorithm.
       Defaults to ``2**18``.

    Returns
    -------
    ``float``
       L1-norm of the output.

    See Also
    --------
    :class:`.L1Norm`
    :func:`.state_norm`

    Notes
    -----
    The algorithm will terminate after either ``rtol`` tolerance is met, or
    ``max_length`` simulation steps are required --- whichever occurs first.

    References
    ----------
    .. [#] http://www.mathworks.com/matlabcentral/fileexchange/41587-system-l1-norm/content/l1norm.m
       J.F. Whidborne (April 28, 1995).

    .. [#] Rutland, Neil K., and Paul G. Lane. "Computing the 1-norm of the
       impulse response of linear time-invariant systems."
       Systems & control letters 26.3 (1995): 211-221.
    """  # noqa: E501

    sys = LinearSystem(sys)
    if not sys.analog:
        raise ValueError("system (%s) must be analog" % sys)

    # Setup state-space system and check stability/conditioning
    # we will subtract out D and add it back in at the end
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
        W = solve_continuous_lyapunov(A.T + sig*np.eye(len(A)), -C.T.dot(C))
        return np.sqrt(x.dot(W).dot(x.T) / 2 / sig)  # eq (39)

    xtol = -alpha * 1e-4
    _, fopt, _, _ = fminbound(
        _normtail, 0, -alpha, (A, B.T, C), xtol=xtol, full_output=True)

    # Setup parameters for iterative optimization
    L, U = abs(G0), fopt
    N = 2**4
    T = -1 / alpha

    while N <= max_length and .5 * (U - L) / L >= rtol:  # eq (25)

        # Step 1. Improve the lower bound by simulating more.
        dt = T / N
        dsys = cont2discrete((A, B, C, 0), dt=dt)
        Phi = dsys.A

        y = dsys.impulse(N)
        abs_y = abs(y)
        L_impulse = np.sum(abs_y)

        # bound the missing response from t > T from below
        L_tail = abs(G0 - np.sum(y))  # eq (33)
        L = max(L, L_impulse + L_tail)

        # Step 2. Improve the upper bound using refined interval method.
        x = _state_impulse(Phi, x0=B, k=N, delay=0)  # eq (38)
        abs_e = np.squeeze(abs(C.dot(x.T)))
        x = x[:-1]  # compensate for computing where thresholds crossed
        abs_y = abs_y[:-1]  # compensate for computing where thresholds crossed

        # find intervals that could have zero-crossings and adjust their
        # upper bounds (the lower bound is exact for the other intervals)
        CTC = C.T.dot(C)
        W = solve_continuous_lyapunov(
            A.T, Phi.T.dot(CTC).dot(Phi) - CTC)  # eq (36)
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

    return (U + L) / 2 + abs(np.asarray(D).squeeze()), .5 * (U - L) / L
