import numpy as np
from scipy import linalg
from scipy.signal import cont2discrete as _cont2discrete

from nengolib.signal.system import LinearSystem

__all__ = ['cont2discrete', 'discrete2cont']


def cont2discrete(sys, dt, method='zoh', alpha=None):
    """Convert linear system from continuous to discrete time-domain.

    This is a wrapper around :func:`scipy.signal.cont2discrete`, with the
    same interface (apart from the type of the first parameter).

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    dt : ``float``
       Time-step for discrete simulation of target system.
    method : ``string``, optional
       Method of discretization. Defaults to zero-order hold discretization
       (``'zoh'``), which assumes that the input signal is held constant over
       each discrete time-step. [#]_
    alpha : ``float`` or ``None``, optional
       Weighting parameter for use with ``method='gbt'``.

    Returns
    -------
    discrete_sys : :class:`.LinearSystem`
       Discretized linear system (``analog=False``).

    See Also
    --------
    :func:`.discrete2cont`
    :func:`scipy.signal.cont2discrete`

    Notes
    -----
    Discretization is often performed automatically whenever needed;
    usually it is unnecessary to deal with this routine directly. One
    exception is when combining systems across domains (see example).

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Discretization

    Examples
    --------
    Simulating an alpha synapse with a pure transmission delay:

    >>> from nengolib.signal import z, cont2discrete
    >>> from nengolib import Alpha
    >>> sys = Alpha(0.003)
    >>> dsys = z**(-20) * cont2discrete(sys, dt=sys.default_dt)
    >>> y = dsys.impulse(50)

    >>> assert np.allclose(np.sum(y), 1, atol=1e-3)
    >>> t = dsys.ntrange(len(y))

    >>> import matplotlib.pyplot as plt
    >>> plt.step(t, y, where='post')
    >>> plt.fill_between(t, np.zeros_like(y), y, step='post', alpha=.3)
    >>> plt.xlabel("Time (s)")
    >>> plt.show()
    """

    sys = LinearSystem(sys)
    if not sys.analog:
        raise ValueError("system (%s) is already discrete" % sys)
    return LinearSystem(
        _cont2discrete(sys.ss, dt=dt, method=method, alpha=alpha)[:-1],
        analog=False)


def discrete2cont(sys, dt, method='zoh', alpha=None):
    """Convert linear system from discrete to continuous time-domain.

    This is the inverse of :func:`.cont2discrete`. This will not work in
    general, for instance with the ZOH method when the system has discrete
    poles at ``0`` (e.g., systems with pure time-delay elements).

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    dt : ``float``
       Time-step used to *undiscretize* ``sys``.
    method : ``string``, optional
       Method of discretization. Defaults to zero-order hold discretization
       (``'zoh'``), which assumes that the input signal is held constant over
       each discrete time-step.
    alpha : ``float`` or ``None``, optional
       Weighting parameter for use with ``method='gbt'``.

    Returns
    -------
    continuous_sys : :class:`.LinearSystem`
       Continuous linear system (``analog=True``).

    See Also
    --------
    :func:`.cont2discrete`
    :func:`scipy.signal.cont2discrete`

    Examples
    --------
    Converting a double-exponential synapse back and forth between domains:

    >>> from nengolib.signal import discrete2cont, cont2discrete
    >>> from nengolib import DoubleExp
    >>> sys = DoubleExp(0.005, 0.2)
    >>> assert dsys == discrete2cont(cont2discrete(sys, dt=0.1), dt=0.1)
    """

    sys = LinearSystem(sys)
    if sys.analog:
        raise ValueError("system (%s) is already continuous" % sys)

    if dt <= 0:
        raise ValueError("dt (%s) must be positive" % (dt,))

    ad, bd, cd, dd = sys.ss
    n = ad.shape[0]
    m = n + bd.shape[1]

    if method == 'gbt':
        if alpha is None or alpha < 0 or alpha > 1:
            raise ValueError("alpha (%s) must be in range [0, 1]" % (alpha,))

        I = np.eye(n)
        ar = linalg.solve(alpha*dt*ad.T + (1-alpha)*dt*I, ad.T - I).T
        M = I - alpha*dt*ar

        br = np.dot(M, bd) / dt
        cr = np.dot(cd, M)
        dr = dd - alpha*np.dot(cr, bd)

    elif method in ('bilinear', 'tustin'):
        return discrete2cont(sys, dt, method='gbt', alpha=0.5)

    elif method in ('euler', 'forward_diff'):
        return discrete2cont(sys, dt, method='gbt', alpha=0.0)

    elif method == 'backward_diff':
        return discrete2cont(sys, dt, method='gbt', alpha=1.0)

    elif method == 'zoh':
        M = np.zeros((m, m))
        M[:n, :n] = ad
        M[:n, n:] = bd
        M[n:, n:] = np.eye(bd.shape[1])
        E = linalg.logm(M) / dt

        ar = E[:n, :n]
        br = E[:n, n:]
        cr = cd
        dr = dd

    else:
        raise ValueError("invalid method: '%s'" % (method,))

    return LinearSystem((ar, br, cr, dr), analog=True)
