import numpy as np
from numpy.linalg import solve
from scipy import linalg
from scipy.signal import cont2discrete as _cont2discrete, lfilter

from nengolib.signal.system import LinearSystem

__all__ = ['cont2discrete', 'discrete2cont', 'apply_filter', 'impulse']


def cont2discrete(sys, dt, method='zoh', alpha=None):
    sys = LinearSystem(sys)
    if not sys.analog:
        raise ValueError("system (%s) is already discrete" % sys)
    return LinearSystem(
        _cont2discrete(sys.ss, dt=dt, method=method, alpha=alpha)[:-1],
        analog=False)


def discrete2cont(sys, dt, method='zoh', alpha=None):
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
        ar = solve(alpha*dt*ad.T + (1-alpha)*dt*I, ad.T - I).T
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


def apply_filter(sys, dt, u, axis=-1):
    """Simulates sys on u for length timesteps of width dt."""
    # TODO: properly handle SIMO systems
    # https://github.com/scipy/scipy/issues/5753\
    sys = LinearSystem(sys)
    if dt is not None:
        num, den = cont2discrete(sys, dt).tf
    elif not sys.analog:
        num, den = sys.tf
    else:
        raise ValueError("system (%s) must be discrete if not given dt" % sys)

    # convert from the polynomial representation, and add back the leading
    # zeros that were dropped by poly1d, since lfilter will shift it the
    # wrong way (it will add the leading zeros back to the end, effectively
    # removing the delay)
    num, den = map(np.asarray, (num, den))
    num = np.append([0]*(len(den) - len(num)), num)
    return lfilter(num, den, u, axis)


def impulse(sys, dt, length, axis=-1):
    """Simulates sys on a delta impulse for length timesteps of width dt."""
    impulse = np.zeros(length)
    impulse[0] = 1. / dt if dt is not None else 1
    return apply_filter(sys, dt, impulse, axis)
