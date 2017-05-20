import numpy as np
from scipy import linalg
from scipy.signal import cont2discrete as _cont2discrete

from nengolib.signal.system import LinearSystem

__all__ = ['cont2discrete', 'discrete2cont']


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
