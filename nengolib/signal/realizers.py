from collections import namedtuple

import numpy as np

from scipy.linalg import inv

from nengolib.signal.lyapunov import state_norm, l1_norm, hankel
from nengolib.signal.reduction import balanced_transformation
from nengolib.signal.system import LinearSystem

__all__ = ['Identity', 'Balanced', 'Hankel', 'L1Norm', 'H2Norm']


class RealizerResult(namedtuple('Transformation',
                                ['sys', 'T', 'Tinv', 'realization'])):
    """Resulting namedtuple returned by the Realizer.

    Contains:
      sys         - original LinearSystem
      T           - similarity transformation
      Tinv        - inverse of T
      realization - transformed LinearSystem
    """

    __slots__ = ()

    def __new__(cls, sys, T, Tinv, realization):
        return tuple.__new__(cls, (sys, T, Tinv, realization))


def _realize(sys, radii, T, Tinv=None):
    sys = LinearSystem(sys)
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim == 0:
        r = np.ones(len(sys)) * r
    elif r.ndim > 1:
        raise ValueError("radii (%s) must be a 1-dim array or scalar" % (
            radii,))
    elif len(r) != len(sys):
        raise ValueError("radii (%s) length must match state dimension %d" % (
            radii, len(sys)))

    T = T * r[None, :]
    if Tinv is None:  # this needs to be computed eventually anyways
        Tinv = inv(T)
    else:
        Tinv = Tinv / r[:, None]

    return RealizerResult(sys, T, Tinv, sys.transform(T, Tinv))


class AbstractRealizer(object):
    """Abstract class for computing realizing transformations.

    The given radii should be with respect to the realized state-space.
    """

    def __call__(self, sys, radii=1):
        raise NotImplementedError("realizer must be callable")


class Identity(AbstractRealizer):
    """Only applies the radius to transform the current state-space."""

    def __call__(self, sys, radii=1):
        sys = LinearSystem(sys)
        I = np.eye(len(sys))
        return _realize(sys, radii, I, I)


class Balanced(AbstractRealizer):
    """Balanced realization given by the Gramiam matrices.

    Informally, this evenly distributes the energy of the state-vector
    across all dimensions. This has the effect of normalizing the
    representation of the state-space.
    """

    def __call__(self, sys, radii=1):
        sys = LinearSystem(sys)
        T, Tinv, _ = balanced_transformation(sys)
        return _realize(sys, radii, T, Tinv)


class Hankel(AbstractRealizer):
    """Diagonal realization given by Hankel singular values.

    This (generously) bounds the worst-case state vector by the given radii.
    Thus, the radii that are given should be much smaller than the actual
    desired radius in order to compensate.

    The worst-case output is given by the L1-norm of a system which in turn is
    bounded by 2 times the sum of the Hankel singular values [1]_.

    References:
        [1] Khaisongkram, W., and D. Banjerdpongchai. "On computing the
            worst-case norm of linear systems subject to inputs with magnitude
            bound and rate limit." International Journal of
            Control 80.2 (2007): 190-219.
    """

    def __call__(self, sys, radii=1):
        # TODO: this recomputes the same control_gram multiple times over
        sys = LinearSystem(sys)
        T = np.diag([2 * np.sum(hankel(sub)) for sub in sys])
        return _realize(sys, radii, T)


class L1Norm(AbstractRealizer):
    """Diagonal realization given by the L1-norm.

    This tightly bounds the worst-case state vector by the given radii.
    Thus, the radii that are given should be much smaller (even smaller than
    with Hankel) than the actual desired radius in order to compensate.

    This enforces that any input (even full spectrum white-noise) bounded by
    [-1, +1] will keep the state within the given radii.

    However, in practice the worst-case may be highly unachievable since in
    general it requires an input that rapidly oscillates between +1 and -1.

    See ``l1_norm`` for details.
    """

    def __init__(self, rtol=1e-6, max_length=2**18):
        self.rtol = rtol
        self.max_length = max_length
        super(L1Norm, self).__init__()

    def __call__(self, sys, radii=1):
        # TODO: this also recomputes many subcalculations in l1_norm
        sys = LinearSystem(sys)
        T = np.diag(np.atleast_1d(np.squeeze(
            [l1_norm(sub, rtol=self.rtol, max_length=self.max_length)[0]
             for sub in sys])))
        return _realize(sys, radii, T)


class H2Norm(AbstractRealizer):
    """Diagonal realization given by the H2-norm of the system's state."""

    def __call__(self, sys, radii=1):
        sys = LinearSystem(sys)
        T = np.diag(state_norm(sys, 'H2'))
        return _realize(sys, radii, T)
