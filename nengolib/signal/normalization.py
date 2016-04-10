import numpy as np

from nengolib.signal.reduction import hankel
from nengolib.signal.system import LinearSystem, decompose_states, canonical


__all__ = ['scale_state', 'AbstractNormalizer', 'Controllable',
           'Observable', 'AbstractScaleState', 'HankelNorm']


def scale_state(sys, radii=1.0):
    """Scales the system to compensate for radii of the state."""
    # TODO: move to another file
    sys = LinearSystem(sys)
    A, B, C, D = sys.ss
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim > 1:
        raise ValueError("radii (%s) must be a 1-dim array or scalar" % (
            radii,))
    elif r.ndim == 0:
        r = np.ones(len(A)) * r
    elif len(r) != len(A):
        raise ValueError("radii (%s) length must match state dimension %d" % (
            radii, len(A)))
    A = A / r[:, None] * r
    B = B / r[:, None]
    C = C * r
    return LinearSystem((A, B, C, D), analog=sys.analog)


class AbstractNormalizer(object):
    """Abstract class for normalizing a LinearSystem."""

    def __call__(self, sys):
        raise NotImplementedError("normalizer must be callable")


class Controllable(AbstractNormalizer):
    """Normalizes the system in controllable canonical form."""

    def __call__(self, sys, radii=1.0):
        return scale_state(canonical(sys, controllable=True), radii=radii), {}


class Observable(AbstractNormalizer):
    """Normalizes the system in observable canonical form."""

    def __call__(self, sys, radii=1.0):
        return scale_state(canonical(sys, controllable=False), radii=radii), {}


class AbstractScaleState(AbstractNormalizer):
    """Abstract class for normalizing a LinearSystem by scaling the states."""

    def __call__(self, sys, radii=1.0):
        sys = LinearSystem(sys)
        radii *= np.asarray(self.radii(sys))
        return scale_state(sys, radii=radii), {'radii': radii}

    def radii(self, sys):
        raise NotImplementedError("normalizer must implement radii method")


class HankelNorm(AbstractScaleState):
    """Upper-bounds the worst-case state using Hankel singular values.

    This enforces that any input (even full spectrum white-noise) within
    [-1, 1] will keep the state within the given radii. However, since this is
    an upper-bound, space may be "wasted" by over-estimating.

    An upper-bound on the worst-case output is given by the L1-norm of a system
    which in turn is bounded by 2 times the sum of the Hankel singular
    values _[1].

    References:
        [1] Khaisongkram, W., and D. Banjerdpongchai. "On computing the
            worst-case norm of linear systems subject to inputs with magnitude
            bound and rate limit." International Journal of
            Control 80.2 (2007): 190-219.
    """

    def radii(self, sys):
        return [2 * np.sum(hankel(sub)) for sub in decompose_states(sys)]
