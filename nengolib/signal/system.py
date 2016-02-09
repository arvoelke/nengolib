import numpy as np
from scipy.signal import (
    cont2discrete, zpk2ss, ss2tf, tf2ss, zpk2tf, lfilter, normalize,
    abcd_normalize)

from nengo.synapses import LinearFilter
from nengo.utils.compat import is_integer, is_number, with_metaclass

from nengolib.utils.meta import ReuseUnderlying

__all__ = [
    'sys2ss', 'sys2tf', 'canonical', 'sys_equal', 'apply_filter', 'impulse',
    'is_exp_stable', 'scale_state', 'NengoLinearFilterMixin', 'LinearSystem']


def _raise_invalid_sys():
    raise ValueError(
        "sys must be an instance of LinearSystem, a scalar, or a tuple of "
        "2 (tf), 3 (zpk), or 4 (ss) arrays.")


def sys2ss(sys):
    """Converts an LTI system in any form to state-space."""
    if isinstance(sys, LinearSystem):
        return sys.ss
    elif isinstance(sys, LinearFilter):
        return tf2ss(sys.num, sys.den)
    elif is_number(sys):
        return tf2ss(sys, 1)
    elif len(sys) == 2:
        return tf2ss(*sys)
    elif len(sys) == 3:
        return zpk2ss(*sys)
    elif len(sys) == 4:
        return sys
    else:
        _raise_invalid_sys()


def _ss2tf(A, B, C, D):
    """Fix ss2tf to handle zero-order state-space models."""
    # https://github.com/scipy/scipy/issues/5760
    if not (len(A) or len(B) or len(C)):
        D = np.asarray(D).flatten()
        if len(D) != 1:
            raise ValueError("D must be scalar for zero-order models")
        return ((D[0],), 1.)
    return ss2tf(A, B, C, D)


def sys2tf(sys):
    """Converts an LTI system in any form to a transfer function."""
    def _tf(num, den):
        return (np.poly1d(num), np.poly1d(den))

    if isinstance(sys, LinearSystem):
        # use cached attribute in case already computed
        return sys.tf  # _tf called via recursion to sys2tf
    elif isinstance(sys, LinearFilter):
        return _tf(sys.num, sys.den)
    elif is_number(sys):
        return _tf(sys, 1)
    elif len(sys) == 2:
        return _tf(*sys)
    elif len(sys) == 3:
        return _tf(*zpk2tf(*sys))
    elif len(sys) == 4:
        nums, den = _ss2tf(*sys)
        if len(nums) != 1:
            # TODO: support MIMO systems
            # https://github.com/scipy/scipy/issues/5753
            raise NotImplementedError("System must be SISO")
        return _tf(nums[0], den)
    else:
        _raise_invalid_sys()


def _is_canonical(A, B, C, D):
    """Returns true iff (A, B, C, D) is in controllable canonical form."""
    n = len(A)
    if not np.allclose(B[0], 1.0):
        return False
    if n <= 1:
        return True
    return (np.allclose(B[1:], 0) and
            np.allclose(A[1:, :-1], np.eye(n-1)) and
            np.allclose(A[1:, -1], 0))


def canonical(sys):
    """Converts SISO to controllable canonical form."""
    # TODO: raise nicer error if not SISO
    ss = abcd_normalize(*sys2ss(sys))
    if not _is_canonical(*ss):
        ss = sys2ss(sys2tf(ss))
        assert _is_canonical(*ss)
    return LinearSystem(ss)


def sys_equal(sys1, sys2):
    """Returns true iff sys1 and sys2 have the same transfer functions."""
    # TODO: doesn't do pole-zero cancellation
    tf1 = normalize(*sys2tf(sys1))
    tf2 = normalize(*sys2tf(sys2))
    for t1, t2 in zip(tf1, tf2):
        if len(t1) != len(t2) or not np.allclose(t1, t2):
            return False
    return True


def apply_filter(u, sys, dt, axis=-1):
    """Simulates sys on u for length timesteps of width dt."""
    # TODO: properly handle SIMO systems
    # https://github.com/scipy/scipy/issues/5753
    num, den = sys2tf(sys)
    if dt is not None:
        (num,), den, _ = cont2discrete((num, den), dt)
    return lfilter(num, den, u, axis)


def impulse(sys, dt, length, axis=-1):
    """Simulates sys on a delta impulse for length timesteps of width dt."""
    impulse = np.zeros(length)
    impulse[0] = 1
    return apply_filter(impulse, sys, dt, axis)


def _is_exp_stable(A):
    w, v = np.linalg.eig(A)
    return (w.real < 0).all()  # <=> e^(At) goes to 0


def is_exp_stable(sys):
    """Returns true iff system is exponentially stable."""
    return _is_exp_stable(sys2ss(sys)[0])


def scale_state(A, B, C, D, radii=1.0):
    """Scales the system to compensate for radii of the state."""
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
    return A, B, C, D


class _CanonicalStep(LinearFilter.Step):

    def __init__(self, sys, output):
        A, B, C, D = canonical(sys).ss
        self._a = A[0, :]
        assert len(C) == 1
        self._c = C[0, :]
        assert D.size == 1
        self._d = D.flatten()[0]
        self._x = np.zeros((len(self._a), len(np.atleast_1d(output))))
        self.output = output

    def __call__(self, u):
        r = np.dot(self._a, self._x)
        self._x[1:, :] = self._x[:-1, :]
        self._x[0, :] = r + u
        self.output[...] = np.dot(self._c, self._x) + self._d*u


class NengoLinearFilterMixin(LinearFilter):

    # Note: we don't use self.analog because that information will get
    # lost if the instance uses operations inherited from LinearSystem
    analog = True

    def make_step(self, dt, output, method='zoh'):
        A, B, C, D = sys2ss(self)
        if self.analog:  # pragma: no cover
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt, method=method)
        return _CanonicalStep((A, B, C, D), output)


class LinearSystem(with_metaclass(ReuseUnderlying, NengoLinearFilterMixin)):
    """Single-input single-output linear system with set of operations."""

    # Reuse the underlying system whenever it is an instance of the same
    # class. This allows us to avoid recomputing the tf/ss for the same
    # instance, i.e.
    #    sys1 = LinearSystem(...)
    #    tf1 = sys1.tf  # computes once
    #    sys2 = LinearSystem(sys1)
    #    assert sys1 is sys2  # reuses underlying instance
    #    tf2 = sys2.tf  # already been computed

    _tf = None
    _ss = None

    def __init__(self, sys):
        assert not isinstance(sys, LinearSystem)  # guaranteed by metaclass
        self._sys = sys
        # Don't initialize superclass so that it uses this num/den instead

    @property
    def tf(self):
        if self._tf is None:
            self._tf = sys2tf(self._sys)
        return self._tf

    @property
    def ss(self):
        if self._ss is None:
            self._ss = sys2ss(self._sys)
            # TODO: throw nicer error if system is acausal
        return self._ss

    @property
    def num(self):
        return self.tf[0]

    @property
    def den(self):
        return self.tf[1]

    @property
    def order_num(self):
        return len(self.num.coeffs) - 1

    @property
    def order_den(self):
        return len(self.den.coeffs) - 1

    @property
    def causal(self):
        return self.order_num <= self.order_den

    def __len__(self):
        return self.order_den

    def __repr__(self):
        return "LinearSystem(sys=(%r, %r))" % (
            np.asarray(self.num), np.asarray(self.den))

    def __str__(self):
        return "(%s, %s)" % (np.asarray(self.num), np.asarray(self.den))

    def __neg__(self):
        n, d = self.tf
        return LinearSystem((-n, d))

    def __pow__(self, other):
        if not is_integer(other):
            return NotImplemented
        n, d = self.tf
        if other > 0:
            return LinearSystem(normalize(n**other, d**other))
        elif other < 0:
            return LinearSystem(normalize(d**-other, n**-other))
        else:
            assert other == 0
            return LinearSystem(1)

    def __invert__(self):
        return self.__pow__(-1)

    def __add__(self, other):
        n1, d1 = self.tf
        n2, d2 = LinearSystem(other).tf
        if len(d1) == len(d2) and np.allclose(d1, d2):
            # short-cut to avoid needing pole-zero cancellation
            return LinearSystem((n1 + n2, d1))
        # TODO: pole-zero cancellation
        return LinearSystem(normalize(n1*d2 + n2*d1, d1*d2))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        n1, d1 = self.tf
        n2, d2 = LinearSystem(other).tf
        # TODO: pole-zero cancellation
        return LinearSystem(normalize(n1*n2, d1*d2))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__mul__(~LinearSystem(other))

    def __rdiv__(self, other):
        return (~self).__mul__(LinearSystem(other))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __eq__(self, other):
        return sys_equal(self.tf, LinearSystem(other).tf)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((tuple(self.num), tuple(self.den)))
