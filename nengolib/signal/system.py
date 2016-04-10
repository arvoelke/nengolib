import warnings

import numpy as np
from scipy.signal import (
    cont2discrete, zpk2ss, ss2tf, ss2zpk, tf2ss, tf2zpk, zpk2tf,
    normalize, abcd_normalize)

from nengo.synapses import LinearFilter
from nengo.utils.compat import is_integer, is_number, with_metaclass

__all__ = [
    'sys2ss', 'sys2tf', 'sys2zpk', 'canonical', 'sys_equal', 'ss_equal',
    'is_exp_stable', 'decompose_states', 'NengoLinearFilterMixin',
    'LinearSystem', 's', 'z']


_LSYS, _LFILT, _NUM, _TF, _ZPK, _SS = range(6)


def _sys2form(sys):
    if isinstance(sys, LinearSystem):
        return _LSYS
    elif isinstance(sys, LinearFilter):
        return _LFILT
    elif is_number(sys):
        return _NUM
    elif len(sys) == 2:
        return _TF
    elif len(sys) == 3:
        return _ZPK
    elif len(sys) == 4:
        return _SS
    else:
        raise ValueError(
            "sys must be an instance of LinearSystem, a scalar, or a tuple of "
            "2 (tf), 3 (zpk), or 4 (ss) arrays.")


def _tf(num, den):
    return (np.poly1d(num), np.poly1d(den))


def _ss2tf(A, B, C, D):
    # https://github.com/scipy/scipy/issues/5760
    if not (len(A) or len(B) or len(C)):
        D = np.asarray(D).flatten()
        if len(D) != 1:
            raise ValueError("D must be scalar for zero-order models")
        return (D[0], 1.)
    nums, den = ss2tf(A, B, C, D)
    if len(nums) != 1:
        # TODO: support MIMO systems
        # https://github.com/scipy/scipy/issues/5753
        raise NotImplementedError("System must be SISO")
    return nums[0], den


_sys2ss = {
    _LSYS: lambda sys: sys.ss,
    _LFILT: lambda sys: tf2ss(sys.num, sys.den),
    _NUM: lambda sys: tf2ss(sys, 1),
    _TF: lambda sys: tf2ss(*sys),
    _ZPK: lambda sys: zpk2ss(*sys),
    _SS: lambda sys: sys,
}

_sys2zpk = {
    _LSYS: lambda sys: sys.zpk,
    _LFILT: lambda sys: tf2zpk(sys.num, sys.den),
    _NUM: lambda sys: tf2zpk(sys, 1),
    _TF: lambda sys: tf2zpk(*sys),
    _ZPK: lambda sys: sys,
    _SS: lambda sys: ss2zpk(*sys),
}

_sys2tf = {
    _LSYS: lambda sys: sys.tf,
    _LFILT: lambda sys: _tf(sys.num, sys.den),
    _NUM: lambda sys: _tf(sys, 1),
    _TF: lambda sys: _tf(*sys),
    _ZPK: lambda sys: _tf(*zpk2tf(*sys)),
    _SS: lambda sys: _tf(*_ss2tf(*sys)),
}


def sys2ss(sys):
    """Converts an LTI system in any form to state-space."""
    return _sys2ss[_sys2form(sys)](sys)


def sys2zpk(sys):
    """Converts an LTI system in any form to zero-pole form."""
    return _sys2zpk[_sys2form(sys)](sys)


def sys2tf(sys):
    """Converts an LTI system in any form to a transfer function."""
    return _sys2tf[_sys2form(sys)](sys)


def _is_ccf(A, B, C, D):
    """Returns true iff (A, B, C, D) is in controllable canonical form."""
    n = len(A)
    if not np.allclose(B[0], 1.0):
        return False
    if n <= 1:
        return True
    return (np.allclose(B[1:], 0) and
            np.allclose(A[1:, :-1], np.eye(n-1)) and
            np.allclose(A[1:, -1], 0))


def canonical(sys, controllable=True):
    """Converts SISO to controllable/observable canonical form."""
    # TODO: raise nicer error if not SISO
    sys = LinearSystem(sys)
    ss = abcd_normalize(*sys.ss)
    if not _is_ccf(*ss):
        # TODO: if already observable than this might hurt the accuracy
        ss = sys2ss(sys2tf(ss))
        assert _is_ccf(*ss)
    if not controllable:
        ss = (ss[0].T, ss[2].T, ss[1].T, ss[3])
    return LinearSystem(ss, sys.analog)


def sys_equal(sys1, sys2, rtol=1e-05, atol=1e-08):
    """Returns true iff sys1 and sys2 have the same transfer functions."""
    # TODO: doesn't do pole-zero cancellation
    sys1 = LinearSystem(sys1)
    sys2 = LinearSystem(sys2)
    if sys1.analog != sys2.analog:
        raise ValueError("cannot compare analog with discrete system")
    tf1 = normalize(*sys2tf(sys1))
    tf2 = normalize(*sys2tf(sys2))
    for t1, t2 in zip(tf1, tf2):
        if len(t1) != len(t2) or not np.allclose(t1, t2, rtol=rtol, atol=atol):
            return False
    return True


def ss_equal(sys1, sys2, rtol=1e-05, atol=1e-08):
    """Returns true iff sys1 and sys2 have the same realizations."""
    sys1 = LinearSystem(sys1)
    sys2 = LinearSystem(sys2)
    if sys1.analog != sys2.analog:
        raise ValueError("cannot compare analog with discrete system")
    return (np.allclose(sys1.A, sys2.A, rtol=rtol, atol=atol) and
            np.allclose(sys1.B, sys2.B, rtol=rtol, atol=atol) and
            np.allclose(sys1.C, sys2.C, rtol=rtol, atol=atol) and
            np.allclose(sys1.D, sys2.D, rtol=rtol, atol=atol))


def _is_exp_stable(A):
    # TODO: we can avoid this computation if in zpk form
    w, v = np.linalg.eig(A)
    return (w.real < 0).all()  # <=> e^(At) goes to 0


def is_exp_stable(sys):
    """Returns true iff system is exponentially stable."""
    return _is_exp_stable(LinearSystem(sys).A)


def decompose_states(sys):
    """Returns the LinearSystem for each state."""
    sys = LinearSystem(sys)
    r = []
    for i in range(len(sys)):
        subsys = (sys.A, sys.B, np.eye(len(sys))[i:i+1, :], [[0]])
        r.append(LinearSystem(subsys, analog=sys.analog))
    return r


class _DigitalStep(LinearFilter.Step):

    def __init__(self, sys, output, y0=None, dtype=np.float64):
        A, B, C, D = canonical(sys).ss
        self._a = A[0, :]
        assert len(C) == 1
        self._c = C[0, :]
        assert D.size == 1
        self._d = D.flatten()[0]
        self._x = np.zeros(
            (len(self._a), len(np.atleast_1d(output))), dtype=dtype)
        self.output = output
        if y0 is not None:
            self.output[...] = y0

    def __call__(self, t, u):
        r = np.dot(self._a, self._x)
        self.output[...] = np.dot(self._c, self._x) + self._d*u
        self._x[1:, :] = self._x[:-1, :]
        self._x[0, :] = r + u
        return self.output


class NengoLinearFilterMixin(LinearFilter):

    seed = None

    def make_step(self, shape_in, shape_out, dt, rng, y0=None,
                  dtype=np.float64, method='zoh'):
        assert shape_in == shape_out
        output = np.zeros(shape_out)

        if self.analog:
            # Note: equivalent to cont2discrete in discrete.py, but repeated
            # here to avoid circular dependency.
            A, B, C, D, _ = cont2discrete(self.ss, dt, method=method)
            sys = LinearSystem((A, B, C, D), analog=False)
        else:
            sys = self

        if not sys.has_passthrough:
            # This makes our system behave like it does in Nengo
            sys *= z  # discrete shift of the system to remove delay
        else:
            warnings.warn("Synapse (%s) has extra delay due to passthrough "
                          "(https://github.com/nengo/nengo/issues/938)" % sys)

        return _DigitalStep(sys, output, y0=y0, dtype=dtype)


class LinearSystemType(type):

    def __call__(self, sys, analog=None):
        # if analog argument is given, then we must check if sys already has
        # an analog attribute (they must match)
        if analog is not None:
            # LinearFilter is the highest base class that contains analog
            if isinstance(sys, LinearFilter) and analog != sys.analog:
                raise TypeError("Cannot reuse existing instance (%s) with a "
                                "different analog attribute." % sys)
        else:
            # otherwise no analog attribute to inherit and none supplied
            analog = True  # default

        # if the given system is already a LinearSystem, then we should
        # reuse the instance. note that if an analog argument was provided
        # then it had to match the given system.
        if isinstance(sys, self):
            return sys

        # otherwise create a new instance with the determined analog attribute
        return super(LinearSystemType, self).__call__(sys, analog)


class LinearSystem(with_metaclass(LinearSystemType, NengoLinearFilterMixin)):
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
    _zpk = None

    def __init__(self, sys, analog):
        assert not isinstance(sys, LinearSystem)  # guaranteed by metaclass
        assert analog is not None  # guaranteed by metaclass
        self._sys = sys
        self._analog = analog
        # Don't initialize superclass so that it uses this num/den instead

    @property
    def analog(self):
        return self._analog

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
    def zpk(self):
        if self._zpk is None:
            self._zpk = sys2zpk(self._sys)
        return self._zpk

    @property
    def A(self):
        return self.ss[0]

    @property
    def B(self):
        return self.ss[1]

    @property
    def C(self):
        return self.ss[2]

    @property
    def D(self):
        return self.ss[3]

    @property
    def zeros(self):
        return self.zpk[0]

    @property
    def poles(self):
        return self.zpk[1]

    @property
    def gain(self):
        return self.zpk[2]

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
        if _sys2form(self._sys) == _SS or self._ss is not None:
            return len(self.A)  # avoids conversion to transfer function
        return len(self.den.coeffs) - 1

    @property
    def causal(self):
        return self.order_num <= self.order_den

    @property
    def has_passthrough(self):
        return self.num[self.order_den] != 0

    @property
    def proper(self):
        return self.causal and not self.has_passthrough

    def __len__(self):
        return self.order_den

    def __repr__(self):
        return "%s(sys=(%r, %r), analog=%r)" % (
            self.__class__.__name__, np.asarray(self.num),
            np.asarray(self.den), self.analog)

    def __str__(self):
        return "(%s, %s, %s)" % (
            np.asarray(self.num), np.asarray(self.den), self.analog)

    def _check_other(self, other):
        if isinstance(other, LinearFilter):  # base class containing analog
            if self.analog != other.analog:
                raise ValueError("incompatible %s objects: %s, %s; both must "
                                 "be analog or digital" % (
                                     self.__class__.__name__, self, other))

    def __neg__(self):
        n, d = self.tf
        return LinearSystem((-n, d), self.analog)

    def __pow__(self, other):
        if not is_integer(other):
            return NotImplemented
        n, d = self.tf
        if other > 0:
            return LinearSystem(normalize(n**other, d**other), self.analog)
        elif other < 0:
            return LinearSystem(normalize(d**-other, n**-other), self.analog)
        else:
            assert other == 0
            return LinearSystem(1., self.analog)

    def __invert__(self):
        return self.__pow__(-1)

    def __add__(self, other):
        self._check_other(other)
        n1, d1 = self.tf
        n2, d2 = LinearSystem(other, self.analog).tf
        if len(d1) == len(d2) and np.allclose(d1, d2):
            # short-cut to avoid needing pole-zero cancellation
            return LinearSystem((n1 + n2, d1), self.analog)
        return LinearSystem(normalize(n1*d2 + n2*d1, d1*d2), self.analog)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        self._check_other(other)
        n1, d1 = self.tf
        n2, d2 = LinearSystem(other, self.analog).tf
        return LinearSystem(normalize(n1*n2, d1*d2), self.analog)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        self._check_other(other)
        return self.__mul__(~LinearSystem(other, self.analog))

    def __rdiv__(self, other):
        self._check_other(other)
        return (~self).__mul__(LinearSystem(other, self.analog))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __eq__(self, other):
        self._check_other(other)
        return sys_equal(self.tf, LinearSystem(other, self.analog).tf)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        num, den = normalize(*self.tf)
        return hash((tuple(num), tuple(den), self.analog))


s = LinearSystem(([1, 0], [1]), analog=True)  # differential operator
z = LinearSystem(([1, 0], [1]), analog=False)  # shift operator
