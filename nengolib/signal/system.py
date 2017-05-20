import warnings

import numpy as np
from scipy.linalg import inv
from scipy.signal import (
    cont2discrete, zpk2ss, ss2tf, ss2zpk, tf2ss, tf2zpk, zpk2tf, normalize)

from nengo.synapses import LinearFilter
from nengo.utils.compat import is_integer, is_number, with_metaclass

__all__ = [
    'sys2ss', 'sys2tf', 'sys2zpk', 'canonical', 'sys_equal', 'ss_equal',
    'NengoLinearFilterMixin', 'LinearSystem', 's', 'z']


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


def _ss(abcd):
    return tuple(map(np.atleast_2d, abcd))


def _ss2tf(A, B, C, D):
    # https://github.com/scipy/scipy/issues/5760
    if not (len(A) or len(B) or len(C)):
        D = np.asarray(D).flatten()
        if len(D) != 1:
            raise ValueError("D must be scalar for zero-order models")
        return (D[0], 1.)  # pragma: no cover; solved in scipy>=0.18rc2
    nums, den = ss2tf(A, B, C, D)
    if len(nums) != 1:
        # TODO: support MIMO/SIMO/MISO systems
        # https://github.com/scipy/scipy/issues/5753
        raise NotImplementedError("System (%s, %s, %s, %s) must be SISO to "
                                  "convert to transfer function" %
                                  (A, B, C, D))
    return nums[0], den


_sys2ss = {
    _LSYS: lambda sys: _ss(sys.ss),
    _LFILT: lambda sys: _ss(tf2ss(sys.num, sys.den)),
    _NUM: lambda sys: _ss(tf2ss(sys, 1)),
    _TF: lambda sys: _ss(tf2ss(*sys)),
    _ZPK: lambda sys: _ss(zpk2ss(*sys)),
    _SS: lambda sys: _ss(sys),
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
    if A.size == 0 and B.size == 0 and C.size == 0:  # scipy 0.17.0
        return True  # TODO: assumes SISO?
    if not np.allclose(B[0], 1.0):
        return False
    if n <= 1:
        return True
    return (np.allclose(B[1:], 0) and
            np.allclose(A[1:, :-1], np.eye(n-1)) and
            np.allclose(A[1:, -1], 0))


def canonical(sys, controllable=True):
    """Converts SISO to controllable/observable canonical form."""
    sys = LinearSystem(sys)
    ss = sys.ss  # abcd_normalize(*sys.ss)
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
        raise ValueError("Cannot compare analog with digital system")
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
        raise ValueError("Cannot compare analog with digital system")
    return (np.allclose(sys1.A, sys2.A, rtol=rtol, atol=atol) and
            np.allclose(sys1.B, sys2.B, rtol=rtol, atol=atol) and
            np.allclose(sys1.C, sys2.C, rtol=rtol, atol=atol) and
            np.allclose(sys1.D, sys2.D, rtol=rtol, atol=atol))


class _CanonicalStep(LinearFilter.Step):
    """Stepper for LinearSystem in canonical SISO form."""

    def __init__(self, sys, output, y0=None, dtype=np.float64):
        A, B, C, D = canonical(sys).ss
        self._a = A[0, :]
        assert len(C) == 1
        self._c = C[0, :]
        assert D.size == 1
        self._d = D.item()
        self._x = np.zeros((len(self._a),) + output.shape, dtype=dtype)
        self.output = output

    def __call__(self, _, u):
        self.output[...] = np.dot(self._c, self._x) + self._d*u
        r = np.dot(self._a, self._x)
        self._x[1:, :] = self._x[:-1, :]
        self._x[:1, :] = r + u
        return self.output


class NengoLinearFilterMixin(LinearFilter):

    seed = None
    default_dt = 0.001

    def make_step(self, shape_in, shape_out, dt, rng, y0=None,
                  dtype=np.float64, method='zoh'):
        assert shape_in == shape_out
        output = np.zeros(shape_out)
        if y0 is not None:
            output[...] = y0
        if y0 is None or not np.allclose(y0, 0):
            warnings.warn(
                "y0 (%s!=0) does not properly initialize the system; see "
                "Nengo issue #1124." % y0, UserWarning)

        if self.analog:
            # TODO: equivalent to cont2discrete in discrete.py, but repeated
            # here to avoid circular dependency.
            A, B, C, D, _ = cont2discrete(self.ss, dt, method=method)
            sys = LinearSystem((A, B, C, D), analog=False)
        else:
            sys = self

        if not sys.has_passthrough:
            # This makes our system behave like it does in Nengo
            sys *= z  # discrete shift of the system to remove delay
        else:
            warnings.warn(
                "Synapse (%s) has extra delay due to passthrough "
                "(https://github.com/nengo/nengo/issues/938)." % sys)

        return _CanonicalStep(sys, output, y0=y0, dtype=dtype)


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
        # HACK: Don't initialize superclass, so that it uses this num/den

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
    def is_tf(self):
        return _sys2form(self._sys) == _TF or self._tf is not None

    @property
    def is_ss(self):
        return _sys2form(self._sys) == _SS or self._ss is not None

    @property
    def is_zpk(self):
        return _sys2form(self._sys) == _ZPK or self._zpk is not None

    @property
    def size_in(self):
        if self.is_ss:
            return self.B.shape[1]
        return 1

    @property
    def size_out(self):
        if self.is_ss:
            return self.C.shape[0]
        return 1

    @property
    def is_SISO(self):
        return self.size_in == 1 and self.size_out == 1

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
        if self.is_ss:
            return len(self.A)  # avoids conversion to transfer function
        return len(self.den.coeffs) - 1

    @property
    def causal(self):
        """Returns True if and only if the system is causal / proper."""
        return self.order_num <= self.order_den

    @property
    def has_passthrough(self):
        """Returns True if and only if the system has a passthrough."""
        # Note there may be numerical issues for values close to 0
        # since scipy routines occasionally "normalize "those to 0
        if self.is_ss:
            return np.any(self.D != 0)
        return self.num[self.order_den] != 0

    @property
    def strictly_proper(self):
        """Returns True if and only if the system is *strictly* proper."""
        return self.causal and not self.has_passthrough

    @property
    def dcgain(self):
        # http://www.mathworks.com/help/control/ref/dcgain.html
        return self(0 if self.analog else 1)

    @property
    def is_stable(self):
        w = self.poles  # eig(A)
        if not len(w):
            assert len(self) == 0  # only a passthrough
            return True
        if not self.analog:
            return np.max(abs(w)) < 1  # within unit circle
        return np.max(w.real) < 0  # within left half-plane

    def __call__(self, s):
        return self.num(s) / self.den(s)

    def __len__(self):
        return self.order_den

    def __repr__(self):
        return "%s(sys=%r, analog=%r)" % (
            type(self).__name__, self._sys, self.analog)

    def __str__(self):
        if self.is_ss:
            return "(A=%s, B=%s, C=%s, D=%s, analog=%s)" % (
                self.A, self.B, self.C, self.D, self.analog)
        return "(num=%s, den=%s, analog=%s)" % (
            np.asarray(self.num), np.asarray(self.den), self.analog)

    def _check_other(self, other):
        if isinstance(other, LinearFilter):  # base class containing analog
            if self.analog != other.analog:
                raise ValueError("incompatible %s objects: %s, %s; both must "
                                 "be analog or digital" % (
                                     type(self).__name__, self, other))

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
        if isinstance(other, LinearFilter):  # base class containing analog
            if self.analog != other.analog:
                return False
        return sys_equal(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        num, den = normalize(*self.tf)
        return hash((tuple(num), tuple(den), self.analog))

    @property
    def controllable(self):
        """Returns the LinearSystem in controllable canonical form."""
        return canonical(self, controllable=True)

    @property
    def observable(self):
        """Returns the LinearSystem in observable canonical form."""
        return canonical(self, controllable=False)  # observable

    def transform(self, T, Tinv=None):
        """Changes basis of state-space matrices to T.

        Then dot(T, x_new(t)) = x_old(t) after this transformation.
        """
        A, B, C, D = self.ss
        if Tinv is None:
            Tinv = inv(T)
        TA = np.dot(Tinv, np.dot(A, T))
        TB = np.dot(Tinv, B)
        TC = np.dot(C, T)
        TD = D
        return LinearSystem((TA, TB, TC, TD), analog=self.analog)

    def __iter__(self):
        """Yields the LinearSystem corresponding to each state."""
        I = np.eye(len(self))
        for i in range(len(self)):
            sys = (self.A, self.B, I[i:i+1, :], np.zeros((1, self.size_in)))
            yield LinearSystem(sys, analog=self.analog)

    @property
    def X(self):
        """Returns the multiple-output system for the state-space vector."""
        C = np.eye(len(self))
        D = np.zeros((len(self), self.size_in))
        return LinearSystem((self.A, self.B, C, D), analog=self.analog)

    def filt(self, u, dt=None, axis=0, y0=0, copy=True, filtfilt=False):
        """Filter the input using this linear system."""
        # Defaults y0=0 because y0=None has strange behaviour;
        # see unit test: test_system.py -k test_filt_issue_nengo938
        u = np.asarray(u)  # nengo PR 1123
        if not self.is_SISO:
            # Work-in-progress for issue # 106
            # TODO: relax all of these constraints
            if u.ndim == 1:
                u = u[:, None]
            if u.shape[1] != self.size_in:
                raise ValueError("Filtering with non-SISO systems requires "
                                 "the second dimension of x (%s) to equal "
                                 "the system's size_in (%s)." %
                                 (u.shape[1], self.size_in))

            if axis != 0 or y0 is None or not np.allclose(y0, 0) or \
               not copy or filtfilt:
                raise ValueError("Filtering with non-SISO systems requires "
                                 "axis=0, y0=0, copy=True, filtfilt=False.")

            warnings.warn("Filtering with non-SISO systems is an "
                          "experimental feature that may not behave as "
                          "expected.", UserWarning)

            if self.analog:
                dt = self.default_dt if dt is None else dt
                A, B, C, D, _ = cont2discrete(self.ss, dt, method='zoh')
            else:
                A, B, C, D = self.ss

            x = np.zeros(len(self), dtype=u.dtype)
            if self.size_out > 1:
                shape_out = (len(u), self.size_out)
            else:
                shape_out = (len(u),)
            y = np.empty(shape_out, dtype=u.dtype)
            for i, u_i in enumerate(u):
                y[i] = np.dot(C, x) + np.dot(D, u_i)
                x = np.dot(A, x) + np.dot(B, u_i)
            return y

        return super(LinearSystem, self).filt(u, dt, axis, y0, copy, filtfilt)

    def impulse(self, length, dt=None):
        """Impulse response with ``length`` timesteps and width ``dt``."""
        if dt is None:
            if self.analog:
                h = 1. / self.default_dt
            else:
                h = 1.
        else:
            h = 1. / dt
        delta = np.zeros(length)
        delta[0] = h
        return self.filt(delta, dt=dt, y0=0)


s = LinearSystem(([1, 0], [1]), analog=True)  # differential operator
z = LinearSystem(([1, 0], [1]), analog=False)  # shift operator
