from __future__ import division

import numpy as np
import pytest

import nengo

from scipy.linalg import inv
from scipy.signal import lfilter

from nengolib.signal.system import (
    sys2ss, sys2zpk, sys2tf, canonical, sys_equal, ss_equal, LinearSystem,
    s, z)
from nengolib import Network, Lowpass, Alpha
from nengolib.compat import warns
from nengolib.signal import cont2discrete, shift
from nengolib.synapses import PadeDelay


def test_sys_conversions():
    sys = Alpha(0.1)

    tf = sys2tf(sys)
    ss = sys2ss(sys)
    zpk = sys2zpk(sys)

    assert sys_equal(sys2ss(tf), ss)
    assert sys_equal(sys2ss(ss), ss)  # unchanged
    assert sys_equal(sys2tf(tf), tf)  # unchanged
    assert sys_equal(sys2tf(ss), tf)

    assert sys_equal(sys2zpk(zpk), zpk)  # sanity check
    assert sys_equal(sys2zpk(tf), zpk)  # sanity check
    assert sys_equal(sys2zpk(ss), zpk)
    assert sys_equal(sys2tf(zpk), tf)
    assert sys_equal(sys2ss(zpk), ss)

    # should also work with nengo's synapse types
    assert sys_equal(sys2zpk(nengo.Alpha(0.1)), zpk)
    assert sys_equal(sys2tf(nengo.Alpha(0.1)), tf)
    assert sys_equal(sys2ss(nengo.Alpha(0.1)), ss)

    # system can also be just a scalar
    assert sys_equal(sys2tf(2.0), (1, 0.5))
    assert np.allclose(sys2ss(5)[3], 5)
    assert sys_equal(sys2zpk(5), 5)

    with pytest.raises(ValueError):
        sys2ss(np.zeros(5))

    with pytest.raises(ValueError):
        sys2zpk(np.zeros(5))

    with pytest.raises(ValueError):
        sys2tf(np.zeros(5))

    with pytest.raises(ValueError):
        # _ss2tf(...): passthrough must be single element
        sys2tf(([], [], [], [1, 2]))


def test_check_sys_equal():
    assert not sys_equal(np.ones(2), np.ones(3))

    assert s != z
    assert not z == s
    assert LinearSystem(5, analog=True) != LinearSystem(5, analog=False)

    with pytest.raises(ValueError):
        sys_equal(s, z)

    with pytest.raises(ValueError):
        ss_equal(s, z)


def test_canonical():
    sys = ([1], [1], [1], [0])
    assert sys_equal(canonical(sys), sys)

    sys = ([[1, 0], [1, 0]], [[1], [0]], [[1, 1]], [0])
    assert sys_equal(canonical(sys), sys)

    sys = ([[1, 0], [0, 1]], [[0], [1]], [[1, 1]], [0])
    assert sys_equal(canonical(sys), sys)

    sys = ([[1, 0], [0, 1]], [[1], [0]], [[1, 1]], [0])
    assert sys_equal(canonical(sys), sys)

    sys = ([[1, 0], [0, 0]], [[1], [0]], [[1, 1]], [0])
    assert sys_equal(canonical(sys), sys)

    sys = ([[1, 0], [0, 0]], [[0], [1]], [[1, 1]], [0])
    assert sys_equal(canonical(sys), sys)

    sys = ([[1, 0, 1], [0, 1, 1], [1, 0, 0]], [[0], [1], [-1]],
           [[1, 1, 1]], [0])
    assert sys_equal(canonical(sys), sys)

    sys = nengo.Alpha(0.1)
    csys = canonical(sys, controllable=True)
    osys = canonical(sys, controllable=False)

    assert ss_equal(csys, LinearSystem(sys).controllable)
    assert ss_equal(osys, LinearSystem(sys).observable)

    assert sys_equal(csys, osys)
    assert not ss_equal(csys, osys)  # different state-space realizations

    A, B, C, D = csys.ss
    assert sys_equal(csys, sys)
    assert ss_equal(csys,
                    ([[-20, -100], [1, 0]], [[1], [0]], [[0, 100]], [[0]]))

    assert sys_equal(osys, sys)
    assert ss_equal(osys,
                    ([[-20, 1], [-100, 0]], [[0], [100]], [[1, 0]], [[0]]))


def test_is_stable():
    sys = Lowpass(0.1)
    assert sys.is_stable

    assert not (~s).is_stable  # integrator

    assert LinearSystem(1).is_stable

    assert (~(z * (z - 0.5))).is_stable
    assert not (z / (z - 1)).is_stable  # discrete integrator


@pytest.mark.parametrize("sys", [PadeDelay(0.1, 4), PadeDelay(0.2, 5, 5)])
def test_decompose_states(sys):
    assert np.dot(sys.C, list(sys)) + sys.D == sys


def test_non_siso_manipulation():
    sys = Alpha(0.1)
    A, B, C, D = sys.ss

    SIMO = LinearSystem((A, B, np.eye(len(A)), [[0], [0]]))
    assert not SIMO.is_SISO
    assert SIMO.size_in == 1
    assert SIMO.size_out == 2
    assert SIMO.shape == (2, 1)
    assert not SIMO.has_passthrough
    assert ss_equal(_eval(SIMO), SIMO)
    assert isinstance(str(SIMO), str)
    assert ss_equal(canonical(SIMO), SIMO)
    for sub1, sub2 in zip(sys, SIMO):
        assert ss_equal(sub1, sub2)

    MISO = LinearSystem((A, [[1, 1]], C, [[0, 1]]))
    assert not MISO.is_SISO
    assert MISO.size_in == 2
    assert MISO.size_out == 1
    assert MISO.shape == (1, 2)
    assert MISO.has_passthrough
    assert ss_equal(_eval(MISO), MISO)
    assert isinstance(str(MISO), str)

    MIMO = LinearSystem((A, [[1, 1]], np.eye(len(A)), np.zeros((2, 2))))
    assert not MIMO.is_SISO
    assert MIMO.size_in == MIMO.size_out == 2
    assert MIMO.shape == (2, 2)
    assert not MIMO.has_passthrough
    assert ss_equal(_eval(MIMO), MIMO)
    assert isinstance(str(MIMO), str)
    for sub1, sub2 in zip(MISO, MIMO):
        assert ss_equal(sub1, sub2)


def test_non_siso_filtering(rng):
    sys = PadeDelay(0.1, order=4)
    length = 1000

    SIMO = sys.X
    assert not SIMO.is_SISO
    assert SIMO.size_in == 1
    assert SIMO.size_out == len(sys)

    x = SIMO.impulse(length)
    for i, (sub1, sub2) in enumerate(zip(sys, SIMO)):
        assert sub1 == sub2
        y1 = sub1.impulse(length)
        y2 = sub2.impulse(length)
        _transclose(shift(y1), shift(y2), x[:, i])

    B = np.asarray([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]) * sys.B
    u = rng.randn(length, 3)

    Bu = u.dot([1, 2, 3])
    assert Bu.shape == (length,)
    MISO = LinearSystem((sys.A, B, sys.C, np.zeros((1, 3))), analog=True)
    assert not MISO.is_SISO
    assert MISO.size_in == 3
    assert MISO.size_out == 1

    y = cont2discrete(MISO, dt=0.001).filt(u)
    assert y.shape == (length,)
    assert np.allclose(shift(sys.filt(Bu)), y)

    MIMO = MISO.X
    assert not MIMO.is_SISO
    assert MIMO.size_in == 3
    assert MIMO.size_out == 4

    y = MIMO.filt(u)
    I = np.eye(len(sys))
    for i, sub1 in enumerate(MIMO):
        sub2 = LinearSystem((sys.A, B, I[i:i+1], np.zeros((1, 3))))
        _transclose(sub1.filt(u), sub2.filt(u), y[:, i])


def test_bad_filt():
    sys = PadeDelay(0.1, order=4).X
    with pytest.raises(ValueError):
        sys.filt(np.ones((4, 4)))
    with pytest.raises(ValueError):
        sys.filt(np.ones((4, 1)), filtfilt=True)
    with pytest.raises(ValueError):
        sys.filt(np.ones((4,)), copy=False)


@pytest.mark.parametrize("sys", [
    Lowpass(0.01), Alpha(0.2), LinearSystem(([1, 1], [0.01, 1]))])
def test_simulation(sys, Simulator, plt, seed):
    assert isinstance(sys, LinearSystem)
    old_sys = nengo.LinearFilter(sys.num, sys.den)
    assert sys == old_sys

    with Network() as model:
        stim = nengo.Node(output=nengo.processes.WhiteSignal(
            1.0, high=10, seed=seed))
        out_new = nengo.Node(size_in=2)
        out_old = nengo.Node(size_in=2)
        nengo.Connection(stim, out_new, transform=[[1], [-1]], synapse=sys)
        nengo.Connection(stim, out_old, transform=[[1], [-1]], synapse=old_sys)
        p_new = nengo.Probe(out_new)
        p_old = nengo.Probe(out_old)

    with Simulator(model) as sim:
        sim.run(1.0)

    plt.figure()
    plt.plot(sim.trange(), sim.data[p_new])
    plt.plot(sim.trange(), sim.data[p_old])

    assert np.allclose(sim.data[p_new], sim.data[p_old])


def test_discrete_synapse(Simulator):
    # Test that discrete synapses are simulated properly
    delay_steps = 50

    with Network() as model:
        stim = nengo.Node(output=np.sin)
        output = nengo.Node(size_in=1)
        nengo.Connection(stim, output, synapse=z**-delay_steps)
        p_stim = nengo.Probe(stim, synapse=None)
        p_output = nengo.Probe(output, synapse=None)

    with Simulator(model) as sim:
        sim.run(1.0)

    assert np.allclose(sim.data[p_output][delay_steps:],
                       sim.data[p_stim][:-delay_steps])


def _apply_filter(sys, dt, u):
    # "Correct" implementation of filt that has a single time-step delay
    # see Nengo issue #938
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
    return lfilter(num, den, u, axis=-1)


def test_filt():
    u = np.asarray([1.0, 0, 0])
    dt = 0.1
    num, den = [1], [1, 2, 1]
    sys1 = nengo.LinearFilter(num, den)
    sys2 = LinearSystem((num, den))  # uses a different make_step
    y1 = sys1.filt(u, dt=dt, y0=0)
    y2 = sys2.filt(u, dt=dt, y0=0)
    assert np.allclose(y1, y2)


def test_filt_issue_nengo1124():
    with warns(UserWarning):
        Lowpass(0.1).filt(np.asarray([1, 0]), dt=0.001, y0=1)

    with warns(UserWarning):
        Lowpass(0.1).filt(np.asarray([1, 0]), dt=0.001, y0=None)


def test_filt_issue_64():
    # Issue #64
    # Note we need a double delay because of Nengo issue #938 (see below)
    assert np.allclose((~z**2).filt([1, 0, 0]), [0, 1, 0])


def _transclose(*arrays):
    assert len(arrays) >= 2
    for i in range(len(arrays)-1):
        assert np.allclose(arrays[i], arrays[i+1]), i


def test_filt_issue_nengo938():
    # Testing related to nengo issue #938
    # test combinations of _apply_filter / filt on nengo / nengolib
    # using a passthrough / (strictly) proper and y0=0 / y0=None
    # ... in an **ideal** world all of these would produce the same results
    # but we assert behaviour here so that it is at least documented
    # and so that we are alerted to any changes in these differences
    # https://github.com/nengo/nengo/issues/938
    # https://github.com/nengo/nengo/issues/1124

    sys_prop_nengo = nengo.LinearFilter([1], [1, 0])
    sys_prop_nglib = LinearSystem(([1], [1, 0]))
    sys_pass_nengo = nengo.LinearFilter([1e-9, 1], [1, 0])
    sys_pass_nglib = LinearSystem(([1e-9, 1], [1, 0]))

    u = np.asarray([1.0, 0.5, 0])
    dt = 0.001

    def filt_scipy(sys):
        return _apply_filter(sys, dt=dt, u=u)

    def filt_nengo(sys, y0):
        return sys.filt(u, dt=dt, y0=y0)

    # Strictly proper transfer function
    prop_nengo_apply = filt_scipy(sys_prop_nengo)
    prop_nglib_apply = filt_scipy(sys_prop_nglib)
    prop_nengo_filt0 = filt_nengo(sys_prop_nengo, y0=0)
    prop_nglib_filt0 = filt_nengo(sys_prop_nglib, y0=0)
    prop_nengo_filtN = filt_nengo(sys_prop_nengo, y0=None)
    prop_nglib_filtN = filt_nengo(sys_prop_nglib, y0=None)

    # => two equivalence classes
    _transclose(prop_nengo_apply, prop_nglib_apply)
    _transclose(prop_nengo_filt0, prop_nglib_filt0, prop_nglib_filtN)

    # One-step delay difference between these two classes
    _transclose(prop_nengo_apply[1:], prop_nengo_filt0[:-1])

    # Passthrough transfer functions
    pass_nengo_apply = filt_scipy(sys_pass_nengo)
    pass_nglib_apply = filt_scipy(sys_pass_nglib)
    pass_nengo_filt0 = filt_nengo(sys_pass_nengo, y0=0)
    pass_nglib_filt0 = filt_nengo(sys_pass_nglib, y0=0)
    pass_nengo_filtN = filt_nengo(sys_pass_nengo, y0=None)
    pass_nglib_filtN = filt_nengo(sys_pass_nglib, y0=None)

    # => almost all are equivalent (except nengo with y0=None)
    _transclose(pass_nengo_apply, pass_nglib_apply, pass_nengo_filt0,
                pass_nglib_filt0, pass_nglib_filtN)
    assert not np.allclose(prop_nengo_filtN, pass_nengo_filtN)

    # And belongs to the same equivalence class as the very first
    _transclose(prop_nengo_apply, pass_nengo_apply)


def test_sys_multiplication():
    # Check that alpha is just two lowpass multiplied together
    assert Lowpass(0.1) * Lowpass(0.1) == Alpha(0.1)


def test_sim_new_synapse(Simulator):
    # Create a new synapse object and simulate it
    synapse = Lowpass(0.1) - Lowpass(0.01)
    with Network() as model:
        stim = nengo.Node(output=np.sin)
        x = nengo.Node(size_in=1)
        nengo.Connection(stim, x, synapse=synapse)
        p_stim = nengo.Probe(stim, synapse=None)
        p_x = nengo.Probe(x, synapse=None)

    with Simulator(model) as sim:
        sim.run(0.1)

    assert np.allclose(shift(synapse.filt(sim.data[p_stim])),
                       sim.data[p_x])


def _eval(sys):
    return eval(
        repr(sys), {}, {'LinearSystem': LinearSystem, 'array': np.array})


def test_linear_system():
    tau = 0.05
    sys = Lowpass(tau)
    dsys = (1 - np.exp(-1)) * z / (1 - np.exp(-1) * z)

    # Test attributes before state-space/zpk computed
    assert sys.is_tf
    assert not sys.is_ss
    assert not sys.is_zpk

    # Test representations
    assert sys == (1, [tau, 1])
    assert sys_equal(sys.tf, sys)
    assert sys_equal(sys.ss, sys)
    assert sys_equal(sys.zpk, sys)

    # Test attributes after state-space/zpk computed
    assert sys.is_tf
    assert sys.is_ss
    assert sys.is_zpk

    # Size in/out-related properties
    assert sys.is_SISO
    assert dsys.is_SISO
    assert sys.size_in == sys.size_out == dsys.size_in == dsys.size_out == 1

    # Test attributes
    assert np.allclose(sys.num, (1/tau,))
    assert np.allclose(sys.den, (1, 1/tau))
    assert sys.causal
    assert sys.strictly_proper
    assert not sys.has_passthrough
    assert not (sys/s).has_passthrough
    assert (sys*s).has_passthrough
    assert (sys*s).causal
    assert not (sys*s).strictly_proper
    assert not (sys*s*s).has_passthrough and not (sys*s*s).causal
    assert (sys*s*s + sys*s).has_passthrough

    assert np.allclose(sys.A, -1/tau)
    assert np.allclose(sys.B, 1)
    assert np.allclose(sys.C, 1/tau)
    assert np.allclose(sys.D, 0)

    assert np.allclose(sys.zeros, [0])
    assert np.allclose(sys.poles, [-1/tau])
    assert np.allclose(sys.gain, 1/tau)
    assert np.allclose(sys.zpk[0], np.array([]))
    assert np.allclose(sys.zpk[1], np.array([-1/tau]))
    assert np.allclose(sys.zpk[2], 1/tau)

    assert sys.order_num == 0
    assert sys.order_den == 1
    assert len(sys) == 1  # order_den
    assert len(LinearSystem(sys.ss)) == 1  # uses state-space rep

    # Test dcgain and __call__
    assert np.allclose(sys.dcgain, 1)
    assert np.allclose(dsys.dcgain, 1)
    assert np.allclose((s*sys)(1e12), 1.0 / tau)  # initial value theorem
    assert np.allclose((s*sys)(0), 0)  # final value theorem
    assert np.allclose(((1 - ~z)*dsys)(1), 0)  # final value theorem

    # Test multiplication and squaring
    assert sys*2 == 2*sys
    assert (0.4*sys) + (0.6*sys) == sys
    assert sys + sys == 2*sys
    assert sys * sys == sys**2
    assert sys_equal(sys + sys*sys, sys*sys + sys)

    # Test pow
    with pytest.raises(TypeError):
        sys**0.5
    assert sys**0 == LinearSystem(1)

    # Test inversion
    inv = ~sys
    assert inv == ([tau, 1], 1)
    assert not inv.causal

    assert inv == 1 / sys
    assert inv == sys**(-1)

    # Test repr/str
    copy = _eval(sys)
    assert copy == sys

    # Test addition/subtraction
    assert sys + 2 == ((2*tau, 3), (tau, 1))
    assert 3 + sys == (-sys)*(-1) + 3
    assert (4 - sys) + 2 == (-sys) + 6
    assert np.allclose((sys - sys).num, 0)

    # Test division
    assert sys / 2 == sys * 0.5
    assert 2 / sys == 2 * inv

    cancel = sys / sys
    assert np.allclose(cancel.num, cancel.den)

    # Test inequality
    assert sys != (sys*2)

    # Test usage of differential building block
    assert sys == 1 / (tau*s + 1)


def test_linear_system_type():
    # Test that sys1 is reused by sys2
    sys1 = LinearSystem(1)
    sys2 = LinearSystem(sys1)
    sys3 = LinearSystem(1)

    assert sys1 is sys2
    assert sys1 is not sys3

    # Test that sys1 is still reused even with weird arg/kwarg ordering
    sys4 = LinearSystem(sys1, analog=True)
    sys5 = LinearSystem(sys=sys1, analog=True)
    sys6 = LinearSystem(analog=True, sys=sys1)

    assert sys1 is sys4
    assert sys1 is sys5
    assert sys1 is sys6

    # Test that analog argument gets inherited properly
    assert LinearSystem(s).analog
    assert LinearSystem(s, analog=True).analog
    assert not LinearSystem(z).analog
    assert not LinearSystem(z, analog=False).analog
    assert LinearSystem(nengo.Lowpass(0.1)).analog
    assert not LinearSystem(LinearSystem(([1], [1]), analog=False)).analog

    # Test that analog argument must match
    with pytest.raises(TypeError):
        LinearSystem(sys1, analog=False)

    with pytest.raises(TypeError):
        LinearSystem(sys1, False)

    with pytest.raises(TypeError):
        LinearSystem(sys1, analog=False)

    with pytest.raises(TypeError):
        LinearSystem(s, analog=False)

    with pytest.raises(TypeError):
        LinearSystem(z, analog=True)

    with pytest.raises(TypeError):
        LinearSystem(LinearSystem(([1], [1]), analog=True), analog=False)

    with pytest.raises(TypeError):
        LinearSystem(LinearSystem(([1], [1]), analog=False), analog=True)


def test_invalid_operations():
    with pytest.raises(ValueError):
        z + s

    with pytest.raises(ValueError):
        s - z

    with pytest.raises(ValueError):
        z * s

    with pytest.raises(ValueError):
        z / s


def test_hashing():
    assert len(set((z, s))) == 2
    assert len(set((s, 5*s/5))) == 1


def test_zerodim_system():
    sys = LinearSystem(1)
    assert len(sys) == 0
    assert ss_equal(sys, (0, 0, 0, 1))

    # However, this following system could have dimension 0 or 1
    # depending on whether we're before or after scipy 0.18
    # see https://github.com/scipy/scipy/issues/5760
    # TODO: ideally it would stay 0, but documenting this weirdness for now
    ss_sys = LinearSystem(sys.ss)
    assert len(ss_sys) in (0, 1)


def test_similarity_transform():
    sys = Alpha(0.1)

    TA, TB, TC, TD = sys.transform(np.eye(2), np.eye(2)).ss
    A, B, C, D = sys2ss(sys)
    assert np.allclose(A, TA)
    assert np.allclose(B, TB)
    assert np.allclose(C, TC)
    assert np.allclose(D, TD)

    T = [[1, 1], [-0.5, 0]]
    rsys = sys.transform(T)
    assert ss_equal(rsys, sys.transform(T, inv(T)))

    TA, TB, TC, TD = rsys.ss
    assert not np.allclose(A, TA)
    assert not np.allclose(B, TB)
    assert not np.allclose(C, TC)
    assert np.allclose(D, TD)
    assert sys_equal(sys, (TA, TB, TC, TD))

    length = 1000
    dt = 0.001
    x_old = np.asarray(
        [sub.impulse(length=length, dt=dt) for sub in sys])
    x_new = np.asarray(
        [sub.impulse(length=length, dt=dt) for sub in rsys])

    # dot(T, x_new(t)) = x_old(t)
    assert np.allclose(np.dot(T, x_new), x_old)


def test_impulse():
    dt = 0.001
    tau = 0.005
    length = 500

    delta = np.zeros(length)
    delta[0] = 1. / dt

    sys = Lowpass(tau)
    response = sys.impulse(length, dt)
    assert not np.allclose(response[0], 0)

    # should give the same result as using filt
    assert np.allclose(response, sys.filt(delta, dt))

    # and should default to the same dt
    assert sys.default_dt == dt
    assert np.allclose(response, sys.impulse(length))

    # should also accept discrete systems
    dss = cont2discrete(sys, dt=dt)
    assert not dss.analog
    assert np.allclose(response, dss.impulse(length) / dt)
    assert np.allclose(response, dss.impulse(length, dt=dt))


def test_impulse_dt():
    length = 1000
    sys = Alpha(0.1)
    # the dt should not alter the magnitude of the response
    assert np.allclose(
        max(sys.impulse(length, dt=0.001)),
        max(sys.impulse(length, dt=0.0005)),
        atol=1e-4)


def test_invalid_impulse():
    with pytest.raises(ValueError):
        s.impulse(length=10, dt=None)  # must be digital if dt is None
