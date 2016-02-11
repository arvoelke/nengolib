from __future__ import division

import numpy as np
import pytest
from scipy.signal import cont2discrete

import nengo
from nengo.synapses import filt

from nengolib.signal.system import (
    sys2ss, sys2zpk, sys2tf, canonical, sys_equal, impulse, is_exp_stable,
    scale_state, LinearSystem, s)
from nengolib import Network, Lowpass, Alpha, LinearFilter
from nengolib.signal import state_norm


def test_sys_conversions():
    sys = Alpha(0.1)

    tf = sys2tf(sys)
    ss = sys2ss(sys)

    assert sys_equal(sys2ss(tf), ss)
    assert sys_equal(sys2ss(ss), ss)  # unchanged
    assert sys_equal(sys2tf(tf), tf)  # unchanged
    assert sys_equal(sys2tf(ss), tf)

    zpk = sys2zpk(ss)
    assert sys_equal(sys2zpk(zpk), zpk)  # sanity check
    assert sys_equal(sys2zpk(tf), zpk)  # sanity check
    assert sys_equal(sys2tf(zpk), tf)
    assert sys_equal(sys2ss(zpk), ss)

    # should also work with nengo's synapse types
    assert sys_equal(sys2zpk(nengo.Alpha(0.1)), zpk)
    assert sys_equal(sys2tf(nengo.Alpha(0.1)), tf)
    assert sys_equal(sys2ss(nengo.Alpha(0.1)), ss)

    # system can also be just a scalar
    assert sys_equal(sys2tf(2.0), (1, 0.5))
    assert sys_equal(sys2tf(sys2ss(5)), (5, 1))

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
    assert not sys_equal(np.zeros(2), np.zeros(3))


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
    assert sys_equal(canonical(sys), sys)


def test_impulse():
    dt = 0.001
    tau = 0.005
    length = 10

    delta = np.zeros(length)  # TODO: turn into a little helper?
    delta[1] = 1.0

    sys = Lowpass(tau)
    response = impulse(sys, dt, length)
    np.allclose(response, filt(delta, sys, dt))

    dss = cont2discrete(sys2ss(sys), dt=dt)[:-1]
    np.allclose(response, impulse(dss, None, length))


def test_is_exp_stable():
    sys = Lowpass(0.1)
    assert is_exp_stable(sys)

    sys = LinearFilter([1], [1, 0])  # integrator
    assert not is_exp_stable(sys)


def test_scale_state():
    syn = Lowpass(0.1)
    ss = sys2ss(syn)
    scaled = scale_state(*ss, radii=2)

    # Check that it's still the same stystem, even though different matrices
    assert sys_equal(ss, scaled)
    assert not np.allclose(ss[1], scaled[1])

    # Check that the state vectors have half the power
    assert np.allclose(
        state_norm(ss, analog=True)/2, state_norm(scaled, analog=True))


def test_invalid_scale_state():
    syn = Lowpass(0.1)
    ss = sys2ss(syn)

    scale_state(*ss, radii=[1])

    with pytest.raises(ValueError):
        scale_state(*ss, radii=[[1]])

    with pytest.raises(ValueError):
        scale_state(*ss, radii=[1, 2])


@pytest.mark.parametrize("sys", [
    Lowpass(0.01), Alpha(0.2), LinearFilter([1, 1], [0.01, 1])])
def test_simulation(sys, Simulator, plt):
    assert isinstance(sys, LinearSystem)
    old_sys = nengo.LinearFilter(sys.num, sys.den)
    assert sys == old_sys

    with Network() as model:
        stim = nengo.Node(output=nengo.processes.WhiteSignal(1.0))
        out_new = nengo.Node(size_in=2)
        out_old = nengo.Node(size_in=2)
        nengo.Connection(stim, out_new, transform=[[1], [-1]], synapse=sys)
        nengo.Connection(stim, out_old, transform=[[1], [-1]], synapse=old_sys)
        p_new = nengo.Probe(out_new)
        p_old = nengo.Probe(out_old)

    sim = Simulator(model)
    sim.run(1.0)

    plt.figure()
    plt.plot(sim.trange(), sim.data[p_new])
    plt.plot(sim.trange(), sim.data[p_old])

    assert np.allclose(sim.data[p_new], sim.data[p_old])


def test_sys_multiplication():
    # Check that alpha is just two lowpass multiplied together
    assert Lowpass(0.1) * Lowpass(0.1) == Alpha(0.1)


def test_sim_new_synapse(Simulator):
    # Create a new synapse object and simulate it
    with Network() as model:
        stim = nengo.Node(output=np.sin)
        x = nengo.Node(size_in=1)
        nengo.Connection(stim, x, synapse=Lowpass(0.1) - Lowpass(0.01))
    sim = Simulator(model)
    sim.run(0.1)


def test_linear_system():
    tau = 0.05
    sys = Lowpass(tau)

    # Test representations
    assert sys == (1, [tau, 1])
    assert sys_equal(sys.tf, sys)
    assert sys_equal(sys.ss, sys)

    # Test attributes
    assert np.allclose(sys.num, (1,))
    assert np.allclose(sys.den, (tau, 1))
    assert sys.causal
    assert not sys.has_passthrough
    assert not (sys/s).has_passthrough
    assert (sys*s).has_passthrough
    assert not (sys*s*s).has_passthrough and not (sys*s*s).causal
    assert (sys*s*s + sys*s).has_passthrough

    assert sys.order_num == 0
    assert sys.order_den == 1
    assert len(sys) == 1  # order_den

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
    copy = eval(
        repr(sys), {}, {'LinearSystem': LinearSystem, 'array': np.array})
    assert copy == sys
    assert str(copy) == str(sys)

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
