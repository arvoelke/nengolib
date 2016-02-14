from __future__ import division

import numpy as np
import pytest

import nengo

from nengolib.signal.system import (
    sys2ss, sys2zpk, sys2tf, canonical, sys_equal, is_exp_stable, scale_state,
    LinearSystem, s, z)
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
    csys = canonical(sys, controllable=True)
    osys = canonical(sys, controllable=False)

    A, B, C, D = csys.ss
    assert sys_equal(csys, sys)
    assert np.allclose(csys.A, [[-20, -100], [1, 0]])
    assert np.allclose(csys.B, [[1], [0]])
    assert np.allclose(csys.C, [[0, 100]])
    assert np.allclose(csys.D, [[0]])

    assert sys_equal(osys, sys)
    assert np.allclose(osys.A, [[-20, 1], [-100, 0]])
    assert np.allclose(osys.B, [[0], [100]])
    assert np.allclose(osys.C, [[1, 0]])
    assert np.allclose(osys.D, [[0]])


def test_is_exp_stable():
    sys = Lowpass(0.1)
    assert is_exp_stable(sys)

    sys = LinearFilter([1], [1, 0])  # integrator
    assert not is_exp_stable(sys)


def test_scale_state():
    sys = Lowpass(0.1)
    scaled = scale_state(sys, radii=2)
    assert not np.allclose(sys.B, scaled.B)
    assert not np.allclose(sys.C, scaled.C)

    # Check that it's still the same system, even though different matrices
    assert sys_equal(sys, scaled)

    # Check that the state vectors have half the power
    assert np.allclose(state_norm(sys)/2, state_norm(scaled))


def test_invalid_scale_state():
    sys = Lowpass(0.1)

    scale_state(sys, radii=[1])

    with pytest.raises(ValueError):
        scale_state(sys, radii=[[1]])

    with pytest.raises(ValueError):
        scale_state(sys, radii=[1, 2])


@pytest.mark.parametrize("sys", [
    Lowpass(0.01), Alpha(0.2), LinearFilter([1, 1], [0.01, 1])])
def test_simulation(sys, Simulator, plt):
    assert isinstance(sys, LinearSystem)
    old_sys = nengo.LinearFilter(sys.num, sys.den)
    assert sys == old_sys

    with Network() as model:
        stim = nengo.Node(output=nengo.processes.WhiteSignal(1.0, high=10))
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


def test_discrete_synapse(Simulator):
    # Test that discrete synapses are simulated properly
    delay_steps = 1000

    with Network() as model:
        stim = nengo.Node(output=np.sin)
        output = nengo.Node(size_in=1)
        nengo.Connection(stim, output, synapse=z**-delay_steps)
        p_stim = nengo.Probe(stim, synapse=None)
        p_output = nengo.Probe(output, synapse=None)

    sim = Simulator(model)
    sim.run(5.0)

    assert np.allclose(sim.data[p_output][delay_steps:],
                       sim.data[p_stim][:-delay_steps])


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
    assert sys.proper
    assert not sys.has_passthrough
    assert not (sys/s).has_passthrough
    assert (sys*s).has_passthrough
    assert (sys*s).causal
    assert not (sys*s).proper
    assert not (sys*s*s).has_passthrough and not (sys*s*s).causal
    assert (sys*s*s + sys*s).has_passthrough

    assert np.allclose(sys.A, -1/tau)
    assert np.allclose(sys.B, 1)
    assert np.allclose(sys.C, 1/tau)
    assert np.allclose(sys.D, 0)

    assert np.allclose(sys.zeros, [0])
    assert np.allclose(sys.poles, [-1/tau])
    assert np.allclose(sys.gain, 1/tau)

    assert sys.order_num == 0
    assert sys.order_den == 1
    assert len(sys) == 1  # order_den
    assert len(LinearSystem(sys.ss)) == 1  # uses state-space rep

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
    assert not LinearSystem(LinearFilter([1], [1], analog=False)).analog

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
        LinearSystem(LinearFilter([1], [1], analog=True), analog=False)

    with pytest.raises(TypeError):
        LinearSystem(LinearFilter([1], [1], analog=False), analog=True)


def test_invalid_operations():
    with pytest.raises(ValueError):
        z == s

    with pytest.raises(ValueError):
        s != z

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
