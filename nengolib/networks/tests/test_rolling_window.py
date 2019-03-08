import pytest

import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.processes import WhiteSignal
from nengo.utils.numpy import rms

from nengolib.networks.rolling_window import (
    _pade_readout, _legendre_readout, t_default, RollingWindow)
from nengolib import Network
from nengolib.signal import (LinearSystem, shift, nrmse, Identity,
                             EvalPoints, Encoders)
from nengolib.stats import cube
from nengolib.synapses import PadeDelay, LegendreDelay


@pytest.mark.parametrize("legendre", [True, False])
@pytest.mark.parametrize("d,tol", [
    (2, 0.5), (3, 0.2), (4, 0.05), (5, 1e-2), (7, 1e-4), (12, 1e-9)])
def test_readout(legendre, d, tol):
    theta = 0.1

    if legendre:
        sys = LegendreDelay(theta, d)
        readout = _legendre_readout
    else:
        sys = PadeDelay(theta, d)
        readout = _pade_readout
    # decoding at r=1 (t=-theta) is equivalent to decoding a delay of theta
    C = readout(d, 1)
    assert np.allclose(sys.C, C)
    assert np.allclose(sys.D, 0)

    freqs = np.linspace(0, 5, 100)
    s = 2j * np.pi * freqs

    # check that frequency response has small error at low-frequencies
    # for a variety of different readouts
    for r in np.linspace(0, 1, 100):
        C = readout(d, r)

        sys = LinearSystem((sys.A, sys.B, C, sys.D), analog=True)
        error = np.abs(sys(s) - np.exp(-r*theta*s))
        assert 0 < rms(error) < tol, r


@pytest.mark.parametrize("legendre", [True, False])
def test_direct_window(legendre, Simulator, seed, plt):
    theta = 1.0
    T = theta

    assert np.allclose(t_default, np.linspace(0, 1, 1000))

    with Network() as model:
        stim = nengo.Node(output=lambda t: t)
        rw = RollingWindow(theta, n_neurons=1, dimensions=12,
                           neuron_type=nengo.Direct(), process=None,
                           legendre=legendre)
        assert rw.theta == theta
        assert rw.dt == 0.001
        assert rw.process is None
        assert rw.synapse == nengo.Lowpass(0.1)
        assert rw.input_synapse == nengo.Lowpass(0.1)

        nengo.Connection(stim, rw.input, synapse=None)
        output = rw.add_output(function=lambda w: np.sum(w**3)**2)
        p_output = nengo.Probe(output, synapse=None)

    with Simulator(model) as sim:
        sim.run(T)

    actual = sim.data[p_output].squeeze()
    t = sim.trange()
    ideal = shift(np.cumsum(t**3)**2)

    plt.figure()
    plt.plot(t, actual, label="Output")
    plt.plot(t, ideal, label="Ideal", linestyle='--')
    plt.legend()

    assert nrmse(actual, ideal) < 0.005


@pytest.mark.parametrize("legendre", [True, False])
def test_basis(legendre, plt):
    theta = 0.1
    d = 12

    rw = RollingWindow(theta=theta, n_neurons=1, dimensions=d, radii=2,
                       realizer=Identity(), legendre=legendre, process=None)

    B = rw.basis()
    assert B.shape == (len(t_default), d)

    # since using the Identity realizer
    assert np.allclose(B, rw.canonical_basis()*rw.radii)

    Binv = rw.inverse_basis()
    assert Binv.shape == (d, len(t_default))

    assert np.allclose(Binv.dot(B), np.eye(d))

    x = rw.sys.X.impulse(100)
    assert np.allclose(Binv.dot(B.dot(x.T)), x.T)

    plt.subplot(2, 1, 1)
    plt.plot(B)
    plt.subplot(2, 1, 2)
    plt.plot(Binv.T)


@pytest.mark.parametrize("legendre", [True, False])
def test_window_example(legendre, Simulator, seed, plt):
    theta = 0.1
    n_neurons = 1000
    d = 6
    high = 10

    T = 1.0
    dt = 0.002
    tau_probe = None

    # we set the radii here for testing on nengo<2.4.0
    # (see warning in RollingWindow._make_core)
    radii = 0.3

    with Network(seed=seed) as model:
        stim = nengo.Node(output=WhiteSignal(T, high=high, seed=seed))

        rw_rate = RollingWindow(
            theta=theta, n_neurons=n_neurons, dimensions=d, radii=radii,
            neuron_type=nengo.LIFRate(), dt=dt, process=stim.output,
            legendre=legendre)
        assert isinstance(rw_rate.state.eval_points, EvalPoints)
        assert isinstance(rw_rate.state.encoders, Encoders)

        rw_drct = RollingWindow(
            theta=theta, n_neurons=1, dimensions=d, radii=radii,
            neuron_type=nengo.Direct(), dt=dt, process=None,
            legendre=legendre)

        def function(w):
            return -np.max(np.abs(w)), abs(w[0]*w[-1])

        nengo.Connection(stim, rw_rate.input, synapse=None)
        nengo.Connection(stim, rw_drct.input, synapse=None)

        delay_rate = rw_rate.add_output(t=1)
        delay_drct = rw_drct.output  # rw_drct.add_output('delay', t=1)

        output_rate = rw_rate.add_output(function=function)
        output_drct = rw_drct.add_output(function=function)

        p_stim = nengo.Probe(stim, synapse=tau_probe)
        p_delay_rate = nengo.Probe(delay_rate, synapse=tau_probe)
        p_delay_drct = nengo.Probe(delay_drct, synapse=tau_probe)
        p_output_rate = nengo.Probe(output_rate, synapse=tau_probe)
        p_output_drct = nengo.Probe(output_drct, synapse=tau_probe)
        p_state_rate = nengo.Probe(rw_rate.state, synapse=tau_probe)
        p_state_drct = nengo.Probe(rw_drct.state, synapse=tau_probe)

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    plt.subplot(3, 1, 1)
    plt.plot(sim.trange(), sim.data[p_stim], label="Input")
    plt.plot(sim.trange(), sim.data[p_delay_rate], label="Rate")
    plt.plot(sim.trange(), sim.data[p_delay_drct], label="Direct",
             linestyle='--')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(sim.trange(), sim.data[p_output_rate], label="Rate")
    plt.plot(sim.trange(), sim.data[p_output_drct], label="Direct",
             linestyle='--')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(sim.trange(), sim.data[p_state_rate], label="Rate", lw=1)
    plt.plot(sim.trange(), sim.data[p_state_drct], label="Direct",
             linestyle='--')

    assert nrmse(sim.data[p_delay_rate], sim.data[p_delay_drct]) < 0.05
    assert nrmse(sim.data[p_output_rate], sim.data[p_output_drct]) < 0.2
    assert nrmse(sim.data[p_state_rate], sim.data[p_state_drct]) < 0.05


def test_window_function():
    process = WhiteSignal(1.0, 10)
    with pytest.raises(ValidationError):
        RollingWindow(theta=1.0, n_neurons=1, process=process,
                      eval_points=cube)

    rw = RollingWindow(theta=1.0, n_neurons=1, process=None)
    with pytest.raises(ValidationError):
        rw.add_output(function=lambda a, b: a)
