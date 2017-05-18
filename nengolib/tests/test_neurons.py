import numpy as np
import pytest

import nengo
from nengo.utils.numpy import rmse

from nengolib import Network, PerfectLIF
from nengolib.compat import get_activities
from nengolib.neurons import Tanh


def _test_lif(Simulator, seed, neuron_type, u, dt, n=500, t=2.0):
    with Network(seed=seed) as model:
        stim = nengo.Node(u)
        x = nengo.Ensemble(n, 1, neuron_type=neuron_type)
        nengo.Connection(stim, x, synapse=None)
        p = nengo.Probe(x.neurons)

    with Simulator(model, dt=dt) as sim:
        sim.run(t)

    expected = get_activities(sim.model, x, [u]) * t
    actual = (sim.data[p] > 0).sum(axis=0)

    return rmse(actual, expected, axis=0)


def test_perfect_lif_performance(Simulator, seed):
    for dt in np.linspace(0.0005, 0.002, 3):
        for u in np.linspace(-1, 1, 6):
            error_perfect_lif = _test_lif(Simulator, seed, PerfectLIF(), u, dt)
            assert error_perfect_lif < 1
            seed += 1


def test_perfect_lif_invariance(Simulator, seed):
    # as long as the simulation time is divisible by dt, the same number of
    # spikes should be observed
    t = 1.0
    errors = []
    for dt in (0.0001, 0.0005, 0.001, 0.002):
        assert np.allclose(int(t / dt), t / dt)
        error = _test_lif(Simulator, seed, PerfectLIF(), 0, dt, t=t)
        errors.append(error)
    assert np.allclose(errors, errors[0])


def test_tanh(Simulator, seed):
    T = 0.1
    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        x = nengo.Ensemble(2, 1, neuron_type=Tanh())
        nengo.Connection(
            stim, x.neurons, transform=np.ones((2, 1)), synapse=None)
        p_stim = nengo.Probe(stim, synapse=None)
        p_x = nengo.Probe(x.neurons, synapse=None)

    with Simulator(model) as sim:
        sim.run(T)

    assert np.allclose(sim.data[x].gain, 1)
    assert np.allclose(sim.data[x].bias, 0)
    assert np.allclose(sim.data[p_x], np.tanh(sim.data[p_stim]))


def test_tanh_decoding(Simulator):
    with Network() as model:
        nengo.Probe(nengo.Ensemble(10, 1, neuron_type=Tanh()), synapse=None)
    with pytest.raises(NotImplementedError):
        Simulator(model)
