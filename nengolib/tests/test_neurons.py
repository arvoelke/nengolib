import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib import Network, PerfectLIF


def _test_lif(Simulator, seed, neuron_type, u, dt, n=500, t=2.0):
    with Network(seed=seed) as model:
        stim = nengo.Node(u)
        x = nengo.Ensemble(n, 1, neuron_type=neuron_type)
        nengo.Connection(stim, x, synapse=None)
        p = nengo.Probe(x.neurons)

    sim = Simulator(model, dt=dt)
    sim.run(t)

    expected = nengo.builder.ensemble.get_activities(sim.model, x, [u]) * t
    actual = (sim.data[p] > 0).sum(axis=0)

    return rmse(actual, expected, axis=0)


def test_perfect_lif_performance(Simulator, seed, logger):
    for dt in np.linspace(0.0005, 0.002, 3):
        for u in np.linspace(-1, 1, 6):
            error_lif = _test_lif(Simulator, seed, nengo.LIF(), u, dt)
            error_perfect_lif = _test_lif(Simulator, seed, PerfectLIF(), u, dt)
            logger.info("dt=%s, u=%s: %s <? %s (ratio=%s)",
                        dt, u, error_perfect_lif, error_lif,
                        error_perfect_lif / error_lif)
            assert error_perfect_lif < 1
            assert error_perfect_lif < error_lif
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
