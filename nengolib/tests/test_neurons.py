import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib import Network, PerfectLIF


def _test_lif(Simulator, seed, neuron_type, u, dt=0.001, n=5000,
              synapse=0.005, t=1.0):
    with Network(seed=seed) as model:
        stim = nengo.Node(u)
        x = nengo.Ensemble(n, 1, neuron_type=neuron_type)
        nengo.Connection(stim, x, synapse=synapse)
        p = nengo.Probe(x.neurons)

    sim = Simulator(model, dt=dt)
    sim.run(t)

    expected = nengo.builder.ensemble.get_activities(sim.model, x, [u])
    actual = (sim.data[p] > 0).sum(axis=0) / t

    return rmse(actual, expected)


def test_perfect_lif(Simulator, seed, logger):
    for u in np.linspace(-1, 1, 6):
        error_lif = _test_lif(Simulator, seed, nengo.LIF(), u)
        error_perfect_lif = _test_lif(Simulator, seed, PerfectLIF(), u)
        logger.info("%s <? %s (ratio=%s)", error_perfect_lif, error_lif,
                    error_perfect_lif / error_lif)
        assert error_perfect_lif < error_lif
