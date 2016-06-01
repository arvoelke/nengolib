import pytest

import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib import Connection, Network
from nengolib.solvers import BiasedSolver


@pytest.mark.parametrize("d", [1, 2])
def test_connection(Simulator, seed, d):
    with Network(seed=seed) as model:
        stim = nengo.Node(output=lambda t: np.sin(t*2*np.pi), size_out=d)
        x = nengo.Ensemble(5, d, neuron_type=nengo.LIFRate())
        default = nengo.Node(size_in=d)
        improved = nengo.Node(size_in=d)

        stim_conn = Connection(stim, x, synapse=None)
        default_conn = nengo.Connection(x, default)
        improved_conn = Connection(x, improved)

        p_default = nengo.Probe(default)
        p_improved = nengo.Probe(improved)
        p_stim = nengo.Probe(stim, synapse=0.005)

    assert not isinstance(stim_conn.solver, BiasedSolver)
    assert not isinstance(default_conn.solver, BiasedSolver)
    assert isinstance(improved_conn.solver, BiasedSolver)

    with Simulator(model) as sim:
        sim.run(1.0)

    assert (rmse(sim.data[p_default], sim.data[p_stim]) >
            rmse(sim.data[p_improved], sim.data[p_stim]))
