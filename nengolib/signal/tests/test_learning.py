import numpy as np
import pytest

import nengo

from nengolib.compat import get_activities
from nengolib.signal.learning import pes_learning_rate
from nengolib import Network


def test_pes_learning_rate(Simulator, plt, seed):
    n = 50
    dt = 0.0005
    T = 1.0
    initial = 0.7
    desired = -0.9
    epsilon = 1e-3  # get to factor epsilon with T seconds

    # Get activity vector and initial decoders
    with Network(seed=seed) as model:
        x = nengo.Ensemble(
            n, 1, seed=seed, neuron_type=nengo.neurons.LIFRate())
        y = nengo.Node(size_in=1)
        conn = nengo.Connection(x, y, synapse=None)

    sim = Simulator(model, dt=dt)
    a = get_activities(sim.model, x, [initial])
    d = sim.data[conn].weights
    assert np.any(a > 0)

    # Use util function to calculate learning_rate
    init_error = float(desired - np.dot(d, a))
    learning_rate, gamma = pes_learning_rate(
        epsilon / abs(init_error), a, T, dt)

    # Build model with no filtering on any connections
    with model:
        stim = nengo.Node(output=initial)
        ystar = nengo.Node(output=desired)

        conn.learning_rule_type = nengo.PES(
            pre_tau=1e-15, learning_rate=learning_rate)

        nengo.Connection(stim, x, synapse=None)
        nengo.Connection(ystar, conn.learning_rule, synapse=0, transform=-1)
        nengo.Connection(y, conn.learning_rule, synapse=0)

        p = nengo.Probe(y, synapse=None)
        decoders = nengo.Probe(conn, 'weights', synapse=None)

    sim = Simulator(model, dt=dt)
    sim.run(T)

    # Check that the final error is exactly epsilon
    assert np.allclose(abs(desired - sim.data[p][-1]), epsilon)

    # Check that all of the errors are given exactly by gamma**k
    k = np.arange(len(sim.trange()))
    error = init_error * gamma ** k
    assert np.allclose(sim.data[p].flatten(), desired - error)

    # Check that all of the decoders are equal to their analytical solution
    dk = d.T + init_error * a.T[:, None] * (1 - gamma ** k) / np.dot(a, a.T)
    assert np.allclose(dk, np.squeeze(sim.data[decoders].T))

    plt.figure()
    plt.plot(sim.trange(), sim.data[p], lw=5, alpha=0.5)
    plt.plot(sim.trange(), desired - error, linestyle='--', lw=5, alpha=0.5)


def test_pes_learning_rate_fail():
    with pytest.raises(ValueError):
        pes_learning_rate(0.1, [[1, 2], [3, 4]], 1)
