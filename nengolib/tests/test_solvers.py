import pytest

import numpy as np

import nengo

from nengolib.solvers import BiasedSolver, Temporal
from nengolib import Network
from nengolib.signal import nrmse


@pytest.mark.parametrize("d", [1, 2, 10])
def test_biased_solver(Simulator, seed, d):
    solver = BiasedSolver()
    function = solver.bias_function(d)

    assert solver.bias is None
    assert np.allclose(function(np.ones(d)), np.zeros(d))

    with Network(seed=seed) as model:
        x = nengo.Ensemble(100, d)
        conn = nengo.Connection(x, x, solver=solver)

    with Simulator(model) as sim:
        bias = sim.data[conn].solver_info['bias']

    assert np.allclose(solver.bias, bias)
    assert np.allclose(function(np.ones(d)), bias)

    # NOTE: functional testing of this solver is in test_connection.py


def test_biased_solver_weights(Simulator):
    solver = BiasedSolver(nengo.solvers.LstsqL2(weights=True))
    assert solver.weights

    with Network() as model:
        x = nengo.Ensemble(100, 1)
        nengo.Connection(x, x, solver=solver)

    Simulator(model)


def _test_temporal_solver(plt, Simulator, seed, neuron_type, tau, f, solver):
    dt = 0.002

    # we are cheating a bit here because we'll use the same training data as
    # test data. this makes the unit testing a bit simpler since it's more
    # obvious what will happen when comparing temporal to default
    t = np.arange(0, 0.2, dt)
    stim = np.sin(2*np.pi*10*t)
    function = (f(stim) if tau is None else
                nengo.Lowpass(tau).filt(f(stim), dt=dt))

    with Network(seed=seed) as model:
        u = nengo.Node(output=nengo.processes.PresentInput(stim, dt))
        x = nengo.Ensemble(100, 1, neuron_type=neuron_type)
        output_ideal = nengo.Node(size_in=1)

        post = dict(n_neurons=500, dimensions=1, neuron_type=nengo.LIFRate(),
                    seed=seed+1)
        output_temporal = nengo.Ensemble(**post)
        output_default = nengo.Ensemble(**post)

        nengo.Connection(u, output_ideal, synapse=tau, function=f)
        nengo.Connection(u, x, synapse=None)
        nengo.Connection(
            x, output_temporal, synapse=tau, eval_points=stim[:, None],
            function=function[:, None],
            solver=Temporal(synapse=tau, solver=solver))
        nengo.Connection(
            x, output_default, synapse=tau, eval_points=stim[:, None],
            function=f, solver=solver)

        p_ideal = nengo.Probe(output_ideal, synapse=None)
        p_temporal = nengo.Probe(output_temporal, synapse=None)
        p_default = nengo.Probe(output_default, synapse=None)

    with Simulator(model, dt) as sim:
        sim.run(t[-1])

    plt.plot(sim.trange(), sim.data[p_ideal] - sim.data[p_default],
             label="Default")
    plt.plot(sim.trange(), sim.data[p_ideal] - sim.data[p_temporal],
             label="Temporal")
    plt.legend()

    return (nrmse(sim.data[p_default], target=sim.data[p_ideal]) /
            nrmse(sim.data[p_temporal], target=sim.data[p_ideal]))


@pytest.mark.skipif(nengo.version.version_info < (2, 5, 0),
                    reason="requires nengo>=2.5.0")
def test_temporal_solver(plt, Simulator, seed):
    plt.subplot(3, 1, 1)
    for weights in (False, True):
        assert 1.2 < _test_temporal_solver(  # 1.5153... at dev time
            plt, Simulator, seed, nengo.LIF(), 0.005,
            lambda x: x, nengo.solvers.LstsqL2(weights=weights))

    # LIFRate has no internal dynamics, and so the two solvers
    # are actually numerically equivalent
    plt.subplot(3, 1, 2)
    assert np.allclose(1, _test_temporal_solver(
        plt, Simulator, seed, nengo.LIFRate(), None,
        lambda x: 1-2*x**2, nengo.solvers.LstsqL2()))

    # We'll need to overfit slightly (small reg) to see the improvement for
    # AdaptiveLIF (see thesis for a more principled way to improve)
    plt.subplot(3, 1, 3)
    assert 2.0 < _test_temporal_solver(  # 2.2838... at dev time
        plt, Simulator, seed, nengo.AdaptiveLIF(), 0.1,
        np.sin, nengo.solvers.LstsqL2(reg=1e-5))
