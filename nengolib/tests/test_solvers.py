import pytest

import numpy as np

import nengo

from nengolib.solvers import BiasedSolver
from nengolib import Network


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
