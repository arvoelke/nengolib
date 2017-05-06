import pytest

import numpy as np

import nengo

from nengolib.solvers import BiasedSolver
from nengolib import Network


@pytest.mark.parametrize("d", [1, 2, 10])
def test_biased_solver(Simulator, d):

    solver = BiasedSolver()
    function = solver.bias_function(d)

    assert solver.bias is None
    assert np.allclose(function(np.ones(d)), np.zeros(d))

    with Network() as model:
        x = nengo.Ensemble(100, d)
        conn = nengo.Connection(x, x, solver=solver)

    sim = Simulator(model)
    bias = sim.data[conn].solver_info['bias']

    assert np.allclose(solver.bias, bias)
    assert np.allclose(function(np.ones(d)), bias)

    # note: functional testing of this solver is in test_connection.py


def test_biased_solver_weights(Simulator):

    solver = BiasedSolver(nengo.solvers.LstsqL2(weights=True))
    assert solver.weights

    with Network() as model:
        x = nengo.Ensemble(100, 1)
        nengo.Connection(x, x, solver=solver)

    Simulator(model)
