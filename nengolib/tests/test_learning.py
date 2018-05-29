import numpy as np
from scipy.linalg import inv

import nengo

from nengolib.learning import RLS
from nengolib import Network
from nengolib.compat import get_activities
from nengolib.learning import SimRLS
from nengolib.signal import nrmse


def test_rls_repr():
    assert repr(RLS()) == "RLS(learning_rate=1.0, pre_synapse=Lowpass(0.005))"
    assert (repr(RLS(learning_rate=1e-5)) ==
            "RLS(learning_rate=1e-05, pre_synapse=Lowpass(0.005))")
    assert (repr(RLS(pre_synapse=None)) ==
            "RLS(learning_rate=1.0, pre_synapse=None)")
    assert (repr(RLS(learning_rate=1e-2, pre_synapse=nengo.Lowpass(1e-4))) ==
            "RLS(learning_rate=0.01, pre_synapse=Lowpass(0.0001))")


def _test_RLS_network(Simulator, seed, dims, lrate, neuron_type, tau,
                      T_train, T_test, tols):
    # Input is a scalar sinusoid with given frequency
    n_neurons = 100
    freq = 5

    # Learn a linear transformation within T_train seconds
    transform = np.random.RandomState(seed=seed).randn(dims, 1)
    lr = RLS(learning_rate=lrate, pre_synapse=tau)

    with Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.sin(freq*2*np.pi*t))
        x = nengo.Ensemble(n_neurons, 1, neuron_type=neuron_type)
        y = nengo.Node(size_in=dims)
        y_on = nengo.Node(size_in=dims)
        y_off = nengo.Node(size_in=dims)

        e = nengo.Node(
            size_in=dims,
            output=lambda t, e: e if t < T_train else np.zeros_like(e))

        nengo.Connection(u, y, synapse=None, transform=transform)
        nengo.Connection(u, x, synapse=None)
        conn_on = nengo.Connection(
            x, y_on, synapse=None, learning_rule_type=lr,
            function=lambda _: np.zeros(dims))
        nengo.Connection(y, e, synapse=None, transform=-1)
        nengo.Connection(y_on, e, synapse=None)
        nengo.Connection(e, conn_on.learning_rule, synapse=tau)

        nengo.Connection(x, y_off, synapse=None, transform=transform)

        p_y = nengo.Probe(y, synapse=tau)
        p_y_on = nengo.Probe(y_on, synapse=tau)
        p_y_off = nengo.Probe(y_off, synapse=tau)
        p_inv_gamma = nengo.Probe(conn_on.learning_rule, 'inv_gamma')

    with Simulator(model) as sim:
        sim.run(T_train + T_test)

    # Check _descstr
    ops = [op for op in sim.model.operators if isinstance(op, SimRLS)]
    assert len(ops) == 1
    assert str(ops[0]).startswith('SimRLS')

    test = sim.trange() >= T_train

    on_versus_off = nrmse(
        sim.data[p_y_on][test], target=sim.data[p_y_off][test])

    on_versus_ideal = nrmse(
        sim.data[p_y_on][test], target=sim.data[p_y][test])

    off_versus_ideal = nrmse(
        sim.data[p_y_off][test], target=sim.data[p_y][test])

    A = get_activities(
        sim.model, x, np.linspace(-1, 1, 1000)[:, None])
    gamma_off = A.T.dot(A) + np.eye(n_neurons) / lr.learning_rate
    gamma_on = inv(sim.data[p_inv_gamma][-1])

    gamma_off /= np.linalg.norm(gamma_off)
    gamma_on /= np.linalg.norm(gamma_on)
    gamma_diff = nrmse(gamma_on, target=gamma_off)

    assert on_versus_off < tols[0]
    assert on_versus_ideal < tols[1]
    assert off_versus_ideal < tols[2]
    assert gamma_diff < tols[3]


def test_scalar_rate(Simulator, seed):
    _test_RLS_network(
        Simulator, seed, dims=1, lrate=1,
        neuron_type=nengo.LIFRate(), tau=None, T_train=1, T_test=0.5,
        tols=[0.02, 1e-3, 0.02, 0.3])


def test_multidim(Simulator, seed):
    _test_RLS_network(
        Simulator, seed, dims=11, lrate=1,
        neuron_type=nengo.LIFRate(), tau=None, T_train=1, T_test=0.5,
        tols=[0.02, 1e-3, 0.02, 0.3])


def test_scalar_spiking(Simulator, seed):
    _test_RLS_network(
        Simulator, seed, dims=1, lrate=1e-5,
        neuron_type=nengo.LIF(), tau=0.01, T_train=1, T_test=0.5,
        tols=[0.03, 0.04, 0.04, 0.3])
