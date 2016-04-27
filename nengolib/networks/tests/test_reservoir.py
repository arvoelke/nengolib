import numpy as np
import pytest

import nengo
from nengo.exceptions import NetworkContextError
from nengo.processes import WhiteSignal
from nengo.utils.numpy import rmse
from nengo.utils.testing import warns

from nengolib.networks.reservoir import Reservoir

from nengolib import Network, Lowpass


def test_basic(Simulator, seed, plt):
    # solve for a standard nengo connection using a feed-forward reservoir

    train_t = 5.0
    test_t = 0.5
    dt = 0.001

    n_neurons = 100
    synapse = 0.01

    process = WhiteSignal(max(train_t, test_t), high=10, rms=0.3)

    def function(x): return x**2

    with Network() as model:
        ens = nengo.Ensemble(n_neurons, 1, seed=seed)  # <- must have seed!
        res = Reservoir(ens, ens.neurons, synapse)

        # Solve for the readout that approximates a function of the *filtered*
        # stimulus. We include a lowpass here because the final RMSE will be
        # with respect to the lowpass stimulus, which is also consistent
        # with what the NEF is doing. But in a general recurrent reservoir
        # this filter could hypothetically be anything.
        res.train(
            lambda x: function(Lowpass(synapse).filt(x, dt=dt)),
            train_t, dt, process, seed=seed+1)

        assert res.size_in == 1
        assert res.size_mid == n_neurons
        assert res.size_out == 1

        # Validation
        _, (_, _, check_output) = res.run(
            test_t, dt, process, seed=seed+2)

        stim = nengo.Node(output=process)
        output = nengo.Node(size_in=1)

        nengo.Connection(stim, ens, synapse=None)
        nengo.Connection(ens, output, function=function, synapse=None)

        # note the reservoir output already includes a synapse
        p_res = nengo.Probe(res.output, synapse=None)
        p_normal = nengo.Probe(output, synapse=synapse)
        p_stim = nengo.Probe(stim, synapse=synapse)

    with Simulator(model, dt=dt, seed=seed+2) as sim:
        sim.run(test_t)

    # Since the seed for the two test processes were the same, the validation
    # run should produce the same output as the test simulation.
    assert np.allclose(check_output, sim.data[p_res])

    ideal = function(sim.data[p_stim])

    plt.figure()
    plt.plot(sim.trange(), sim.data[p_res], label="Reservoir")
    plt.plot(sim.trange(), sim.data[p_normal], label="Standard")
    plt.plot(sim.trange(), ideal, label="Ideal")
    plt.legend()

    assert np.allclose(rmse(sim.data[p_res], ideal),
                       rmse(sim.data[p_normal], ideal), atol=1e-2)


def test_multiple(Simulator, seed, plt):
    # check that multiple dimensions go to multiple input nodes

    train_t = 2.0
    test_t = 0.5
    dt = 0.001

    process = WhiteSignal(max(train_t, test_t), high=10, rms=0.3)

    w = [1, -1, 0.6, -0.3]
    function = (lambda x: w[0]*x[:, 0] + w[1]*x[:, 1] + w[2]*x[:, 2] +
                w[3]*x[:, 1]*x[:, 2])

    with Network() as model:
        stim = nengo.Node(output=process, size_out=3)

        # isolate the objects within the reservoir so that we can train
        # it even after connecting a node into it
        with Network():
            a = nengo.Node(size_in=1)
            b = nengo.Node(size_in=2)
            c = nengo.Node(size_in=2, output=lambda t, x: x[0]*x[1])
            nengo.Connection(b, c, synapse=None)

            p_a = nengo.Probe(a, synapse=None)
            p_b = nengo.Probe(b, synapse=None)
            p_c = nengo.Probe(c, synapse=None)

            res = Reservoir([a, b], [a, b, c])

        nengo.Connection(stim[0], a, synapse=None)
        nengo.Connection(stim[1:], b, synapse=None)
        p = nengo.Probe(res.output, synapse=None)
        p_stim = nengo.Probe(stim, synapse=None)

    d, info = res.train(
        function, train_t, dt, process, seed=seed,
        solver=nengo.solvers.LstsqL2(reg=1e-6))

    assert np.allclose(np.squeeze(d), w)
    assert res.size_in == 3
    assert res.size_mid == 4
    assert res.size_out == 1

    sim, data_in, data_mid = info['sim'], info['data_in'], info['data_mid']

    assert np.allclose(sim.data[p_a], data_in[:, 0:1])
    assert np.allclose(sim.data[p_b], data_in[:, 1:])
    assert np.allclose(sim.data[p_c], data_in[:, 1:2] * data_in[:, 2:])
    assert np.allclose(
        np.hstack((sim.data[p_a], sim.data[p_b], sim.data[p_c])), data_mid)

    with Simulator(model, dt=dt, seed=seed+1) as sim:
        sim.run(test_t)

    ideal = function(sim.data[p_stim])

    plt.figure()
    plt.plot(sim.trange(), sim.data[p_a], label="a")
    plt.plot(sim.trange(), sim.data[p_b], label="b")
    plt.plot(sim.trange(), sim.data[p_c], label="c")
    plt.plot(sim.trange(), sim.data[p], label="Output")
    plt.plot(sim.trange(), ideal, label="Ideal")
    plt.legend()

    assert np.allclose(np.squeeze(sim.data[p]), ideal)


def test_context():
    with Network():
        # make some arbitrary object for the reservoirs. it's okay that it's
        # not within the same network, because we won't simulate it
        a = nengo.Node(size_in=1)

    with Network() as outer:
        with Network() as inner:
            assert Reservoir(a, a).network is inner
            assert Reservoir(a, a, network=inner).network is inner
            assert Reservoir(a, a, network=outer).network is outer
        assert Reservoir(a, a).network is outer
        assert Reservoir(a, a, network=inner).network is inner
        assert Reservoir(a, a, network=outer).network is outer

    with pytest.raises(NetworkContextError):
        Reservoir(a, a)
    assert Reservoir(a, a, network=inner).network is inner
    assert Reservoir(a, a, network=outer).network is outer


def test_seed_warning():
    with Network():
        ens = nengo.Ensemble(100, 1)
        res = Reservoir(ens, ens)
        with warns(UserWarning):
            res.train(lambda x: x, 1.0, 0.001, WhiteSignal(1.0, high=10))


def test_bad_inputs_outputs():
    with Network():
        only_out = nengo.Node(output=[0])
        no_out = nengo.Node(size_out=0)
        okay = nengo.Ensemble(1, 1)
        bad = nengo.Network()

        with pytest.raises(ValueError):  # must contain at least one input
            Reservoir([], okay)

        with pytest.raises(ValueError):  # must contain at least one output
            Reservoir(okay, [])

        with pytest.raises(ValueError):  # must contain at least one input
            Reservoir(only_out, okay)

        with pytest.raises(ValueError):  # must contain at least one output
            Reservoir(okay, no_out)

        with pytest.raises(TypeError):  # must be connectable object
            Reservoir(bad, okay)

        with pytest.raises(TypeError):  # must be connectable object
            Reservoir(okay, bad)


def test_bad_function():
    with Network():
        a = nengo.Node(size_in=1)
        res = Reservoir(a, a)
        with pytest.raises(RuntimeError):  # expected signal length 1000
            res.train(lambda x: 0, 1.0, 0.001, WhiteSignal(1.0, high=10))
