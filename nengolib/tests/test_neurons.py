import numpy as np
import pytest

from scipy.stats import kstest

import nengo
from nengo.utils.numpy import rmse

from nengolib.neurons import init_lif, Tanh
from nengolib import Network
from nengolib.compat import get_activities


@pytest.mark.parametrize("x0", [-1, -0.5, None, 0.5, 1])
def test_init_lif(Simulator, seed, x0):
    u = 0 if x0 is None else x0
    n_neurons = 1000
    t = 2.0
    ens_kwargs = dict(
        n_neurons=n_neurons,
        dimensions=1,
        max_rates=nengo.dists.Uniform(10, 100),
        seed=seed,
    )

    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(u)

        zero = nengo.Ensemble(**ens_kwargs)
        init = nengo.Ensemble(**ens_kwargs)

        nengo.Connection(stim, zero, synapse=None)
        nengo.Connection(stim, init, synapse=None)

        p_zero_spikes = nengo.Probe(zero.neurons, 'spikes', synapse=None)
        p_zero_v = nengo.Probe(zero.neurons, 'voltage', synapse=None)

        p_init_spikes = nengo.Probe(init.neurons, 'spikes', synapse=None)
        p_init_v = nengo.Probe(init.neurons, 'voltage', synapse=None)

    with Simulator(model, seed=seed) as sim:
        init_lif(sim, init, x0=x0)
        sim.run(t)

    # same tuning curves
    a = get_activities(sim.model, zero, [u])
    assert np.allclose(a, get_activities(sim.model, init, [u]))

    # calculate difference between actual spike counts and ideal
    count_zero = np.count_nonzero(sim.data[p_zero_spikes], axis=0)
    count_init = np.count_nonzero(sim.data[p_init_spikes], axis=0)
    e_zero = count_zero - a * t
    e_init = count_init - a * t

    # initialized error is close to zero, better than uninitialized,
    # with std. dev. close to the uninitialized
    assert np.abs(np.mean(e_init)) < 0.05
    assert np.abs(np.mean(e_zero)) > 0.1
    assert np.abs(np.std(e_init) - np.std(e_zero)) < 0.05

    # subthreshold neurons are the same between populations
    subthresh = np.all(sim.data[p_init_spikes] == 0, axis=0)
    assert np.allclose(subthresh,
                       np.all(sim.data[p_zero_spikes] == 0, axis=0))
    assert 0 < np.count_nonzero(subthresh) < n_neurons
    is_active = ~subthresh

    # uninitialized always under-counts (unless subthreshold)
    # the other exception is when a neuron spikes at the very end
    # since the simulation does not start in its refractory
    assert np.allclose(e_zero[subthresh], 0)
    very_end = sim.trange() >= t - init.neuron_type.tau_ref
    exception = np.any(
        sim.data[p_zero_spikes][very_end, :] > 0, axis=0)
    # no more than 10% should be exceptions (heuristic)
    assert np.count_nonzero(exception) < 0.1 * n_neurons
    assert np.all(e_zero[is_active & (~exception)] < 0)

    # uninitialized voltages start at 0 (plus first time-step)
    assert np.all(sim.data[p_zero_v][0, :] < 0.2)

    # initialized sub-threshold voltages remain constant
    # (steady-state input)
    assert np.allclose(sim.data[p_init_v][0, subthresh],
                       sim.data[p_init_v][-1, subthresh])

    def uniformity_test(spikes):
        # test uniformity of ISIs
        # returns (r, d, p) where r is the [0, 1) relative
        # position of the first spike-time within the ISI
        # d is the KS D-statistic which is the absolute max
        # distance from the uniform distribution, and p
        # is the p-value of this statistic
        t_spike = sim.trange()[
            [np.where(s > 0)[0][0] for s in spikes[:, is_active].T]]
        assert t_spike.shape == (np.count_nonzero(is_active),)
        isi_location = (t_spike - sim.dt) * a[is_active]
        return (isi_location,) + kstest(isi_location, 'uniform')

    r, d, p = uniformity_test(sim.data[p_init_spikes])
    assert np.all(r >= 0)
    assert np.all(r < 1)
    assert d < 0.1

    r, d, p = uniformity_test(sim.data[p_zero_spikes])
    assert np.all(r >= 0.7)  # heuristic
    assert np.all(r < 1)
    assert d > 0.7
    assert p < 1e-5


def test_init_lif_invalid(Simulator):
    with nengo.Network() as model:
        x = nengo.Ensemble(100, 1, neuron_type=nengo.Direct())

    with Simulator(model) as sim:
        with pytest.raises(ValueError):
            init_lif(sim, x)

    with nengo.Network() as model:
        x = nengo.Ensemble(100, 2)

    with Simulator(model) as sim:
        with pytest.raises(ValueError):
            init_lif(sim, x, x0=0)


def test_init_lif_rng(Simulator, seed):
    with nengo.Network() as model:
        x = nengo.Ensemble(100, 1)

    with Simulator(model) as sim:
        v1, r1 = init_lif(sim, x, rng=np.random.RandomState(seed=seed))
        v2, r2 = init_lif(sim, x, rng=np.random.RandomState(seed=seed))

    assert np.allclose(v1, v2)
    assert np.allclose(r1, r2)
    assert v1.shape == (x.n_neurons,)
    assert r1.shape == (x.n_neurons,)


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
            error_perfect_lif = _test_lif(Simulator, seed, nengo.LIF(), u, dt)
            assert error_perfect_lif < 1
            seed += 1


def test_perfect_lif_invariance(Simulator, seed):
    # as long as the simulation time is divisible by dt, the same number of
    # spikes should be observed
    t = 1.0
    errors = []
    for dt in (0.0001, 0.0005, 0.001, 0.002):
        assert np.allclose(int(t / dt), t / dt)
        error = _test_lif(Simulator, seed, nengo.LIF(), 0, dt, t=t)
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
