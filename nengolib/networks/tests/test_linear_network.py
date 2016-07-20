import numpy as np
import pytest

import nengo
from nengo.utils.testing import warns

from nengolib.networks.linear_network import LinearNetwork
from nengolib import Network
from nengolib.signal import (
    apply_filter, impulse, s, z, Controllable, canonical, decompose_states)
from nengolib.synapses import PureDelay, Bandpass


_mock_solver_calls = 0  # global to keep solver's fingerprint static


class MockSolver(nengo.solvers.LstsqL2):

    def __call__(self, A, Y, rng=None, E=None):
        globals()['_mock_solver_calls'] += 1
        return super(MockSolver, self).__call__(A, Y, rng=rng, E=E)


@pytest.mark.parametrize("neuron_type,atol", [(nengo.neurons.Direct(), 1e-14),
                                              (nengo.neurons.LIFRate(), 5e-01),
                                              (nengo.neurons.LIF(), 1e-01)])
def test_linear_network(neuron_type, atol, Simulator, plt, seed, rng):
    n_neurons = 500
    dt = 0.001
    T = 1.0

    sys = nengo.Lowpass(0.1)
    scale_input = 2.0

    synapse = 0.02
    tau_probe = 0.005

    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        subnet = LinearNetwork(
            sys, n_neurons, synapse=synapse, input_synapse=synapse, dt=dt,
            neuron_type=neuron_type)
        nengo.Connection(
            stim, subnet.input, synapse=None, transform=scale_input)

        assert subnet.synapse == subnet.input_synapse
        assert subnet.output_synapse is None

        p_stim = nengo.Probe(subnet.input, synapse=tau_probe)
        p_x = nengo.Probe(subnet.x.output, synapse=tau_probe)
        p_output = nengo.Probe(subnet.output, synapse=tau_probe)

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    expected = apply_filter(sys, dt, sim.data[p_stim], axis=0)

    plt.plot(sim.trange(), sim.data[p_output], label="Actual", alpha=0.5)
    plt.plot(sim.trange(), sim.data[p_x], label="x", alpha=0.5)
    plt.plot(sim.trange(), expected, label="Expected", alpha=0.5)
    plt.legend()

    assert np.allclose(sim.data[p_output], expected, atol=atol)


def test_none_dt(Simulator, seed, rng):
    dt = 0.0001  # approximates continuous case
    T = 1.0

    sys = Bandpass(8, 5)
    synapse = 0.01

    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        subnet = LinearNetwork(
            sys, 1, synapse=synapse, input_synapse=synapse, dt=None,
            neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)

        assert subnet.output_synapse is None

        p_ideal = nengo.Probe(subnet.input, synapse=sys)
        p_output = nengo.Probe(subnet.output, synapse=None)

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    assert np.allclose(sim.data[p_output], sim.data[p_ideal], atol=0.1)


def test_output_filter(Simulator, seed, rng):
    dt = 0.001
    T = 1.0

    sys = PureDelay(0.1, order=3, p=3)
    assert sys.has_passthrough
    synapse = 0.01

    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        subnet = LinearNetwork(
            sys, 1, synapse=synapse, output_synapse=synapse, dt=dt,
            neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)

        assert subnet.input_synapse is None

        p_ideal = nengo.Probe(subnet.input, synapse=sys)
        p_output = nengo.Probe(subnet.output, synapse=None)

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    assert np.allclose(sim.data[p_output][:-1], sim.data[p_ideal][1:])


def test_unfiltered(Simulator, seed, rng):
    dt = 0.001
    T = 1.0

    sys = nengo.Alpha(0.1)
    synapse = 0.01

    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        subnet = LinearNetwork(
            sys, 1, synapse=synapse, dt=dt,
            neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)

        assert subnet.input_synapse is None
        assert subnet.output_synapse is None

        p_ideal = nengo.Probe(subnet.input, synapse=sys)
        p_output = nengo.Probe(subnet.output, synapse=synapse)

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    assert np.allclose(sim.data[p_output], sim.data[p_ideal])


def test_unstable_warning():
    with warns(UserWarning):
        with Network():
            LinearNetwork(
                ~s, 1, synapse=0.02, dt=0.001, normalizer=Controllable())


def test_output_warning():
    with warns(UserWarning):
        LinearNetwork(([1, 1], [1, 1]), 1, synapse=1, dt=1)


def test_invalid_systems():
    with pytest.raises(ValueError):
        LinearNetwork(~z, 1, synapse=1, dt=1)

    with pytest.raises(ValueError):
        LinearNetwork(~s, 1, synapse=~z, dt=1)

    with pytest.raises(ValueError):
        LinearNetwork(1, 1, synapse=1, dt=1)


def test_radii(Simulator, seed, plt):
    sys = canonical(PureDelay(0.1, order=4))
    dt = 0.001
    T = 0.3

    plt.figure()

    # Precompute the exact bounds for an impulse stimulus
    radii = []
    for sub in decompose_states(sys):
        response = impulse(sub, dt=dt, length=int(T / dt))
        amplitude = np.max(abs(response))
        assert amplitude >= 1e-6  # otherwise numerical issues
        radii.append(amplitude)

        plt.plot(response / amplitude, linestyle='--')

    with Network(seed=seed) as model:
        # Impulse stimulus
        stim = nengo.Node(output=lambda t: 1 / dt if t <= dt else 0)

        # Set explicit radii for controllable realization
        subnet = LinearNetwork(sys, n_neurons=1, synapse=0.2,
                               input_synapse=0.2, dt=dt,
                               radii=radii, normalizer=Controllable(),
                               neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)
        p = nengo.Probe(subnet.x.output, synapse=None)

    assert subnet.info == {}

    sim = nengo.Simulator(model, dt=dt)
    sim.run(T)

    plt.plot(sim.data[p], lw=5, alpha=0.5)

    assert np.allclose(np.max(abs(sim.data[p]), axis=0), 1, atol=1e-3)


def test_solver(tmpdir, Simulator, seed, rng):
    assert _mock_solver_calls == 0

    for _ in range(3):
        model = LinearNetwork(
            nengo.Lowpass(0.1), 10, synapse=0.1, dt=0.001,
            solver=MockSolver(reg=rng.rand()), seed=seed)

        Simulator(model, model=nengo.builder.Model(
            decoder_cache=nengo.cache.DecoderCache(cache_dir=str(tmpdir))))

    # checks workaround for https://github.com/nengo/nengo/issues/1044
    assert _mock_solver_calls == 3
