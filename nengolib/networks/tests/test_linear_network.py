import numpy as np
import pytest

import nengo
from nengo.utils.testing import warns

from nengolib.networks.linear_network import LinearNetwork
from nengolib import Network
from nengolib.signal import (
    apply_filter, impulse, s, Controllable, decompose_states)
from nengolib.synapses import PadeDelay


@pytest.mark.parametrize("neuron_type,atol", [(nengo.neurons.Direct(), 1e-14),
                                              (nengo.neurons.LIFRate(), 5e-01),
                                              (nengo.neurons.LIF(), 1e-01)])
def test_linear_network(neuron_type, atol, Simulator, plt, seed, rng):
    n_neurons = 500
    dt = 0.001
    T = 1.0

    sys = nengo.Lowpass(0.1)
    scale_input = 2.0

    tau_probe = 0.005

    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        subnet = LinearNetwork(
            sys, n_neurons, synapse=0.02, dt=dt, neuron_type=neuron_type)
        nengo.Connection(
            stim, subnet.input, synapse=None, transform=scale_input)

        assert subnet.synapse == subnet.input_synapse

        p_stim = nengo.Probe(subnet.input, synapse=tau_probe)
        p_x = nengo.Probe(subnet.x.output, synapse=tau_probe)
        p_output = nengo.Probe(subnet.output, synapse=tau_probe)

    sim = Simulator(model, dt=dt)
    sim.run(T)

    expected = apply_filter(sys, dt, sim.data[p_stim], axis=0)

    plt.plot(sim.trange(), sim.data[p_output], label="Actual", alpha=0.5)
    plt.plot(sim.trange(), sim.data[p_x], label="x", alpha=0.5)
    plt.plot(sim.trange(), expected, label="Expected", alpha=0.5)
    plt.legend()

    assert np.allclose(sim.data[p_output], expected, atol=atol)


def test_unfiltered(Simulator, seed, rng):
    dt = 0.001
    T = 1.0

    sys = nengo.Alpha(0.1)
    synapse = 0.01

    with Network(seed=seed) as model:
        stim = nengo.Node(
            output=nengo.processes.WhiteSignal(T, high=10, seed=seed))
        subnet = LinearNetwork(
            sys, 1, synapse=synapse, input_synapse=None, dt=dt,
            neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)

        assert subnet.input_synapse is None

        p_ideal = nengo.Probe(subnet.input, synapse=sys)
        p_output = nengo.Probe(subnet.output, synapse=synapse)

    sim = Simulator(model, dt=dt)
    sim.run(T)

    assert np.allclose(sim.data[p_output], sim.data[p_ideal])


def test_expstable():
    with warns(UserWarning):
        with Network():
            LinearNetwork(
                ~s, 1, synapse=0.02, dt=0.001, normalizer=Controllable())


def test_radii(Simulator, seed, plt):
    sys = PadeDelay(3, 4, 0.1)
    dt = 0.001
    T = 0.3

    plt.figure()

    # Precompute the exact bounds for an impulse stimulus (note: this by
    # default will use the canonical controllable realization).
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
        subnet = LinearNetwork(sys, n_neurons=1, synapse=0.2, dt=dt,
                               radii=radii, normalizer=Controllable(),
                               neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)
        p = nengo.Probe(subnet.x.output, synapse=None)

    assert subnet.info == {}

    sim = nengo.Simulator(model, dt=dt)
    sim.run(T)

    plt.plot(sim.data[p], lw=5, alpha=0.5)

    assert np.allclose(np.max(abs(sim.data[p]), axis=0), 1, atol=1e-4)
