import numpy as np
import pytest

import nengo
from nengo.utils.testing import warns

from nengolib.networks.linear_network import LinearNetwork
from nengolib import Network
from nengolib.signal import apply_filter, s


@pytest.mark.parametrize("neuron_type,atol", [(nengo.neurons.Direct(), 1e-08),
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


def test_expstable():
    with warns(UserWarning):
        with Network():
            LinearNetwork(~s, 1, synapse=0.02, dt=0.001)
