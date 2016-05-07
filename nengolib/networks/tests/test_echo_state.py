import pytest

import nengo
from nengo.processes import WhiteSignal
from nengo.utils.numpy import rmse, rms

from nengolib.networks.echo_state import EchoState

from nengolib import Network
from nengolib.synapses import Highpass


@pytest.mark.parametrize("include_bias", [True, False])
def test_echo_state(Simulator, plt, seed, rng, include_bias):
    test_t = 1.0
    train_t = 5.0
    dt = 0.001

    n_neurons = 1000
    dimensions = 2
    process = WhiteSignal(train_t, high=10)

    with Network() as model:
        stim = nengo.Node(output=process, size_out=dimensions)
        esn = EchoState(
            n_neurons, dimensions, include_bias=include_bias, rng=rng)
        nengo.Connection(stim, esn.input, synapse=None)

        p = nengo.Probe(esn.output, synapse=None)
        p_stim = nengo.Probe(stim, synapse=None)

    # train the reservoir to compute a highpass filter
    def function(x): return Highpass(0.01).filt(x, dt=dt)

    esn.train(function, test_t, dt, process)

    with Simulator(model, dt=dt, seed=seed) as sim:
        sim.run(test_t)

    ideal = function(sim.data[p_stim])

    plt.figure()
    plt.plot(sim.trange(), sim.data[p_stim], label="Input")
    plt.plot(sim.trange(), sim.data[p], label="Output")
    plt.plot(sim.trange(), ideal, label="Ideal")
    plt.legend()

    assert rmse(sim.data[p], ideal) <= 0.5 * rms(ideal)
