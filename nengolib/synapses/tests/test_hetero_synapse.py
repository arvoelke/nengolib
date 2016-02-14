import pytest

import numpy as np

import nengo

from nengolib.synapses.hetero_synapse import HeteroSynapse
from nengolib import Network
from nengolib.signal import z
from nengolib.stats import sphere
from nengolib.synapses import Lowpass, Alpha


def test_hetero_neurons(Simulator):

    n_neurons = 100
    dt = 0.001
    T = 0.1
    dims_in = 2

    taus = nengo.dists.Uniform(0.001, 0.1).sample(n_neurons)
    synapses = [Lowpass(tau) for tau in taus]
    encoders = sphere.sample(n_neurons, dims_in)

    hs = HeteroSynapse(synapses, dt)

    def embed_encoders(x):
        # Reshapes the vectors to be the same dimensionality as the
        # encoders, and then takes the dot product row by row.
        # See http://stackoverflow.com/questions/26168363/ for a more
        # efficient solution.
        return np.sum(encoders * hs.from_vector(x), axis=1)

    with Network() as model:
        # Input stimulus
        stim = nengo.Node(size_in=dims_in)
        for i in range(dims_in):
            nengo.Connection(
                nengo.Node(output=nengo.processes.WhiteSignal(T)),
                stim[i], synapse=None)

        # HeteroSynapse node
        syn = nengo.Node(size_in=dims_in, output=hs)

        # For comparing results
        x = [nengo.Ensemble(n_neurons, dims_in, seed=0, encoders=encoders)
             for _ in range(2)]  # expected, actual

        # Expected
        for i, synapse in enumerate(synapses):
            t = np.zeros_like(encoders)
            t[i, :] = encoders[i, :]
            nengo.Connection(stim, x[0].neurons, transform=t, synapse=synapse)

        # Actual
        nengo.Connection(stim, syn, synapse=None)
        nengo.Connection(
            syn, x[1].neurons, function=embed_encoders, synapse=None)

        # Probes
        p_exp = nengo.Probe(x[0].neurons, synapse=None)
        p_act = nengo.Probe(x[1].neurons, synapse=None)

    # Check correctness
    sim = Simulator(model, dt=dt)
    sim.run(T)

    assert np.allclose(sim.data[p_act], sim.data[p_exp])


def test_hetero_vector(Simulator):
    n_neurons = 20
    dt = 0.0005
    T = 0.1
    dims_in = 2
    synapses = [Alpha(0.1), Lowpass(0.005)]
    assert dims_in == len(synapses)

    encoders = sphere.sample(n_neurons, dims_in)

    with Network() as model:
        # Input stimulus
        stim = nengo.Node(size_in=dims_in)
        for i in range(dims_in):
            nengo.Connection(
                nengo.Node(output=nengo.processes.WhiteSignal(T)),
                stim[i], synapse=None)

        # HeteroSynapse Nodes
        syn_elemwise = nengo.Node(
            size_in=dims_in,
            output=HeteroSynapse(synapses, dt, elementwise=True))

        # For comparing results
        x = [nengo.Ensemble(n_neurons, dims_in, seed=0, encoders=encoders)
             for _ in range(2)]  # expected, actual

        # Expected
        for j, synapse in enumerate(synapses):
            nengo.Connection(stim[j], x[0][j], synapse=synapse)

        # Actual
        nengo.Connection(stim, syn_elemwise, synapse=None)
        nengo.Connection(syn_elemwise, x[1], synapse=None)

        # Probes
        p_exp = nengo.Probe(x[0], synapse=None)
        p_act_elemwise = nengo.Probe(x[1], synapse=None)

    # Check correctness
    sim = Simulator(model, dt=dt)
    sim.run(T)

    assert np.allclose(sim.data[p_act_elemwise], sim.data[p_exp])


def test_hetero_multi_vector(Simulator):
    n_neurons = 20
    dt = 0.0005
    T = 0.1
    dims_in = 2
    synapses = [Alpha(0.1), Lowpass(0.005), Alpha(0.02)]

    dims_out = len(synapses)*dims_in
    encoders = sphere.sample(n_neurons, dims_out)

    with Network() as model:
        # Input stimulus
        stim = nengo.Node(size_in=dims_in)
        for i in range(dims_in):
            nengo.Connection(
                nengo.Node(output=nengo.processes.WhiteSignal(T)),
                stim[i], synapse=None)

        # HeteroSynapse Nodes
        syn_dot = nengo.Node(
            size_in=dims_in, output=HeteroSynapse(synapses, dt))
        syn_elemwise = nengo.Node(
            size_in=dims_out, output=HeteroSynapse(
                np.repeat(synapses, dims_in), dt, elementwise=True))

        # For comparing results
        x = [nengo.Ensemble(n_neurons, dims_out, seed=0, encoders=encoders)
             for _ in range(3)]  # expected, actual 1, actual 2

        # Expected
        for j, synapse in enumerate(synapses):
            nengo.Connection(
                stim, x[0][j*dims_in:(j+1)*dims_in], synapse=synapse)

        # Actual (method #1 = matrix multiplies)
        nengo.Connection(stim, syn_dot, synapse=None)
        nengo.Connection(syn_dot, x[1], synapse=None)

        # Actual (method #2 = elementwise)
        for j in range(len(synapses)):
            nengo.Connection(
                stim, syn_elemwise[j*dims_in:(j+1)*dims_in], synapse=None)
        nengo.Connection(syn_elemwise, x[2], synapse=None)

        # Probes
        p_exp = nengo.Probe(x[0], synapse=None)
        p_act_dot = nengo.Probe(x[1], synapse=None)
        p_act_elemwise = nengo.Probe(x[2], synapse=None)

    # Check correctness
    sim = Simulator(model, dt=dt)
    sim.run(T)

    assert np.allclose(sim.data[p_act_dot], sim.data[p_exp])
    assert np.allclose(sim.data[p_act_elemwise], sim.data[p_exp])


def test_hetero_casting():
    sys = ~z
    hs = HeteroSynapse(sys)
    assert [sys] == hs.systems


def test_invalid_system():
    with pytest.raises(ValueError):
        HeteroSynapse(Lowpass(0.1))  # no dt provided
