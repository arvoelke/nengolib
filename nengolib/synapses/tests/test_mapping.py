import numpy as np
import pytest

import nengo

from nengolib.synapses.mapping import ss2sim
from nengolib import Network, Lowpass, Alpha, Triangle
from nengolib.signal import apply_filter


def test_mapping(Simulator, plt):
    sys = Alpha(0.1)
    synapse = Lowpass(0.01)
    dt = 0.001

    ss = ss2sim(sys, synapse)
    dss = ss2sim(sys, synapse, dt)

    with Network() as model:
        stim = nengo.Node(output=np.sin)
        x = nengo.Node(size_in=2)
        dx = nengo.Node(size_in=2)
        out = nengo.Node(size_in=1)
        dout = nengo.Node(size_in=1)

        for X, (A, B, C, D), Y in ((x, ss, out), (dx, dss, dout)):
            print A, B, C, D
            nengo.Connection(stim, X, transform=B, synapse=synapse)
            nengo.Connection(X, X, transform=A, synapse=synapse)
            nengo.Connection(X, Y, transform=C, synapse=None)
            nengo.Connection(stim, Y, transform=D, synapse=None)

        p_stim = nengo.Probe(stim)
        p_out = nengo.Probe(out)
        p_dout = nengo.Probe(dout)

    sim = Simulator(model, dt=dt)
    sim.run(1.0)

    expected = apply_filter(sim.data[p_stim], sys, dt, axis=0)

    plt.plot(sim.trange(), sim.data[p_stim], label="Stim", alpha=0.5)
    plt.plot(sim.trange(), sim.data[p_out], label="Continuous", alpha=0.5)
    plt.plot(sim.trange(), sim.data[p_dout], label="Discrete", alpha=0.5)
    plt.plot(sim.trange(), expected, label="Expected", linestyle='--')
    plt.legend()

    assert np.allclose(sim.data[p_dout], expected)
    assert np.allclose(sim.data[p_out], expected, atol=0.01)


def test_unsupported_synapse():
    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=Alpha(0.1))

    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=Triangle(0.1), dt=0.001)
