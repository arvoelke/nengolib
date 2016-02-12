import numpy as np
import pytest

import nengo

from nengolib.synapses.mapping import ss2sim
from nengolib import Network, Lowpass, Alpha, LinearFilter
from nengolib.signal import apply_filter, s


def test_mapping(Simulator, plt):
    sys = Alpha(0.1)
    syn = Lowpass(0.01)
    gsyn = 2*syn  # scaled lowpass
    isyn = 2/s  # scaled integrator
    dt = 0.001

    ss = ss2sim(sys, syn)  # normal lowpass, continuous
    dss = ss2sim(sys, syn, dt)  # normal lowpass, discrete
    gss = ss2sim(sys, gsyn)  # scaled lowpass, continuous
    gdss = ss2sim(sys, gsyn, dt)  # scaled lowpass, discrete
    iss = ss2sim(sys, isyn)  # scaled integrator, continuous
    idss = ss2sim(sys, isyn, dt)  # scaled integrator, discrete

    with Network() as model:
        stim = nengo.Node(output=np.sin)

        probes = []
        for (A, B, C, D), synapse in ((ss, syn), (dss, syn), (gss, gsyn),
                                      (gdss, gsyn), (iss, isyn), (idss, isyn)):
            x = nengo.Node(size_in=2)
            y = nengo.Node(size_in=1)

            nengo.Connection(stim, x, transform=B, synapse=synapse)
            nengo.Connection(x, x, transform=A, synapse=synapse)
            nengo.Connection(x, y, transform=C, synapse=None)
            nengo.Connection(stim, y, transform=D, synapse=None)

            probes.append(nengo.Probe(y))

        p_stim = nengo.Probe(stim)

    pss, pdss, pgss, pgdss, piss, pidss = probes

    sim = Simulator(model, dt=dt)
    sim.run(1.0)

    expected = apply_filter(sim.data[p_stim], sys, dt, axis=0)

    plt.plot(sim.trange(), sim.data[p_stim], label="Stim", alpha=0.5)
    plt.plot(sim.trange(), sim.data[pss], label="Continuous", alpha=0.5)
    plt.plot(sim.trange(), sim.data[pdss], label="Discrete", alpha=0.5)
    plt.plot(sim.trange(), sim.data[pgss], label="Gain Cont.", alpha=0.5)
    plt.plot(sim.trange(), sim.data[pgdss], label="Gain Disc.", alpha=0.5)
    plt.plot(sim.trange(), sim.data[piss], label="Integ Cont.", alpha=0.5)
    plt.plot(sim.trange(), sim.data[pidss], label="Integ Disc.", alpha=0.5)
    plt.plot(sim.trange(), expected, label="Expected", linestyle='--')
    plt.legend()

    assert np.allclose(sim.data[pss], expected, atol=0.01)
    assert np.allclose(sim.data[pdss], expected)
    assert np.allclose(sim.data[pgss], expected, atol=0.01)
    assert np.allclose(sim.data[pgdss], expected)
    assert np.allclose(sim.data[piss], expected, atol=0.01)
    assert np.allclose(sim.data[pidss], expected)


def test_unsupported_synapse():
    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=Alpha(0.1))

    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=LinearFilter([1, 2], [2, 1]), dt=0.01)

    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=LinearFilter(1))

    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=LinearFilter([1, 0.01], [1]))

    with pytest.raises(TypeError):
        ss2sim(sys=Lowpass(0.1), synapse=LinearFilter([1], [2, 1, 1]))
