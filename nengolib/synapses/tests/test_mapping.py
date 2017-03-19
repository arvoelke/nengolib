import numpy as np
import pytest

import nengo

from nengolib.synapses.mapping import ss2sim
from nengolib import Network, Lowpass, Alpha
from nengolib.signal import apply_filter, s, z, ss_equal, cont2discrete
from nengolib.synapses import Highpass, PureDelay, DoubleExp


def test_mapping(Simulator, plt, seed):
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
    assert ss.analog and gss.analog and iss.analog
    assert not (dss.analog or gdss.analog or idss.analog)

    with Network(seed=seed) as model:
        stim = nengo.Node(output=lambda t: np.sin(20*np.pi*t))

        probes = []
        for mapped, synapse in ((ss, syn), (dss, syn), (gss, gsyn),
                                (gdss, gsyn), (iss, isyn), (idss, isyn)):
            A, B, C, D = mapped.ss
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

    expected = apply_filter(sys, dt, sim.data[p_stim], axis=0)

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


def test_principle3_continuous():
    sys = PureDelay(0.1, order=5)

    tau = 0.01
    syn = Lowpass(tau)

    FH = ss2sim(sys, syn, dt=None)

    assert np.allclose(FH.A, tau * sys.A + np.eye(len(sys)))
    assert np.allclose(FH.B, tau * sys.B)
    assert np.allclose(FH.C, sys.C)
    assert np.allclose(FH.D, sys.D)


def test_principle3_discrete():
    sys = PureDelay(0.1, order=5)

    tau = 0.01
    dt = 0.002
    syn = Lowpass(tau)

    FH = ss2sim(sys, syn, dt=dt)

    a = np.exp(-dt / tau)
    sys = cont2discrete(sys, dt=dt)
    assert np.allclose(FH.A, (sys.A - a * np.eye(len(sys))) / (1 - a))
    assert np.allclose(FH.B, sys.B / (1 - a))
    assert np.allclose(FH.C, sys.C)
    assert np.allclose(FH.D, sys.D)

    # We can also do the discretization ourselves and then pass in dt=None
    assert ss_equal(
        ss2sim(sys, cont2discrete(syn, dt=dt), dt=None), FH)


@pytest.mark.parametrize("sys", [PureDelay(0.1, order=5), Lowpass(0.1)])
def test_doubleexp_continuous(sys):
    tau1 = 0.05
    tau2 = 0.02
    syn = DoubleExp(tau1, tau2)

    FH = ss2sim(sys, syn, dt=None)

    A = sys.A
    FHA = tau1 * tau2 * np.dot(A, A) + (tau1 + tau2) * A + np.eye(len(A))
    B = sys.B
    FHB = (tau1 * tau2 * A + (tau1 + tau2) * np.eye(len(A))).dot(B)
    assert np.allclose(FH.A, FHA)
    assert np.allclose(FH.B, FHB)
    assert np.allclose(FH.C, sys.C)
    assert np.allclose(FH.D, sys.D)


def test_doubleexp_discrete():
    sys = PureDelay(0.1, order=5)

    tau1 = 0.05
    tau2 = 0.02
    dt = 0.002
    syn = DoubleExp(tau1, tau2)

    FH = ss2sim(sys, syn, dt=dt)

    a1 = np.exp(-dt / tau1)
    a2 = np.exp(-dt / tau2)
    t1 = 1 / (1 - a1)
    t2 = 1 / (1 - a2)
    c = [a1 * a2 * t1 * t2, - (a1 + a2) * t1 * t2, t1 * t2]
    sys = cont2discrete(sys, dt=dt)

    A = sys.A
    FHA = c[2] * np.dot(A, A) + c[1] * A + c[0] * np.eye(len(A))
    B = sys.B
    FHB = (c[2] * A + (c[1] + c[2]) * np.eye(len(A))).dot(B)
    assert np.allclose(FH.A, FHA)
    assert np.allclose(FH.B, FHB)
    assert np.allclose(FH.C, sys.C)
    assert np.allclose(FH.D, sys.D)


def test_unsupported_mapping():
    lpf = Lowpass(0.1)

    with pytest.raises(ValueError):
        ss2sim(sys=lpf, synapse=Highpass(0.1))

    with pytest.raises(ValueError):
        ss2sim(sys=~z, synapse=lpf)

    with pytest.raises(ValueError):
        ss2sim(sys=lpf, synapse=~z)

    with pytest.raises(ValueError):
        ss2sim(sys=~z, synapse=~z, dt=1.)
