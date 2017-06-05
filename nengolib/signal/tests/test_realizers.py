import numpy as np
import pytest

from scipy.linalg import inv

import nengo

from nengolib.signal.realizers import (
    _realize, Identity, Balanced, Hankel, L1Norm, H2Norm)
from nengolib import Lowpass, Alpha, Network
from nengolib.networks import LinearNetwork
from nengolib.signal import (
    ss_equal, sys_equal, state_norm, balanced_transformation)
from nengolib.synapses import Bandpass, Highpass, PadeDelay


@pytest.mark.parametrize("radii", [1, 3, [1.5, 0.2]])
def test_func_realize(radii):
    sys = Alpha(0.1)

    T = np.asarray([[1., 2.], [0, -1.]])

    for Tinv in (None, inv(T)):
        realize_result = _realize(sys, radii, T, Tinv)

        assert realize_result.sys is sys
        assert np.allclose(inv(realize_result.T), realize_result.Tinv)

        rsys = realize_result.realization
        assert ss_equal(rsys, sys.transform(realize_result.T))

        # Check that the state vector are related by T
        length = 1000
        dt = 0.001
        x_old = np.asarray([sub.impulse(length, dt) for sub in sys])
        x_new = np.asarray([sub.impulse(length, dt) for sub in rsys])

        r = np.atleast_2d(np.asarray(radii).T).T
        assert np.allclose(np.dot(T, x_new * r), x_old)


def test_invalid_realize():
    sys = Lowpass(0.1)

    with pytest.raises(ValueError):
        _realize(sys, radii=[[1]], T=np.eye(len(sys)))

    with pytest.raises(ValueError):
        _realize(sys, radii=[1, 2], T=np.eye(len(sys)))


@pytest.mark.parametrize("radii", [0.5, 1, [1.5, 0.2]])
def test_identity(radii):
    sys = Alpha(0.1)

    identity = Identity()
    assert repr(identity) == "Identity()"

    I = np.eye(len(sys))
    realize_result = identity(sys, radii)
    assert realize_result.sys is sys
    assert np.allclose(realize_result.T, I * radii)
    assert np.allclose(realize_result.Tinv, inv(I * radii))

    rsys = realize_result.realization
    assert ss_equal(rsys, sys.transform(realize_result.T))

    # Check that it's still the same system, even though different matrices
    assert sys_equal(sys, rsys)
    if radii == 1:
        assert ss_equal(sys, rsys)
    else:
        assert not np.allclose(sys.B, rsys.B)
        assert not np.allclose(sys.C, rsys.C)

    # Check that the state vectors have scaled power
    assert np.allclose(state_norm(sys) / radii, state_norm(rsys))


@pytest.mark.parametrize("sys", [PadeDelay(0.1, 4), PadeDelay(0.05, 5, 5)])
def test_balreal_normalization(sys):
    radii = np.arange(len(sys)) + 1

    balanced = Balanced()
    assert repr(balanced) == "Balanced()"

    realizer_result = balanced(sys, radii)

    T, Tinv, _ = balanced_transformation(sys)

    assert np.allclose(realizer_result.T / radii[None, :], T)
    assert np.allclose(realizer_result.Tinv * radii[:, None], Tinv)
    assert np.allclose(inv(T), Tinv)

    assert sys_equal(sys, realizer_result.realization)


def _test_normalization(Simulator, sys, rng, realizer, l1_lower,
                        lower, radius=5.0, dt=0.0001, T=1.0, eps=1e-5):
    response = sys.X.impulse(int(T / dt), dt=dt)
    assert np.allclose(response[-10:], 0)
    l1_norms = radius * np.sum(abs(response * dt), axis=0)

    with Network() as model:
        stim = nengo.Node(output=lambda t: rng.choice([-radius, radius])
                          if t < T/2 else radius)
        tau = 0.02
        subnet = LinearNetwork(sys, n_neurons_per_ensemble=1, synapse=tau,
                               dt=dt, input_synapse=tau,
                               radii=radius, realizer=realizer,
                               neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)
        p = nengo.Probe(subnet.x.output, synapse=None)

        assert np.allclose(inv(subnet.realizer_result.T),
                           subnet.realizer_result.Tinv)

    trans = subnet.realizer_result.T
    assert trans.shape == (len(sys), len(sys))
    est_worst_x = np.diagonal(trans)
    assert np.allclose(np.diag(est_worst_x), trans)  # make sure diagonal
    assert est_worst_x.shape == (len(sys),)

    assert ((l1_lower*est_worst_x <= l1_norms) |
            (est_worst_x <= eps)).all()
    assert (l1_norms <= est_worst_x + eps).all()

    with Simulator(model, dt=dt) as sim:
        sim.run(T)

    # lower bound includes both approximation error and the gap between
    # random {-1, 1} flip-flop inputs and the true worst-case input
    worst_x = np.max(abs(sim.data[p]), axis=0)
    assert (lower <= worst_x + eps).all()
    assert (worst_x <= 1 + eps).all()


def test_l1_repr():
    assert (repr(L1Norm(rtol=.1, max_length=10)) ==
            "L1Norm(rtol=0.1, max_length=10)")


@pytest.mark.parametrize("sys,lower", [
    (Lowpass(0.005), 1.0), (Alpha(0.01), 0.3), (Bandpass(50, 5), 0.05),
    (Highpass(0.01, 4), 0.1), (PadeDelay(0.1, 2, 2), 0.3)])
def test_hankel_normalization(Simulator, rng, sys, lower):
    _test_normalization(Simulator, sys, rng, Hankel(),
                        l1_lower=0.5, lower=lower)


@pytest.mark.parametrize("radius", [0.5, 5, 10])
@pytest.mark.parametrize("sys", [Lowpass(0.005)])
def test_normalization_radius(Simulator, rng, sys, radius):
    _test_normalization(Simulator, sys, rng, L1Norm(), radius=radius,
                        l1_lower=1-1e-3, lower=1.0)


@pytest.mark.parametrize("sys,lower", [
    (Alpha(0.01), 0.4), (Bandpass(50, 5), 0.1), (Highpass(0.01, 4), 0.2),
    (PadeDelay(0.1, 2, 2), 0.4), (PadeDelay(0.1, 3), 0.3),
    (PadeDelay(0.1, 4, 4), 0.15)])
def test_l1_normalization_crossing(Simulator, rng, sys, lower):
    # note the lower bounds are higher than those for the hankel norm tests
    # for the same systems
    # TODO: highpass performs unexpectedly poorly (0.9 is too tolerant)
    _test_normalization(Simulator, sys, rng, L1Norm(),
                        l1_lower=0.9, lower=lower)


@pytest.mark.parametrize("sys", [
    Alpha(0.01), Bandpass(50, 5), Highpass(0.01, 4),
    PadeDelay(0.1, 2, 2), PadeDelay(1.0, 10),
    PadeDelay(2.0, 4, 4)])
def test_h2_realizer(sys):
    r1 = Hankel()(sys)
    r2 = H2Norm()(sys)

    assert np.allclose(np.diag(np.diagonal(r2.T)), r2.T)  # make sure diagonal
    s1 = np.diag(r1.T)
    s2 = np.diag(r2.T)

    # Hankel and H2Norm have the same shape and order of magnitude
    assert (0.3*s1 <= s2).all()
    assert (6*s1 >= s2).all()
    assert (np.sign(np.diff(s1)) == np.sign(np.diff(s1))).all()
