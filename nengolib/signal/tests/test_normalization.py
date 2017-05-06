import numpy as np
import pytest

import nengo

from nengolib.signal.normalization import (
    scale_state, Controllable, Observable, Balreal, HankelNorm, L1Norm)
from nengolib import Lowpass, Alpha, Network
from nengolib.networks import LinearNetwork
from nengolib.signal import (
    sys_equal, ss_equal, state_norm, impulse, decompose_states, balreal)
from nengolib.synapses import Bandpass, Highpass, PureDelay


@pytest.mark.parametrize("radius", [0.7, 3, [1.5, 0.2]])
def test_scale_state(radius):
    sys = Alpha(0.1)
    scaled = scale_state(sys, radii=radius)
    assert not np.allclose(sys.B, scaled.B)
    assert not np.allclose(sys.C, scaled.C)

    # Check that it's still the same system, even though different matrices
    assert sys_equal(sys, scaled)

    # Check that the state vectors have scaled power
    assert np.allclose(state_norm(sys) / radius, state_norm(scaled))


def test_invalid_scale_state():
    sys = Lowpass(0.1)

    scale_state(sys, radii=[1])

    with pytest.raises(ValueError):
        scale_state(sys, radii=[[1]])

    with pytest.raises(ValueError):
        scale_state(sys, radii=[1, 2])


def test_ccf_normalization():
    sys = Lowpass(0.1) * Lowpass(0.2)
    assert ss_equal(
        Controllable()(sys)[0],
        ([[-15, -50], [1, 0]], [[1], [0]], [[0, 50]], [[0]]))
    assert ss_equal(
        Observable()(sys, radii=[1, 2])[0],
        ([[-15, 2], [-25, 0]], [[0], [25]], [[1, 0]], [[0]]))


def test_radii_immutability():
    radii = np.asarray([1, 2])
    radii.flags.writeable = False
    HankelNorm()(Alpha(0.1), radii=radii)
    assert np.allclose(radii, [1, 2])  # issues/90


@pytest.mark.parametrize("sys", [PureDelay(0.1, 4), PureDelay(0.05, 5, 5)])
def test_balreal_normalization(sys):
    normalized, info = Balreal()(sys)
    balsys, S = balreal(sys)
    assert ss_equal(balsys, normalized)
    assert np.allclose(info['sigma'], S)


def _test_normalization(Simulator, sys, rng, normalizer, l1_lower,
                        lower, radius=5.0, dt=0.0001, T=1.0, eps=1e-5):
    l1_norms = np.empty(len(sys))
    for i, sub in enumerate(decompose_states(sys)):
        response = impulse(sub, dt=dt, length=int(T / dt))
        assert np.allclose(response[-10:], 0)
        l1_norms[i] = radius * np.sum(abs(response * dt))

    with Network() as model:
        stim = nengo.Node(output=lambda t: rng.choice([-radius, radius])
                          if t < T/2 else radius)
        tau = 0.02
        subnet = LinearNetwork(sys, n_neurons_per_ensemble=1, synapse=tau,
                               dt=dt, input_synapse=tau,
                               radii=radius, normalizer=normalizer,
                               neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)
        p = nengo.Probe(subnet.x.output, synapse=None)

    est_worst_x = subnet.info['radii']
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


@pytest.mark.parametrize("sys,lower", [
    (Lowpass(0.005), 1.0), (Alpha(0.01), 0.3), (Bandpass(50, 5), 0.05),
    (Highpass(0.01, 4), 0.1), (PureDelay(0.1, 2, 2), 0.3)])
def test_hankel_normalization(Simulator, rng, sys, lower):
    _test_normalization(Simulator, sys, rng, HankelNorm(),
                        l1_lower=0.5, lower=lower)


@pytest.mark.parametrize("radius", [0.5, 5, 10])
@pytest.mark.parametrize("sys", [Lowpass(0.005)])
def test_normalization_radius(Simulator, rng, sys, radius):
    _test_normalization(Simulator, sys, rng, L1Norm(), radius=radius,
                        l1_lower=1-1e-3, lower=1.0)


@pytest.mark.parametrize("sys,lower", [
    (Alpha(0.01), 0.4), (Bandpass(50, 5), 0.1), (Highpass(0.01, 4), 0.2),
    (PureDelay(0.1, 2, 2), 0.4), (PureDelay(0.1, 3), 0.3),
    (PureDelay(0.1, 4, 4), 0.15)])
def test_l1_normalization_crossing(Simulator, rng, sys, lower):
    # note the lower bounds are higher than those for the hankel norm tests
    # for the same systems
    # TODO: highpass performs unexpectedly poorly (0.9 is too tolerant)
    _test_normalization(Simulator, sys, rng, L1Norm(),
                        l1_lower=0.9, lower=lower)
