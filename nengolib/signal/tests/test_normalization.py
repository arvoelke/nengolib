import numpy as np
import pytest

import nengo

from nengolib.signal.normalization import (
    scale_state, Controllable, Observable, HankelNorm, L1Norm)
from nengolib import Lowpass, Alpha, Network
from nengolib.networks import LinearNetwork
from nengolib.signal import (
    sys_equal, ss_equal, state_norm, impulse, decompose_states)
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


def _test_normalization(Simulator, sys, rng, normalizer, l1_lower,
                        worst_lower, radius=5.0, dt=0.001, T=1.0):
    l1_norms = np.empty(len(sys))
    for i, sub in enumerate(decompose_states(sys)):
        response = impulse(sub, dt=dt, length=int(T / dt))
        assert np.allclose(response[-10:], 0)
        l1_norms[i] = radius * np.sum(abs(response * dt))

    with Network() as model:
        stim = nengo.Node(output=lambda t: rng.choice([-radius, radius])
                          if t < T/2 else radius)
        subnet = LinearNetwork(sys, n_neurons=1, synapse=0.02, dt=dt,
                               radii=radius, normalizer=normalizer,
                               neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)
        p = nengo.Probe(subnet.x.output, synapse=None)

    # lower bound is to see how well Hankel norm approximates L1 norm
    assert (l1_lower*subnet.info['radii'] <= l1_norms).all()
    assert (l1_norms <= subnet.info['radii'] + 1e-4).all()

    sim = Simulator(model, dt=dt)
    sim.run(T)

    worst_x = np.max(abs(sim.data[p]), axis=0)

    # lower bound includes both approximation error and the gap between
    # uniform noise and the true worst-case input
    assert (worst_lower <= worst_x).all()
    assert (worst_x <= 1 + 1e-13).all()


@pytest.mark.parametrize("sys", [
    Lowpass(0.005), Alpha(0.01), Bandpass(50, 5), Highpass(0.01, 4),
    PureDelay(0.1, 2, 2)])
def test_hankel_normalization(Simulator, sys, rng):
    _test_normalization(Simulator, sys, rng, HankelNorm(),
                        l1_lower=0.3, worst_lower=0.15)


@pytest.mark.parametrize("sys", [Lowpass(0.005)])
def test_l1_normalization_positive(Simulator, sys, rng):
    # all the states are always positive
    # TODO: get a test passing with higher-order PadeDelay
    _test_normalization(Simulator, sys, rng, L1Norm(),
                        l1_lower=0.999, worst_lower=0.999)


@pytest.mark.parametrize("sys", [
    Alpha(0.01), Bandpass(50, 5), Highpass(0.01, 4), PureDelay(0.1, 2, 2)])
def test_l1_normalization_crossing(Simulator, sys, rng):
    _test_normalization(Simulator, sys, rng, L1Norm(),
                        l1_lower=0.9, worst_lower=0.2)
