import numpy as np
import pytest

import nengo

from nengolib.signal.normalization import (
    scale_state, Controllable, Observable, HankelNorm)
from nengolib import Lowpass, Alpha, Network
from nengolib.networks import LinearNetwork
from nengolib.signal import (
    sys_equal, ss_equal, state_norm, impulse, decompose_states)
from nengolib.synapses import Bandpass, Highpass


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


@pytest.mark.parametrize("sys", [
    Lowpass(0.005), Alpha(0.01), Bandpass(100, 5), Highpass(0.001, 4)])
def test_hankel_normalization(Simulator, sys, rng):
    radius = 5.0
    dt = 0.001
    T = 1.0

    l1_norms = []
    for sub in decompose_states(sys):
        response = impulse(sub, dt=dt, length=int(T / dt))
        assert np.allclose(response[-10:], 0)
        l1_norms.append(radius * np.sum(abs(response * dt)))

    with Network() as model:
        stim = nengo.Node(output=lambda t: rng.uniform(-radius, radius))
        subnet = LinearNetwork(sys, n_neurons=1, synapse=0.02, dt=dt,
                               radii=radius, normalizer=HankelNorm(),
                               neuron_type=nengo.neurons.Direct())
        nengo.Connection(stim, subnet.input, synapse=None)
        p = nengo.Probe(subnet.x.output, synapse=None)

    # lower bound is to see how well Hankel norm approximates L1 norm
    assert (0.3*subnet.info['radii'] <= l1_norms).all()
    assert (l1_norms <= subnet.info['radii']).all()

    sim = Simulator(model, dt=dt)
    sim.run(T)

    worst_x = np.max(abs(sim.data[p]), axis=0)

    # lower bound includes both approximation error and the gap between
    # uniform noise and the true worst-case input
    assert (0.1 <= worst_x).all()
    assert (worst_x <= 1 + 1e-13).all()
