import pytest

import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.processes import WhiteSignal

from nengolib.signal.dists import EvalPoints, Encoders
from nengolib import Alpha, Network
from nengolib.synapses import PureDelay


def test_alpha_whitesignal(Simulator, seed, rng, plt):
    # Pick test LinearSystem
    sys = Alpha(0.1)
    state = sys.X
    assert state.shape == (2, 1)

    # Pick test process
    n_steps = 1000
    dt = 0.01
    process = WhiteSignal(5.0, high=10, default_dt=dt, seed=seed)

    # Sample evaluation points
    dist = EvalPoints(state, process, n_steps=n_steps)
    assert dist.n_steps == n_steps
    assert dist.dt == dt  # taken from process
    assert isinstance(repr(dist), str)

    n_eval_points = 500
    eval_points = dist.sample(n_eval_points, 2, rng=rng)
    assert eval_points.shape == (n_eval_points, 2)

    # Sample encoders
    encoders = Encoders(state, process).sample(n_eval_points, 2, rng=rng)
    assert encoders.shape == (n_eval_points, 2)

    plt.figure()
    # plt.scatter(*encoders.T, label="Encoders")
    plt.scatter(*eval_points.T, s=2, marker='*', label="Eval Points")

    # Check that most evaluation points fall within radii
    x_m, zero, x_p = sorted(np.unique(encoders[:, 0]))
    assert np.allclose(-x_m, x_p)
    assert np.allclose(zero, 0)
    sl = (1/x_m < eval_points[:, 0]) & (eval_points[:, 0] < 1/x_p)
    assert np.count_nonzero(sl) / float(n_eval_points) >= 0.99

    y_m, zero, y_p = sorted(np.unique(encoders[:, 1]))
    assert np.allclose(zero, 0)
    assert np.allclose(-y_m, y_p)
    sl = (1/y_m < eval_points[:, 1]) & (eval_points[:, 1] < 1/y_p)
    assert np.count_nonzero(sl) / float(n_eval_points) >= 0.99

    # Simulate same process / system in nengo network
    with Network() as model:
        output = nengo.Node(output=process)
        probes = [nengo.Probe(output, synapse=sub) for sub in sys]

    with Simulator(model, dt=dt) as sim:
        sim.run(n_steps*dt)

    plt.scatter(sim.data[probes[0]], sim.data[probes[1]],
                s=1, alpha=0.5, label="Simulated")
    plt.legend()

    # Check that each eval_point is a subset / sample from the ideal
    ideal = np.asarray([sim.data[probes[0]].squeeze(),
                        sim.data[probes[1]].squeeze()]).T
    assert ideal.shape == (n_steps, 2)

    for pt in eval_points:
        dists = np.linalg.norm(ideal - pt[None, :], axis=1)
        assert dists.shape == (n_steps,)
        assert np.allclose(np.min(dists), 0)


def test_invalid_process():
    with pytest.raises(ValidationError):
        assert EvalPoints(Alpha(0.1), process=1)


def test_given_dt():
    process = WhiteSignal(1.0, high=10)
    assert EvalPoints(Alpha(0.1), process, dt=0.1).dt == 0.1


def test_invalid_sample():
    process = WhiteSignal(1.0, high=10)
    sys = PureDelay(0.1, order=4)

    dist = EvalPoints(sys, process)
    with pytest.raises(ValidationError):
        dist.sample(100, len(sys))  # needs to equal sys.size_out

    dist = Encoders(sys, process)
    with pytest.raises(ValidationError):
        dist.sample(100, len(sys))  # needs to equal sys.size_out
