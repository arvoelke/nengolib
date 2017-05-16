import pytest

import nengo

from nengolib.networks.ensemble_cube import EnsembleCube
from nengolib import Network
from nengolib.signal import nrmse


def test_ensemble_cube(Simulator, rng, seed):
    n_neurons = 1000
    d = 2
    half_width = 20
    tau = 0.05

    n_samples = 20
    sample_time = 0.2

    with Network(seed=seed) as model:
        cube = EnsembleCube(n_neurons, d, half_width=half_width)

        process = nengo.processes.PresentInput(
            rng.uniform(-half_width, half_width, (n_neurons, d)),
            sample_time)

        # Here it is the case that we can express the function in terms of
        # each of the dimensions, but the point of this network is that this
        # is not always known explicitly / optimally.
        def function(x):
            return 2*x[0]**2 - x[1]**2 - half_width

        stim = nengo.Node(output=process)
        decoded = nengo.Node(size_in=1)
        ideal = nengo.Node(size_in=1)

        nengo.Connection(stim, cube, synapse=None)
        nengo.Connection(cube, decoded, function=function, synapse=None)
        nengo.Connection(stim, ideal, function=function, synapse=None)

        p_stim = nengo.Probe(stim, synapse=tau)
        p_output = nengo.Probe(cube, synapse=tau)
        p_decoded = nengo.Probe(decoded, synapse=tau)
        p_ideal = nengo.Probe(ideal, synapse=tau)

    with Simulator(model) as sim:
        sim.run(n_samples * sample_time)

    assert nrmse(sim.data[p_output], sim.data[p_stim]) < 0.01  # 1 percent
    assert nrmse(sim.data[p_decoded], sim.data[p_ideal]) < 0.02  # 2 percent


def test_bad_ensemble_cube():
    with pytest.raises(ValueError):
        EnsembleCube(100, 2, radius=1)

    with pytest.raises(ValueError):
        EnsembleCube(
            100, 2, encoders=nengo.dists.UniformHypersphere(surface=True))
