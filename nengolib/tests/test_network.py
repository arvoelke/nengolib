import nengo
from nengo import Network as BaseNetwork

from nengolib import Network
from nengolib.stats import ScatteredHypersphere


def test_network():
    with Network():
        x = nengo.Ensemble(100, 1)
        assert isinstance(x.eval_points, ScatteredHypersphere)
        assert isinstance(x.encoders, ScatteredHypersphere)

    with BaseNetwork():
        x = nengo.Ensemble(100, 1)
        assert not isinstance(x.eval_points, ScatteredHypersphere)
        assert not isinstance(x.encoders, ScatteredHypersphere)


def test_ensemble_array():
    with Network():
        ea = nengo.networks.EnsembleArray(100, 2)
        for ens in ea.ea_ensembles:
            assert isinstance(ens.eval_points, ScatteredHypersphere)
            assert isinstance(ens.encoders, ScatteredHypersphere)
