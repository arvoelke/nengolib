import nengo
from nengo import Connection as NengoConnection, Network as NengoNetwork

from nengolib.monkeypatch import patch, unpatch
from nengolib import Network, Connection
from nengolib.neurons import PerfectLIF
from nengolib.stats import ScatteredHypersphere


# http://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module

try:
    from importlib import reload  # Python 3.4+

except ImportError:

    try:
        from imp import reload  # Python 3.0 - 3.3

    except ImportError:
        assert reload  # Python 2.X


def test_monkeypatch():
    assert Connection is not NengoConnection
    assert Network is not NengoNetwork

    reload(nengo)
    assert nengo.Connection is NengoConnection
    assert nengo.Network is NengoNetwork

    patch()

    assert nengo.Connection is Connection
    assert nengo.Network is Network

    unpatch()

    assert nengo.Connection is NengoConnection
    assert nengo.Network is NengoNetwork

    patch(network=False)

    assert nengo.Connection is Connection
    assert nengo.Network is NengoNetwork

    reload(nengo)
    patch(connection=False)

    assert nengo.Connection is NengoConnection
    assert nengo.Network is Network

    reload(nengo)


def test_model():
    # monkey-patching affects all configs and connections recursively
    reload(nengo)
    patch()

    with nengo.Network() as model:
        nengo.networks.EnsembleArray(10, 1)

    for conn in model.all_connections:
        assert isinstance(conn, Connection)

    assert isinstance(model, Network)

    # note the subnetwork won't be a nengolib.Network, but it will still
    # inherit the top-level config
    for ens in model.all_ensembles:
        assert isinstance(ens.eval_points, ScatteredHypersphere)
        assert isinstance(ens.encoders, ScatteredHypersphere)
        assert isinstance(ens.neuron_type, PerfectLIF)

    reload(nengo)
