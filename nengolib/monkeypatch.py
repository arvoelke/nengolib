import nengo

from nengolib.network import Network
from nengolib.connection import Connection

from nengo import Network as NengoNetwork, Connection as NengoConnection


__all__ = ['patch', 'unpatch']


def patch(network=True, connection=True):
    """Monkey-patches nengolib.Network and/or nengolib.Connection."""
    if network:
        nengo.Network = Network
    if connection:
        nengo.Connection = Connection


def unpatch():
    """Resets the effect of patch()."""
    nengo.Network = NengoNetwork
    nengo.Connection = NengoConnection
