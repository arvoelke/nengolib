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
    # note: nengolib.PerfectLIF is not substituted for nengo.LIF, but it will
    # become the default due to the Network substitution


def unpatch():
    """Resets the effect of patch()."""
    nengo.Network = NengoNetwork
    nengo.Connection = NengoConnection
