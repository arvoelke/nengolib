import warnings

import nengo
from nengo.params import NumberParam
from nengo.synapses import SynapseParam

from nengolib.network import Network
from nengolib.signal.system import LinearSystem, canonical, is_exp_stable
from nengolib.synapses.mapping import ss2sim

__all__ = ['LinearNetwork']


class LinearNetwork(Network):
    """Network implementing a linear time-invariant (LTI) system."""

    synapse = SynapseParam('synapse')
    dt = NumberParam('dt', low=0, low_open=True)

    def __init__(self, sys, n_neurons, synapse, dt,
                 label=None, seed=None, add_to_container=None, **ens_kwargs):
        super(LinearNetwork, self).__init__(label, seed, add_to_container)

        # Parameter checking
        self.sys = LinearSystem(sys)
        self.n_neurons = n_neurons
        self.synapse = synapse
        self.dt = dt

        if not is_exp_stable(self.sys):
            warnings.warn("System is not exponentially stable: %s" % self.sys)

        # Obtain a normalized state-space representation
        self.sys = canonical(self.sys, controllable=False)
        self.A, self.B, self.C, self.D = ss2sim(
            self.sys, self.synapse, self.dt).ss

        # Attributes
        self.size_in = self.B.shape[1]
        self.size_state = len(self.A)
        self.size_out = len(self.C)

        # Create internal Nengo objects
        self.input = nengo.Node(size_in=self.size_in, label="input")
        self.output = nengo.Node(size_in=self.size_out, label="output")
        self.x = nengo.networks.EnsembleArray(
            self.n_neurons, self.size_state, ens_dimensions=1, **ens_kwargs)

        # Connect everything up using (A, B, C, D)
        self.conn_A = nengo.Connection(
            self.x.output, self.x.input, transform=self.A, synapse=synapse)
        self.conn_B = nengo.Connection(
            self.input, self.x.input, transform=self.B, synapse=synapse)
        self.conn_C = nengo.Connection(
            self.x.output, self.output, transform=self.C, synapse=None)
        self.conn_D = nengo.Connection(
            self.input, self.output, transform=self.D, synapse=None)
