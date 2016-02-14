import warnings

import nengo
from nengo.params import NumberParam
from nengo.synapses import SynapseParam

from nengolib.network import Network
from nengolib.signal.system import LinearSystem, canonical, is_exp_stable
from nengolib.synapses.mapping import ss2sim


class LinearNetwork(Network):
    """Network implementing a linear time-invariant (LTI) system."""

    synapse = SynapseParam('synapse')
    dt = NumberParam('dt', low=0, low_open=True)

    def __init__(self, sys, n_neurons, synapse, dt,
                 label=None, seed=None, add_to_container=None, **ens_kwargs):
        super(LinearNetwork, self).__init__(label, seed, add_to_container)

        # Parameter checking
        self.sys = LinearSystem(sys)
        self.synapse = synapse
        self.dt = dt
        self.n_neurons = n_neurons

        if not is_exp_stable(self.sys):
            warnings.warn("System is not exponentially stable: %s" % self.sys)

        # Obtain a normalized state-space representation
        self.sys = canonical(self.sys, controllable=False)
        A, B, C, D = ss2sim(self.sys, self.synapse, self.dt).ss

        # Attributes
        self.A, self.B, self.C, self.D = (A, B, C, D)
        self.size_in = B.shape[1]
        self.size_state = len(A)
        self.size_out = len(C)

        # Create internal Nengo objects
        self.input = nengo.Node(size_in=self.size_in, label="input")
        self.output = nengo.Node(size_in=self.size_out, label="output")
        self.x = nengo.networks.EnsembleArray(
            self.n_neurons, self.size_state, ens_dimensions=1, **ens_kwargs)
        x_in = self.x.input
        x_out = self.x.output

        # Connect everything up using (A, B, C, D)
        self.conn_A = nengo.Connection(
            x_out, x_in, transform=A, synapse=synapse)
        self.conn_B = nengo.Connection(
            self.input, x_in, transform=B, synapse=synapse)
        self.conn_C = nengo.Connection(
            x_out, self.output, transform=C, synapse=None)
        self.conn_D = nengo.Connection(
            self.input, self.output, transform=D, synapse=None)
