import warnings
from random import random

import nengo
from nengo.params import NumberParam, Default
from nengo.synapses import SynapseParam

from nengolib.network import Network
from nengolib.signal.normalization import HankelNorm as default_normalizer
from nengolib.signal.system import LinearSystem, is_exp_stable
from nengolib.synapses.mapping import ss2sim

__all__ = ['LinearNetwork', 'default_normalizer']


class LinearNetwork(Network):
    """Network implementing a linear time-invariant (LTI) system."""

    # TODO: document the fact that 'radii' is a misnomer because the encoders
    # are on a hypercube, not a hypersphere (and for tractibility reasons).

    synapse = SynapseParam('synapse')
    input_synapse = SynapseParam('input_synapse')
    dt = NumberParam('dt', low=0, low_open=True)

    def __init__(self, sys, n_neurons, synapse, dt, radii=1.0,
                 input_synapse=Default, normalizer=default_normalizer(),
                 solver=None, label=None, seed=None, add_to_container=None,
                 **ens_kwargs):
        super(LinearNetwork, self).__init__(label, seed, add_to_container)

        # Parameter checking
        self.sys = LinearSystem(sys)
        self.n_neurons = n_neurons
        self.synapse = synapse
        self.input_synapse = (synapse if input_synapse is Default
                              else input_synapse)
        self.dt = dt
        self.radii = radii
        self.normalizer = normalizer

        if not is_exp_stable(self.sys):
            # This means certain normalizers won't work, because the worst-case
            # output is now unbounded.
            warnings.warn("system (%s) is not exponentially stable" % self.sys)

        # Obtain a normalized state-space representation
        self.normalized, self.info = self.normalizer(self.sys, self.radii)
        self.A, self.B, self.C, self.D = ss2sim(
            self.normalized, self.synapse, self.dt).ss
        self.size_in = self.B.shape[1]
        self.size_state = len(self.A)
        self.size_out = len(self.C)

        with self:
            # Create internal Nengo objects
            self.input = nengo.Node(size_in=self.size_in, label="input")
            self.output = nengo.Node(size_in=self.size_out, label="output")
            self.x = nengo.networks.EnsembleArray(
                self.n_neurons, self.size_state, ens_dimensions=1,
                **ens_kwargs)

            if solver is not None:
                # https://github.com/nengo/nengo/issues/1044
                assert not hasattr(solver, '_hack')
                solver._hack = random()

                # https://github.com/nengo/nengo/issues/1040
                self.x.add_output('output', function=None, solver=solver)

            # Connect everything up using (A, B, C, D)
            self.conn_A = nengo.Connection(
                self.x.output, self.x.input, transform=self.A,
                synapse=self.synapse)
            self.conn_B = nengo.Connection(
                self.input, self.x.input, transform=self.B,
                synapse=self.input_synapse)
            self.conn_C = nengo.Connection(
                self.x.output, self.output, transform=self.C,
                synapse=None)
            self.conn_D = nengo.Connection(
                self.input, self.output, transform=self.D,
                synapse=None)
