import warnings
from random import random

import nengo
from nengo.params import NumberParam, Default
from nengo.synapses import SynapseParam

from nengolib.network import Network
from nengolib.signal.normalization import HankelNorm as default_normalizer
from nengolib.signal.system import LinearSystem
from nengolib.synapses.mapping import ss2sim

__all__ = ['LinearNetwork', 'default_normalizer']


class LinearNetwork(Network):
    """Network implementing a linear time-invariant (LTI) system."""

    # TODO: document the fact that 'radii' is a misnomer because the encoders
    # are on a hypercube, not a hypersphere (and for tractibility reasons).

    synapse = SynapseParam('synapse')
    input_synapse = SynapseParam('input_synapse', optional=True)
    output_synapse = SynapseParam('input_synapse', optional=True)
    dt = NumberParam('dt', low=0, low_open=True, optional=True)

    def __init__(self, sys, n_neurons_per_ensemble, synapse, dt, radii=1.0,
                 input_synapse=None, output_synapse=None,
                 normalizer=default_normalizer(), solver=Default,
                 label=None, seed=None, add_to_container=None, **ens_kwargs):
        super(LinearNetwork, self).__init__(label, seed, add_to_container)

        # Parameter checking
        self.sys = LinearSystem(sys)
        self.n_neurons_per_ensemble = n_neurons_per_ensemble
        self.synapse = synapse
        self.dt = dt
        self.radii = radii
        self.input_synapse = input_synapse
        self.output_synapse = output_synapse
        self.normalizer = normalizer

        if len(self.sys) == 0:
            raise ValueError("system (%s) is zero order" % self.sys)

        if self.sys.has_passthrough and self.output_synapse is None:
            # the user shouldn't filter the output node themselves. an
            # output synapse should be given so we can do it before the
            # passthrough.
            warnings.warn("output_synapse should be given if the system has "
                          "a passthrough, otherwise filtering the output will "
                          "also filter the passthrough")

        if not self.sys.is_stable:
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
                self.n_neurons_per_ensemble, self.size_state,
                ens_dimensions=1, **ens_kwargs)

            if solver is not Default:
                # https://github.com/nengo/nengo/issues/1044
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
                synapse=self.output_synapse)
            self.conn_D = nengo.Connection(
                self.input, self.output, transform=self.D,
                synapse=None)
