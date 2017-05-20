import logging
import warnings
from random import random

import numpy as np

import nengo
from nengo.params import NumberParam, Default
from nengo.synapses import SynapseParam

from nengolib.network import Network
from nengolib.signal.realizers import Hankel as default_realizer
from nengolib.signal.system import LinearSystem
from nengolib.synapses.mapping import ss2sim

__all__ = ['LinearNetwork', 'default_realizer']


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
                 realizer=default_realizer(), solver=Default,
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
        self.realizer = realizer

        if solver is not Default:
            # https://github.com/nengo/nengo/issues/1044
            solver._hack = random()

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

        # Obtain state-space transformation and realization
        self.realizer_result = self.realizer(self.sys, self.radii)

        # Map the system onto the synapse
        mapped = ss2sim(
            self.realizer_result.realization, self.synapse, self.dt)
        self.A, self.B, self.C, self.D = mapped.ss
        self.size_in = mapped.size_in
        self.size_state = len(mapped)
        self.size_out = mapped.size_out

        with self:
            # Create internal Nengo objects
            self.input = nengo.Node(size_in=self.size_in, label="input")
            self.output = nengo.Node(size_in=self.size_out, label="output")

            x_input, x_output = self._make_core(solver, **ens_kwargs)

            # Connect everything up using (A, B, C, D)
            self.conn_A = nengo.Connection(
                x_output, x_input, transform=self.A,
                synapse=self.synapse)
            self.conn_B = nengo.Connection(
                self.input, x_input, transform=self.B,
                synapse=self.input_synapse)
            self.conn_C = nengo.Connection(
                x_output, self.output, transform=self.C,
                synapse=self.output_synapse)

            if not np.allclose(self.D, 0):
                logging.info("Passthrough (%s) on LinearNetwork with sys=%s",
                             self.D, self.sys)
                self.conn_D = nengo.Connection(
                    self.input, self.output, transform=self.D,
                    synapse=None)

    def _make_core(self, solver, **ens_kwargs):
        self.x = nengo.networks.EnsembleArray(
            self.n_neurons_per_ensemble, self.size_state,
            ens_dimensions=1, **ens_kwargs)

        if solver is not Default:
            # https://github.com/nengo/nengo/issues/1040
            self.x.add_output('output', function=None, solver=solver)

        return self.x.input, self.x.output
