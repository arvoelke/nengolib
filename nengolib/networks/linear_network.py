import logging
import warnings
from random import random

import numpy as np

import nengo
from nengo.params import NumberParam, Default
from nengo.synapses import SynapseParam

from nengolib.network import Network
from nengolib.signal.realizers import Hankel
from nengolib.signal.system import LinearSystem
from nengolib.synapses.mapping import ss2sim

__all__ = ['LinearNetwork']


class LinearNetwork(Network):
    """Network implementing a linear time-invariant (LTI) system.

    This network implements the following linear state-space model:

    .. math::

       \\dot{{\\bf x}}(t) &= A{\\bf x}(t) + B{\\bf u}(t) \\\\
              {\\bf y}(t) &= C{\\bf x}(t) + D{\\bf u}(t)

    This works by first realizing a state-space representation from the
    given ``sys`` and ``realizer``, and then using :func:`.ss2sim` to apply
    a generalization of Principle 3 from the Neural Engineering Framework (NEF)
    to map the system onto the given ``synapse``. This yields a :attr:`.mapped`
    system whose state-space matrices give the transformation matrices for the
    resulting Nengo network.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    n_neurons_per_ensemble : ``integer``
       Number of neurons to use per ensemble (i.e., per dimension).
    synapse : :class:`nengo.synapses.Synapse`
        Recurrent synapse used to implement the dynamics, passed to
        :func:`.ss2sim`.
    dt : ``float`` or ``None``
        Simulation time-step (in seconds), passed to :func:`.ss2sim`.
        If ``None``, then this uses the continuous form of Principle 3
        (i.e., assuming a continuous-time synapse with negligible time-step).
        If provided, then ``sys`` will be discretized and the discrete
        form of Principle 3 will be applied. This should always be given
        for digital simulations.
    radii : ``float`` or ``array_like``, optional
        Radius of each dimension of the realized state-space.
        If a single ``float``, then it will be applied to each dimension.
        If ``array_like``, then its length must match :attr:`.size_state`.
        Defaults to ``1``.
    input_synapse : :class:`nengo.synapses.Synapse`, optional
        Input synapse connecting from :attr:`.input` node. Defaults to ``None``
        to discourage double filtering, but should typically match the
        ``synapse`` parameter.
    output_synapse : :class:`nengo.synapses.Synapse`, optional
        Output synapse connecting to :attr:`.output` node.
        Defaults to ``None``.
    realizer : :class:`.AbstractRealizer`, optional
        Method of obtaining a state-space realization of ``sys``.
        Defaults to :class:`.Hankel`.
    solver : :class:`nengo.solvers.Solver`, optional
        Solver used to decode the state.
        Defaults to :class:`nengo.solvers.LstsqL2` (with ``reg=.1``).
    label : str, optional (Default: None)
        Name of the network.
    seed : int, optional (Default: None)
        Random number seed that will be fed to the random number generator.
        Setting the seed makes the network's build process deterministic.
    add_to_container : bool, optional (Default: None)
        Determines if this network will be added to the current container.
        If None, this network will be added to the network at the top of the
        ``Network.context`` stack unless the stack is empty.
    **ens_kwargs : ``dictionary``, optional
        Additional keyword arguments are passed to the
        :class:`nengo.networks.EnsembleArray` that represents the
        :attr:`.state`.

    See Also
    --------
    :class:`.Network`
    :class:`.RollingWindow`
    :class:`.Hankel`
    :func:`.ss2sim`

    Notes
    -----
    By linearity, the ``input_synapse`` and the ``output_synapse`` are
    interchangeable with one another. However, this will modify the
    state-space (according to these same filters) which may impact the quality
    of representation.

    Examples
    --------
    >>> from nengolib.networks import LinearNetwork
    >>> from nengolib.synapses import Bandpass

    Implementing a 5 Hz :func:`.Bandpass` filter (i.e., a decaying 2D
    oscillator) using 1000 spiking LIF neurons:

    >>> import nengo
    >>> from nengolib import Network
    >>> from nengolib.signal import Balanced
    >>> with Network() as model:
    >>>     stim = nengo.Node(output=lambda t: 100*int(t < .01))
    >>>     sys = LinearNetwork(sys=Bandpass(freq=5, Q=10),
    >>>                         n_neurons_per_ensemble=500,
    >>>                         synapse=.1, dt=1e-3, realizer=Balanced())
    >>>     nengo.Connection(stim, sys.input, synapse=None)
    >>>     p = nengo.Probe(sys.state.output, synapse=.01)
    >>> with nengo.Simulator(model, dt=sys.dt) as sim:
    >>>     sim.run(1.)

    Note there are exactly 5 oscillations within 1 second, in response to a
    saturating impulse:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(*sim.data[p].T)
    >>> plt.xlabel("$x_1(t)$")
    >>> plt.ylabel("$x_2(t)$")
    >>> plt.axis('equal')
    >>> plt.xlim(-1, 1)
    >>> plt.ylim(-1, 1)
    >>> plt.show()
    """

    synapse = SynapseParam('synapse')
    input_synapse = SynapseParam('input_synapse', optional=True)
    output_synapse = SynapseParam('input_synapse', optional=True)
    dt = NumberParam('dt', low=0, low_open=True, optional=True)

    def __init__(self, sys, n_neurons_per_ensemble, synapse, dt, radii=1.0,
                 input_synapse=None, output_synapse=None,
                 realizer=Hankel(), solver=Default,
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
        self._realizer_result = self.realizer(self.sys, self.radii)

        # Map the system onto the synapse
        self._mapped = ss2sim(self.realization, self.synapse, self.dt)

        with self:
            # Create internal Nengo objects
            self._input = nengo.Node(size_in=self.size_in, label="input")
            self._output = nengo.Node(size_in=self.size_out, label="output")

            x_input, x_output = self._make_core(solver, **ens_kwargs)

            # Connect everything up using (A, B, C, D)
            nengo.Connection(
                x_output, x_input, transform=self.A,
                synapse=self.synapse)
            nengo.Connection(
                self.input, x_input, transform=self.B,
                synapse=self.input_synapse)
            nengo.Connection(
                x_output, self.output, transform=self.C,
                synapse=self.output_synapse)

            if not np.allclose(self.D, 0):
                logging.info("Passthrough (%s) on LinearNetwork with sys=%s",
                             self.D, self.sys)
                nengo.Connection(
                    self.input, self.output, transform=self.D,
                    synapse=None)

    def _make_core(self, solver, **ens_kwargs):
        self._state = nengo.networks.EnsembleArray(
            self.n_neurons_per_ensemble, self.size_state,
            ens_dimensions=1, label="x", **ens_kwargs)

        if solver is not Default:
            # https://github.com/nengo/nengo/issues/1040
            self.state.add_output('output', function=None, solver=solver)

        return self.state.input, self.state.output

    @property
    def realizer_result(self):
        """The :class:`.RealizerResult` produced by ``realizer``."""
        return self._realizer_result

    @property
    def realization(self):
        """Realized :class:`.LinearSystem`."""
        return self.realizer_result.realization  # convenience

    @property
    def mapped(self):
        """Mapped :class:`.LinearSystem`."""
        return self._mapped

    @property
    def A(self):
        """``A`` state-space matrix of mapped :class:`.LinearSystem`."""
        return self.mapped.A

    @property
    def B(self):
        """``B`` state-space matrix of mapped :class:`.LinearSystem`."""
        return self.mapped.B

    @property
    def C(self):
        """``C`` state-space matrix of mapped :class:`.LinearSystem`."""
        return self.mapped.C

    @property
    def D(self):
        """``D`` state-space matrix of mapped :class:`.LinearSystem`."""
        return self.mapped.D

    @property
    def size_in(self):
        """Input dimensionality."""
        return self.mapped.size_in

    @property
    def size_state(self):
        """State dimensionality."""
        return len(self.mapped)

    @property
    def size_out(self):
        """Output dimensionality."""
        return self.mapped.size_out

    @property
    def input(self):
        """Nengo object representing the input ``u(t)`` to the system."""
        return self._input

    @property
    def state(self):
        """Nengo object representing the state ``x(t)`` of the system."""
        return self._state

    @property
    def output(self):
        """Nengo object representing the output ``y(t)`` of the system."""
        return self._output
