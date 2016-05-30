import warnings

import numpy as np

import nengo
from nengo.base import NengoObject, ObjView
from nengo.ensemble import Neurons
from nengo.exceptions import NetworkContextError
from nengo.solvers import LstsqL2
from nengo.synapses import SynapseParam
from nengo.utils.compat import is_iterable

__all__ = ['Reservoir']


def _to_list(x):
    """Copies x into a list."""
    return list(x) if is_iterable(x) else [x]


class Reservoir(object):
    """A collection of inputs and outputs within some network.

    This class encapsulates:
        - any number of Nengo objects that take some input, e.g. Nodes or
          Ensembles;
        - any number of Nengo objects that produce some output, e.g. Neurons;
        - a network that these objects can be simulated inside;
        - a synapse used to filter the outputs.

    The network can then be trained by injecting a signal to the specified
    inputs, and solving for the optimal linear readout from the outputs.
    After training, the optimal readout will be available by connecting from
    the ``output`` Node in the model.

    This technique is known as reservoir computing, and this class generalizes
    the concept to support NEF dynamics with structured reservoirs, in
    addition to traditional spiking and non-spiking random pools.

    The inputs, outputs, and any internal objects that will vary randomly
    between builds must be given a fixed seed so that they do not differ
    between training and testing.

    See ``doc/notebooks/examples/reservoir.ipynb`` for more information.
    """

    readout_synapse = SynapseParam('readout_synapse')
    _connectable = (NengoObject, ObjView, Neurons)

    def __init__(self, inputs, outputs, readout_synapse=None, network=None):
        """Builds a reservoir containing inputs and outputs.

        Parameters
        ----------
        inputs : nengo.NengoObject, nengo.ObjView, nengo.Neurons, or iterable
            Input (or inputs) within the network, to receive the input signal.
        outputs : nengo.NengoObject, nengo.ObjView, nengo.Neurons, or iterable
            Output (or outputs) within the network, for the linear readout.
        readout_synapse : nengo.synapses.Synapse (Default: ``None``)
            Optional synapse to filter all of the outputs before solving
            for the linear readout. This is included in the connection to the
            ``output`` Node created within the network.
        network : nengo.Network, optional (Default: ``None``)
            The Nengo network that contains all of the inputs and outputs,
            that can be simulated on its own. If ``None`` is supplied, then
            this will automatically use the current network context.
        """

        self.inputs = _to_list(inputs)
        self.outputs = _to_list(outputs)
        self.readout_synapse = readout_synapse

        # Determine dimensionality of reservoir
        self.size_in = 0
        for obj in self.inputs:
            if not isinstance(obj, self._connectable):
                raise TypeError(
                    "inputs (%s) must be connectable Nengo object" % (inputs,))

            # Increment input size of reservoir
            self.size_in += obj.size_in

        if self.size_in == 0:
            raise ValueError(
                "inputs (%s) must contain at least one input dimension" % (
                    inputs,))

        self.size_mid = 0
        for obj in self.outputs:
            if not isinstance(obj, self._connectable):
                raise TypeError(
                    "outputs (%s) must be connectable Nengo object" % (
                        outputs,))

            # Increment output size of reservoir
            self.size_mid += obj.size_out

        if self.size_mid == 0:
            raise ValueError(
                "outputs (%s) must contain at least one output dimension" % (
                    outputs,))

        # Determine simulation context
        if network is None:
            if not len(nengo.Network.context):
                raise NetworkContextError(
                    "reservoir must be created within a network block if the "
                    "given network parameter is None")
            self.network = nengo.Network.context[-1]
        else:
            self.network = network

        with self.network:
            # Create a node whichs disperses all of the inputs
            self._proxy_in = nengo.Node(size_in=self.size_in)
            in_used = 0
            for obj in self.inputs:
                nengo.Connection(
                    self._proxy_in[in_used:in_used+obj.size_in], obj,
                    synapse=None)
                in_used += obj.size_in
            assert in_used == self.size_in

            # Create a node which collects all of the reservoir outputs
            self._proxy_mid = nengo.Node(size_in=self.size_mid)
            mid_used = 0
            for obj in self.outputs:
                nengo.Connection(
                    obj, self._proxy_mid[mid_used:mid_used+obj.size_out],
                    synapse=None)
                mid_used += obj.size_out
            assert mid_used == self.size_mid

            # Create a dummy node to hold the eventually learned output
            # It will be the 0 scalar until the train method is called
            self.output = nengo.Node(size_in=1)
            self._readout = nengo.Connection(
                self._proxy_mid, self.output, synapse=self.readout_synapse,
                transform=np.zeros((1, self.size_mid)))
            self.size_out = None

    def run(self, t, dt, process, seed=None):
        """Simulate the network on a particular input signal.

        If the network has been trained, this will include the decoded output.

        Parameters
        ----------
        t : float
            A positive number indicating how long the input signal should be
            in simulation seconds.
        dt : float
            A positive number indicating the time elapsed between each
            timestep. The length of each output will be ``int(t / dt)``.
        process : nengo.Process
            An autonomous process that provides a training signal of
            appropriate dimensionality to match the input objects.
        seed : int, optional (Default: ``None``)
            Seed used to initialize the simulator.
        """

        # Setup a sandbox so that the reservoir doesn't keep the
        # input connections and probes added here
        with nengo.Network(add_to_container=False) as sandbox:
            sandbox.add(self.network)

            stim = nengo.Node(output=process, size_out=self.size_in)
            nengo.Connection(stim, self._proxy_in, synapse=None)
            p_in = nengo.Probe(self._proxy_in, synapse=None)
            p_mid = nengo.Probe(self._proxy_mid, synapse=self.readout_synapse)
            p_out = nengo.Probe(self.output, synapse=None)

        with nengo.Simulator(sandbox, dt=dt, seed=seed) as sim:
            sim.run(t, progress_bar=None)

        return sim, (sim.data[p_in], sim.data[p_mid], sim.data[p_out])

    def train(self, function, t, dt, process, seed=None,
              t_init=0, solver=LstsqL2(), rng=None):
        """Train an optimal linear readout.

        Afterwards, the decoded and filtered output will be available in the
        model by connecting from the ``output`` Node, or by invoking the
        ``run`` method.

        Parameters
        ----------
        function : callable
            A function that maps the input signal obtained from simulating the
            process (as an ``M``-by-``D`` array, where ``M`` is the number of
            timesteps, and ``D`` is the input dimensionality), to the desired
            signal (of the same shape).
        t : float
            A positive number indicating how long the training signal should be
            in simulation seconds.
        dt : float
            A positive number indicating the time elapsed between each
            timestep. The length of the test signal will be ``int(t / dt)``.
        process : nengo.Process
            An autonomous process that provides a training signal of
            appropriate dimensionality to match the input objects.
        seed : int, optional (Default: ``None``)
            Seed used to initialize the simulator.
        t_init : int, optional (Default: ``0``)
            The number of seconds to discard from the start.
        solver : nengo.solvers.Solver (Default: ``nengo.solvers.LstsqL2()``)
            Solves for ``D`` such that ``AD ~ Y``.
        rng : ``numpy.random.RandomState``, optional (Default: ``None``)
            Random state passed to the solver.
        """

        # Do a safety check for seeds. Note that even if the overall
        # network has a seed, that doesn't necessarily mean everything will
        # be good, because adding components to the network after training
        # may result in seeds being shuffled around within the objects.
        for ens in self.network.all_ensembles:
            if ens.seed is None:
                warnings.warn("reservoir ensemble (%s) should have its own "
                              "seed to help ensure that its parameters do not "
                              "change between training and testing" % ens)

        sim, (data_in, data_mid, _) = self.run(t, dt, process, seed)

        target = np.atleast_1d(function(data_in))
        if target.ndim == 1:
            target = target[:, None]
        if len(data_in) != len(target):
            raise RuntimeError(
                "function expected to return signal of length %d, received %d "
                "instead" % (len(data_in), len(target)))

        offset = int(t_init / dt)
        decoders, info = solver(data_mid[offset:], target[offset:], rng=rng)

        # Update dummy node
        self.output.size_in = self.output.size_out = self.size_out = (
            target.shape[1])
        self._readout.transform = decoders.T

        info.update({'sim': sim, 'data_in': data_in, 'data_mid': data_mid})
        return decoders, info
