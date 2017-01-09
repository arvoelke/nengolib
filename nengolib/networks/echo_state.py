import numpy as np
from numpy.linalg import eig

from nengo.params import IntParam, NumberParam
from nengo.neurons import NeuronTypeParam
from nengo.synapses import SynapseParam

import nengo
from nengolib import Network
from nengolib.neurons import Tanh
from nengolib.networks.reservoir import Reservoir

__all__ = ['EchoState']


class EchoState(Network, Reservoir):
    """An Echo State Network (ESN) within a Nengo Reservoir.

    This creates a standard Echo State Network (ENS) as a Nengo network,
    defaulting to the standard set of assumptions of non-spiking Tanh units
    and a random recurrent weight matrix [1]_. This is based on the
    minimalist Python implementation from [2]_.

    The network takes some arbitrary time-varying vector as input, encodes it
    randomly, and filters it using nonlinear units and a random recurrent
    weight matrix normalized by its spectral radius.

    This class also inherits ``nengolib.networks.Reservoir``, and thus the
    optimal linear readout is solved for in the same way: the network is
    simulated on a test signal, and then a solver is used to optimize the
    decoding connection weights.

    References:
        [1] http://www.scholarpedia.org/article/Echo_state_network
        [2] http://minds.jacobs-university.de/mantas/code
    """

    n_neurons = IntParam('n_neurons', default=None, low=1)
    dimensions = IntParam('dimensions', default=None, low=1)
    dt = NumberParam('dt', low=0, low_open=True)
    recurrent_synapse = SynapseParam('recurrent_synapse')
    gain = NumberParam('gain', low=0, low_open=True)
    neuron_type = NeuronTypeParam('neuron_type')

    def __init__(self, n_neurons, dimensions, recurrent_synapse=0.005,
                 readout_synapse=None, radii=1.0, gain=1.25, rng=None,
                 neuron_type=Tanh(), include_bias=True, ens_seed=None,
                 label=None, seed=None, add_to_container=None, **ens_kwargs):
        """Initializes the Echo State Network.

        Parameters
        ----------
        n_neurons : int
            The number of neurons to use in the reservoir.
        dimensions : int
            The dimensionality of the input signal.
        recurrent_synapse : nengo.synapses.Synapse (Default: ``0.005``)
            Synapse used to filter the recurrent connection.
        readout_synapse : nengo.synapses.Synapse (Default: ``None``)
            Optional synapse to filter all of the outputs before solving
            for the linear readout. This is included in the connection to the
            ``output`` Node created within the network.
        radii : scalar or array_like, optional (Default: ``1``)
            The radius of each dimension of the input signal, used to normalize
            the incoming connection weights.
        gain : scalar, optional (Default: ``1.25``)
            A scalar gain on the recurrent connection weight matrix.
        rng : ``numpy.random.RandomState``, optional (Default: ``None``)
            Random state used to initialize all weights.
        neuron_type : ``nengo.neurons.NeuronType`` optional \
                      (Default: ``Tanh()``)
            Neuron model to use within the reservoir.
        include_bias : ``bool`` (Default: ``True``)
            Whether to include a bias current to the neural nonlinearity.
            This should be ``False`` if the neuron model already has a bias,
            e.g., ``LIF`` or ``LIFRate``.
        ens_seed : int, optional (Default: ``None``)
            Seed passed to the ensemble of neurons.
        """

        Network.__init__(self, label, seed, add_to_container)

        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.recurrent_synapse = recurrent_synapse
        self.radii = radii  # TODO: make array or scalar parameter?
        self.gain = gain
        self.rng = np.random if rng is None else rng
        self.neuron_type = neuron_type
        self.include_bias = include_bias

        self.W_in = (
            self.rng.rand(self.n_neurons, self.dimensions) - 0.5) / self.radii
        if self.include_bias:
            self.W_bias = self.rng.rand(self.n_neurons, 1) - 0.5
        else:
            self.W_bias = np.zeros((self.n_neurons, 1))
        self.W = self.rng.rand(self.n_neurons, self.n_neurons) - 0.5
        self.W *= self.gain / max(abs(eig(self.W)[0]))

        with self:
            self.ensemble = nengo.Ensemble(
                self.n_neurons, 1, neuron_type=self.neuron_type, seed=ens_seed,
                **ens_kwargs)
            self.input = nengo.Node(size_in=self.dimensions)

            pool = self.ensemble.neurons
            nengo.Connection(
                self.input, pool, transform=self.W_in, synapse=None)
            nengo.Connection(  # note the bias will be active during training
                nengo.Node(output=1, label="bias"), pool,
                transform=self.W_bias, synapse=None)
            nengo.Connection(
                self.ensemble.neurons, pool, transform=self.W,
                synapse=self.recurrent_synapse)

        Reservoir.__init__(
            self, self.input, pool, readout_synapse=readout_synapse,
            network=self)
