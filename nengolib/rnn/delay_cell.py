from nengolib.signal import Balanced
from nengolib.synapses import PadeDelay

import numpy as np

from keras import backend as K
from keras import activations, initializers
from keras.initializers import Constant, Initializer, RandomUniform
from keras.layers import Layer


class InputScaled(Initializer):
    """Divides a constant value by the incoming dimensionality."""

    def __init__(self, value=0):
        super(InputScaled, self).__init__()
        self.value = value

    def __call__(self, shape, dtype=None):
        return K.constant(self.value / shape[0], shape=shape, dtype=dtype)


class InverseUniform(RandomUniform):
    """Reciprocal of the RandomUniform initializer."""

    def __call__(self, shape, dtype=None):
        return 1. / super(InverseUniform, self).__call__(
            shape=shape, dtype=dtype)


class DelayCell(Layer):
    """A layer of trainable low-dimensional delay systems.

    Each unit buffers its encoded input
    by internally representing a low-dimensional
    (i.e., compressed) version of the input window.

    Nonlinear decodings of this representation
    provide computations across the window, such
    as its derivative, energy, median value, etc (*).
    Note that decoders can span across all of the units.

    By default the window lengths are trained via backpropagation,
    as well as the encoding and decoding weights.

    Optionally, the state-space matrices that implement
    the low-dimensional delay system can be trained as well,
    but these are shared across all of the units in the layer.

    (*) Voelker and Eliasmith (2018). Improving spiking dynamical
    networks: Accurate delays, higher-order synapses, and time cells.
    Neural Computation, 30(3): 569-609.

    (*) Voelker and Eliasmith. "Methods and systems for implementing
    dynamic neural networks." U.S. Patent Application No. 15/243,223.
    Filing date: 2016-08-22.
    """

    def __init__(self, units, order,
                 realizer=Balanced(),
                 trainable_encoders=True,
                 trainable_theta=True,
                 trainable_decoders=True,
                 trainable_A=False,
                 trainable_B=False,
                 encoder_initializer=InputScaled(1.),         # TODO
                 theta_initializer=RandomUniform(100, 1000),  # TODO
                 decoder_initializer=None,                    # TODO
                 output_activation='tanh',
                 **kwargs):
        super(DelayCell, self).__init__(**kwargs)

        self.units = units
        self.order = order
        self.realizer = realizer
        self.trainable_encoders = trainable_encoders
        self.trainable_theta = trainable_theta
        self.trainable_decoders = trainable_decoders
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B

        # the delay length will be 1/dt=length and is trainable by
        # learning an internal "time-step", dt, initialized to 1/length
        self._realizer_result = realizer(
            PadeDelay(theta=1., order=self.order))
        self._ss = self._realizer_result.realization
        self._A = self._ss.A
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        self.encoder_initializer = initializers.get(encoder_initializer)
        self.theta_initializer = initializers.get(theta_initializer)

        if decoder_initializer is None:
            assert self._C.shape == (1, self.order)
            C_full = np.zeros((self.units, self.order, self.units))
            for i in range(self.units):
                C_full[i, :, i] = self._C[0]
            decoder_initializer = Constant(
                C_full.reshape(self.units*self.order, self.units))

        self.decoder_initializer = initializers.get(decoder_initializer)
        self.output_activation = activations.get(output_activation)

        # TODO: would it be better to absorb B into the encoders and then
        # initialize it appropriately? trainable encoders+B essentially
        # does this in a low-rank way

        # if the realizer is CCF then we get the following two constraints
        # that could be useful for efficiency
        #assert np.allclose(self._ss.B[1:], 0)  # CCF
        #assert np.allclose(self._ss.B[0], self.order**2)

        self.state_size = self.units*self.order  # flattened
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # TODO: add regularizers

        self.encoders = self.add_weight(
            name='encoders',
            shape=(input_dim, self.units),
            initializer=self.encoder_initializer,
            trainable=self.trainable_encoders)

        self.theta = self.add_weight(
            name='theta',
            shape=(1, self.units, 1),
            initializer=self.theta_initializer,
            trainable=self.trainable_theta)

        self.decoders = self.add_weight(
            name='decoders',
            shape=(self.units*self.order, self.units),
            initializer=self.decoder_initializer,
            trainable=self.trainable_decoders)

        self.AT = self.add_weight(
            name='AT',
            shape=(self.order, self.order),
            initializer=Constant(self._A.T),  # note: transposed
            trainable=self.trainable_A)

        self.B = self.add_weight(
            name='B',
            shape=(1, 1, self.order),  # system is SISO
            initializer=Constant(self._B[None, None, :]),
            trainable=self.trainable_B)

        self.built = True

    def call(self, inputs, states):
        # Implements:
        #
        #     x[t+1] = (I + dt*A)x[t] + (dt*B)u[t]
        #     y[t] = g(Cx[t])
        #
        # this is Euler's discretization of:
        #
        #     theta*dx = Ax + Bu
        #
        # supposing dt=1/theta is small.

        u = K.dot(inputs, self.encoders)

        # TODO: can we avoid the reshaping?
        x = K.reshape(states[0], (-1, self.units, self.order))

        x = x + 1 / self.theta * (K.dot(x, self.AT) +
                                  self.B * K.expand_dims(u, -1))
        x = K.reshape(x, (-1, self.units*self.order))

        y = self.output_activation(K.dot(x, self.decoders))

        return y, [x]
