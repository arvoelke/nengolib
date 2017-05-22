import warnings

import numpy as np

from scipy.linalg import inv
from scipy.special import binom

import nengo
from nengo.params import Default
from nengo.exceptions import ValidationError
from nengo.utils.network import with_self
from nengo.utils.stdlib import checked_call

from nengolib.networks.linear_network import LinearNetwork
from nengolib.signal.dists import EvalPoints, Encoders
from nengolib.signal.realizers import Balanced
from nengolib.synapses.analog import PureDelay

__all__ = ['t_default', 'readout', 'RollingWindow']

t_default = np.linspace(0, 1, 1000)  # default window time points (normalized)


def readout(q, r):
    """C matrix to decode a delay of r*theta from the delay state for theta.

    ``r`` is a ratio between 0 (``t=0``) and 1 (``t=-theta``).
    Equation is taken directly from [1]_.

    References
    ----------
    .. [1] A. R. Voelker and C. Eliasmith, "Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells,"
       2017, Submitted.
    """
    c = np.zeros(q)
    for i in range(q):
        j = np.arange(i+1, dtype=np.float64)
        c[q-1-i] += 1 / binom(q, i) * np.sum(
            binom(q, j) * binom(2*q - 1 - j, i - j) * (-r)**(i - j))
    return c


class RollingWindow(LinearNetwork):
    """Compute nonlinear functions across a rolling window of input history."""

    def __init__(self, theta, n_neurons, dimensions=6, process=None,
                 synapse=0.1, dt=0.001, realizer=Balanced(),
                 solver=nengo.solvers.LstsqL2(reg=1e-2), **kwargs):
        self.theta = theta
        self.dimensions = dimensions
        self.process = process

        super(RollingWindow, self).__init__(
            sys=PureDelay(theta, order=dimensions),
            n_neurons_per_ensemble=n_neurons,
            input_synapse=synapse,
            synapse=synapse,
            dt=dt,
            realizer=realizer,
            solver=solver,
            **kwargs)

    def _make_core(self, solver, **ens_kwargs):
        if self.process is not None:  # set by RollingWindow.__init__
            for illegal in ('eval_points', 'encoders', 'normalize_encoders'):
                if illegal in ens_kwargs and \
                   ens_kwargs[illegal] is not Default:
                    raise ValidationError(
                        "'%s' must not be given (%s) if 'process' is not "
                        "None." % (illegal, ens_kwargs[illegal]),
                        attr=illegal, obj=self)

            # Wrap the realized state by an eval_point and encoder process
            # This is done here automatically for convenience, but finer-grain
            # control can be achieved by passing in your own eval_points and
            # encoders and keeping process=None. These can also be set
            # directly on self.state after initialization, and any previously
            # created connections will still inherit the new eval_points.
            X = self.realizer_result.realization.X  # set by LinearNetwork
            ens_kwargs['eval_points'] = EvalPoints(X, self.process, dt=self.dt)
            ens_kwargs['encoders'] = Encoders(X, self.process, dt=self.dt)
            if nengo.version.version_info >= (2, 4, 0):
                ens_kwargs['normalize_encoders'] = False

            else:  # pragma: no cover
                warnings.warn(
                    "'normalize_encoders' is not supported by nengo<=%s, and "
                    "so the 'radii' for the representation cannot be "
                    "automatically optimized; try tweaking the 'radii' "
                    "manually, or upgrading to nengo>=2.4.0." %
                    nengo.__version__, UserWarning)

        self.state = nengo.Ensemble(
            n_neurons=self.n_neurons_per_ensemble,
            dimensions=self.size_state,
            label="state",
            **ens_kwargs)

        # For plausibility, only linear transformations should be made from
        # the output node. Nonlinear transformations should be decoded from
        # the state via self.add_output(...).
        output = nengo.Node(size_in=self.size_state)
        nengo.Connection(self.state, output, synapse=None, solver=solver)
        return self.state, output

    def canonical_basis(self, t=t_default):
        """Temporal basis functions for PureDelay in canonical form."""
        t = np.atleast_1d(t)
        B = np.asarray([readout(self.dimensions, r) for r in t])
        return B

    def basis(self, t=t_default):
        """Temporal basis functions for realized PureDelay."""
        # Undo change of basis from realizer, and then transform into window
        B = self.canonical_basis(t)
        return B.dot(self.realizer_result.T)

    def inverse_basis(self, t=t_default):
        """Moore-Penrose pseudoinverse of the basis functions."""
        B = self.basis(t)
        return inv(B.T.dot(B)).dot(B.T)

    @with_self
    def add_output(self, name='output', t=t_default, function=lambda w: w,
                   synapse=None, **conn_kwargs):
        """Decodes a function of the window at time points ``-t*theta``.

        ``t`` is a scalar or array-like with elements ranging between 0
        (beginning of window) and 1 (end of window; i.e., ``theta``).

        For example, when ``t=1`` and ``function`` is the identity,
        this is equivalent to decoding a delay of ``theta``.
        """
        B = self.basis(t)

        def wrapped_function(x):
            w = B.dot(x)  # state -> window
            return function(w)

        value, invoked = checked_call(
            function, np.zeros(B.shape[0]))
        if not invoked:
            raise ValidationError(
                "'function' (%s) must accept a single np.array argument of "
                "size=%d." % (function, B.shape[0]),
                attr='function', obj=self)

        output = nengo.Node(size_in=np.asarray(value).size, label=name)
        setattr(self, name, output)

        nengo.Connection(self.state, output, function=wrapped_function,
                         synapse=synapse, **conn_kwargs)
        return output
