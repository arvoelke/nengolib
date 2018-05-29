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
from nengolib.synapses.analog import PadeDelay

__all__ = ['t_default', 'readout', 'RollingWindow']

t_default = np.linspace(0, 1, 1000)  # default window time points (normalized)


def readout(q, r):
    """C matrix to decode a delay of r*theta from the delay state for theta.

    ``r`` is a ratio between 0 (``t=0``) and 1 (``t=-theta``).
    """

    c = np.zeros(q)
    for i in range(q):
        j = np.arange(i+1, dtype=np.float64)
        c[q-1-i] += 1 / binom(q, i) * np.sum(
            binom(q, j) * binom(2*q - 1 - j, i - j) * (-r)**(i - j))
    return c


class RollingWindow(LinearNetwork):
    """Approximate nonlinear functions across a rolling window of input.

    Compresses the input history of finite length into a low-dimensional state
    to support the approximation of nonlinear functions across a rolling
    window of input history.  This can be used to approximate
    FIR filters and window functions in continuous time.

    Parameters
    ----------
    theta : ``float``
        Width of rolling window (time-delay) in seconds.
    n_neurons : ``integer``
        Total number of neurons to use in :class:`nengo.Ensemble`.
    process : :class:`nengo.Process`
        Process modelling a typical input to the network.
        Used to optimize the lengths of axis-aligned ``encoders`` and
        the distribution of ``eval_points``.
        If ``None``, then will use the defaults of uniform unit-length
        ``encoders`` and uniformly distributed ``eval_points``, which
        usually gives sub-optimal performance.
    dimensions : ``integer``, optional
        Order of :func:`.PadeDelay`, or dimensionality of the state vector.
        Defaults to ``6``, which has an approximation error of less than
        one percent for input frequencies less than ``1/theta``
        (see :func:`.pade_delay_error`).
    dt : ``float``, optional
        Simulation time-step (in seconds), passed to :class:`.LinearNetwork`.
        Defaults to ``0.001``, but should be manually specified if the
        simulation time-step is ever changed.
    synapse : :class:`nengo.synapses.Synapse`, optional
        Recurrent synapse. Bigger is typically better.
        Defaults to ``0.1``.
    input_synapse : :class:`nengo.synapses.Synapse`, optional
        Input synapse. Typically should match the value of
        ``synapse``, unless the input has already been filtered by the
        same synapse. Defaults to ``0.1``.
    realizer : :class:`.AbstractRealizer`, optional
        Method of realizing the linear system. Defaults to :class:`.Balanced`.
    solver : :class:`nengo.solvers.Solver`, optional
        Solver to use for connections from the state ensemble.
        Defaults to :class:`nengo.solvers.LstsqL2` (with ``reg=1e-3``).
    **kwargs : ``dictionary``, optional
        Additional keyword arguments passed to :class:`.LinearNetwork`.
        Those that fall under the heading of ``**ens_kwargs`` will be
        passed to the :class:`nengo.Ensemble` that represents the
        :attr:`.state`.

    See Also
    --------
    :class:`.LinearNetwork`
    :func:`.PadeDelay`
    :func:`.pade_delay_error`
    :class:`.Balanced`
    :class:`.EvalPoints`
    :class:`.Encoders`

    Notes
    -----
    This extends :class:`.LinearNetwork` to efficiently implement the
    :func:`.PadeDelay` system (via :class:`.Balanced`, :class:`.EvalPoints`,
    and :class:`.Encoders`), with support for decoding nonlinear functions
    from the state of the network. [#]_

    Function are decoded from the window by evaluating the state-space
    at arbitrary points (as usual in Nengo), and linearly projecting them onto
    basis functions to express each point as a sampled window representation.
    See :func:`.add_output` for details.

    References
    ----------
    .. [#] A. R. Voelker and C. Eliasmith, "Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells",
       Neural Computation (preprint), accepted 09 2017.
       [`URL <https://github.com/arvoelke/delay2017>`__]

    Examples
    --------
    See :doc:`notebooks/examples/rolling_window` for a notebook example.

    >>> from nengolib.networks import RollingWindow, t_default

    Approximate the maximum of a window of width 50 ms, as well as a sampling
    of the window itself. The :class:`.Hankel` realizer happens to be better
    than the default of :class:`.Balanced` for computing the ``max`` function.

    >>> import nengo
    >>> from nengolib import Network
    >>> from nengolib.signal import Hankel
    >>> with Network() as model:
    >>>     process = nengo.processes.WhiteSignal(100., high=25, y0=0)
    >>>     stim = nengo.Node(output=process)
    >>>     rw = RollingWindow(theta=.05, n_neurons=2500, process=process,
    >>>                        neuron_type=nengo.LIFRate(),
    >>>                        realizer=Hankel())
    >>>     nengo.Connection(stim, rw.input, synapse=None)
    >>>     p_stim = nengo.Probe(stim)
    >>>     p_delay = nengo.Probe(rw.output)
    >>>     p_max = nengo.Probe(rw.add_output(function=np.max))
    >>>     p_window = nengo.Probe(rw.add_output(function=lambda w: w[::20]))
    >>> with nengo.Simulator(model, seed=0) as sim:
    >>>     sim.run(.5)

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(211)
    >>> plt.plot(sim.trange(), sim.data[p_stim], label="Input")
    >>> plt.plot(sim.trange(), sim.data[p_delay], label="Delay")
    >>> plt.legend()
    >>> plt.subplot(212)
    >>> plt.plot(sim.trange(), sim.data[p_window], alpha=.2)
    >>> plt.plot(sim.trange(), sim.data[p_max], c='purple', label="max(w)")
    >>> plt.legend()
    >>> plt.xlabel("Time (s)")
    >>> plt.show()

    Visualizing the canonical basis functions. The state of the
    :func:`PadeDelay` system takes a linear combination of these to
    represent the current window of history:

    >>> plt.title("canonical_basis()")
    >>> plt.plot(t_default, rw.canonical_basis())
    >>> plt.xlabel("Normalized Time (Unitless)")
    >>> plt.show()

    Visualizing the realized basis functions. This is a linear transformation
    of the above basis functions according to the realized state-space
    (see ``realizer`` parameter). The state of the **network** takes a linear
    combination of these to represent the current window of history:

    >>> plt.title("basis()")
    >>> plt.plot(t_default, rw.basis())
    >>> plt.xlabel("Normalized Time (Unitless)")
    >>> plt.show()

    Visualizing the inverse basis functions. The functions that can be
    accurately decoded are expressed in terms of the dot-product of the window
    with these functions (see :func:`.add_output` for mathematical details).

    >>> plt.title("inverse_basis().T")
    >>> plt.plot(t_default, rw.inverse_basis().T)
    >>> plt.xlabel("Normalized Time (Unitless)")
    >>> plt.show()
    """

    # TODO: see notebook for more details.

    def __init__(self, theta, n_neurons, process, dimensions=6,
                 synapse=0.1, input_synapse=0.1, dt=0.001, realizer=Balanced(),
                 solver=nengo.solvers.LstsqL2(reg=1e-3), **kwargs):
        self.theta = theta
        self.process = process
        self.dimensions = dimensions

        super(RollingWindow, self).__init__(
            sys=PadeDelay(theta, order=dimensions),
            n_neurons_per_ensemble=n_neurons,
            input_synapse=input_synapse,
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
            X = self.realization.X  # set by LinearNetwork
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

        self._state = nengo.Ensemble(
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
        """Temporal basis functions for PadeDelay in canonical form."""
        t = np.atleast_1d(t)
        B = np.asarray([readout(self.dimensions, r) for r in t])
        return B

    def basis(self, t=t_default):
        """Temporal basis functions for realized PadeDelay."""
        # Undo change of basis from realizer, and then transform into window
        B = self.canonical_basis(t)
        return B.dot(self.realizer_result.T)

    def inverse_basis(self, t=t_default):
        """Moore-Penrose pseudoinverse of the basis functions."""
        B = self.basis(t)
        return inv(B.T.dot(B)).dot(B.T)

    @with_self
    def add_output(self, t=None, function=lambda w: w[-1], label='output',
                   synapse=None, **conn_kwargs):
        """Decodes a function of the window at time points ``-t*theta``.

        Parameters
        ----------
        t : ``array_like``, optional
            A scalar or array-like with elements ranging between ``0``
            (beginning of window) and ``1`` (end of window; i.e., ``theta``).
            Specifies the time-points at which to sample the window's basis
            functions. Defaults to the value of :data:`.t_default`:
            ``1000`` points spaced evenly between ``0`` and ``1``.
        function : ``callable``, optional
            A function that consumes some window ``w`` of requested time-points
            (``len(w) == len(t)``), and returns the desired decoding.
            Defaults to returning the end of the window: ``w[-1]``.
        label : ``string``, optional
            Label for the created :class:`nengo.Node`.
            Defaults to ``'output'``.
        synapse : :class:`nengo.synapses.Synapse`, optional
            Synapse passed to created :class:`nengo.Connection`.
            Defaults to ``None``.
        **conn_kwargs : ``dictionary``, optional
            Additional keyword arguments passed to :class:`nengo.Connection`.

        Returns
        -------
        :class:`nengo.Node`
            Node object that holds the decoded function from the state
            ensemble. The size of the node is equal to the output
            dimensionality of the provided function.

        Notes
        -----
        The approach is to project the state-vector onto the basis functions
        given by the rows of :func:`.basis`, and then supply the resulting
        window representation to the given ``function``. Then we solve for
        the decoders with respect to the state-vector as usual in Nengo.

        Disregarding the linear change of basis from the ``realizer``, the
        :func:`.canonical_basis` functions are polynomials of increasing
        order (from ``0`` up to ``q-1``, where ``q=dimensions``):

        .. math::

            P_i(t) = \\begin{pmatrix}q \\\\ i\\end{pmatrix}^{-1}
                \\sum_{j=0}^i \\begin{pmatrix}q \\\\ j\\end{pmatrix}
                \\begin{pmatrix}2q - 1 - j \\\\ i - j\\end{pmatrix}
                \\left( -t \\right)^{i - j}
                \\text{,} \\quad 0 \\le t \\le 1
                \\text{,} \\quad i = 0 \\ldots q - 1 \\text{.}

        Since the encoders are axis-aligned (when a ``process`` is given),
        the functions that can be accurately decoded by this approach are of
        the form:

        .. math::

           f({\\bf w}) = \\sum_{i=0}^{q-1} f_i ({\\bf v}_i \\cdot {\\bf w})

        where :math:`{\\bf w}` is some history, :math:`{\\bf v}_i` are the
        columns of :func:`.inverse_basis`, and each :math:`f_i` is some
        unknown low-degree nonlinearity.
        """

        if t is None:
            t = t_default

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

        output = nengo.Node(size_in=np.asarray(value).size, label=label)
        nengo.Connection(self.state, output, function=wrapped_function,
                         synapse=synapse, **conn_kwargs)
        return output
