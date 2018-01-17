import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Reset
from nengo.learning_rules import LearningRuleType
from nengo.synapses import SynapseParam, Lowpass
from nengo.version import version_info

__all__ = ['RLS']


class RLS(LearningRuleType):
    """Recursive least-squares rule for online decoder optimization.

    This may be used to learn the weights on a :class:`nengo.Connection`,
    online, in an L2-optimal manner. To be applied in the same scenarios as
    :class:`nengo.PES`, to minimize some error signal.

    In the end, the only real difference between RLS learning and using the
    :class:`nengo.solvers.LstsqL2` solver, is *when* the learning takes
    place. In the former case, the weights are learned online from an error
    signal over time, whereas in the latter case, the weights are learned
    offline in a batch optimization from the provided training data
    (``eval_points`` and ``function``).

    The cost of RLS is :math:`\\mathcal{O}\\left(n^2\\right)` extra
    time and memory. It is typically much more efficient to do the learning
    offline using the :class:`nengo.solvers.LstsqL2` solver.

    Parameters
    ----------
    learning_rate : ``float``, optional
        Effective learning rate. This is better understood as
        :math:`\\frac{1}{\\alpha}`, where :math:`\\alpha` is an
        L2-regularization term. A large learning rate means little
        regularization, which implies quick over-fitting. A small learning
        rate means large regularization, which translates to slower
        learning. Defaults to 1.0. [#]_
    pre_synapse : :class:`nengo.synapses.Synapse`, optional
        Filter applied to the pre-synaptic neural activities, for the
        purpose of applying the weight update.
        Defaults to a :class:`nengo.Lowpass` filter with a time-constant of
        5 ms.

    See Also
    --------
    :class:`nengo.PES`
    :class:`nengo.solvers.LstsqL2`
    :class:`.Temporal`

    Notes
    -----
    RLS works by maintaining the inverse neural correlation matrix,
    :math:`\\Gamma^{-1}`, where :math:`\\Gamma = A^T A + \\alpha I` are the
    regularized correlations, :math:`A` is a matrix of (possibly filtered)
    neural activities, and :math:`\\alpha` is an L2-regularization term
    controlled by the ``learning_rate``. This matrix is used to project the
    error signal and update the weights to be L2-optimal, at each time-step.

    The time-step does not play a role in this learning rule, apart from
    determining the time-scale over which the ``pre_synapse`` is discretized.
    A complete learning update is applied on every time-step.

    Attributes that can be probed from this learning rule:
    ``pre_filtered``, ``error``, ``delta``, ``inv_gamma``.

    References
    ----------
    .. [#] Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns
       of activity from chaotic neural networks. Neuron, 63(4), 544-557.

    Examples
    --------
    See :doc:`notebooks/examples/full_force_learning` for an example of how to
    use RLS to learn spiking FORCE [1]_ and "full-FORCE" networks in Nengo.

    Below, we compare :class:`nengo.PES` against :class:`.RLS`, learning a
    feed-forward communication channel (identity function), online,
    and starting with 100 spiking LIF neurons from scratch (zero weights).
    A faster learning rate for :class:`nengo.PES` results in over-fitting to
    the most recent online example, while a slower learning rate does not
    learn quickly enough. This is a general problem with greedy optimization.
    :class:`.RLS` performs better since it is L2-optimal.

    >>> from nengolib import RLS, Network
    >>> import nengo
    >>> from nengo import PES
    >>> tau = 0.005
    >>> learning_rules = (PES(learning_rate=1e-3, pre_tau=tau),
    >>>                   RLS(learning_rate=1e-5, pre_synapse=tau))

    >>> with Network() as model:
    >>>     u = nengo.Node(output=lambda t: np.sin(2*np.pi*t))
    >>>     probes = []
    >>>     for lr in learning_rules:
    >>>         e = nengo.Node(size_in=1,
    >>>                        output=lambda t, e: e if t < 1 else 0)
    >>>         x = nengo.Ensemble(100, 1, seed=0)
    >>>         y = nengo.Node(size_in=1)
    >>>
    >>>         nengo.Connection(u, e, synapse=None, transform=-1)
    >>>         nengo.Connection(u, x, synapse=None)
    >>>         conn = nengo.Connection(
    >>>             x, y, synapse=None, learning_rule_type=lr,
    >>>             function=lambda _: 0)
    >>>         nengo.Connection(y, e, synapse=None)
    >>>         nengo.Connection(e, conn.learning_rule, synapse=tau)
    >>>         probes.append(nengo.Probe(y, synapse=tau))
    >>>     probes.append(nengo.Probe(u, synapse=tau))

    >>> with nengo.Simulator(model) as sim:
    >>>     sim.run(2.0)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sim.trange(), sim.data[probes[0]],
    >>>          label=str(learning_rules[0]))
    >>> plt.plot(sim.trange(), sim.data[probes[1]],
    >>>          label=str(learning_rules[1]))
    >>> plt.plot(sim.trange(), sim.data[probes[2]],
    >>>          label="Ideal", linestyle='--')
    >>> plt.vlines([1], -1, 1, label="Training -> Testing")
    >>> plt.ylim(-2, 2)
    >>> plt.legend(loc='upper right')
    >>> plt.xlabel("Time (s)")
    >>> plt.show()
    """

    modifies = 'decoders'
    probeable = ('pre_filtered', 'error', 'delta', 'inv_gamma')

    pre_synapse = SynapseParam('pre_synapse', readonly=True)

    def __init__(self, learning_rate=1.0, pre_synapse=Lowpass(tau=0.005)):
        if version_info >= (2, 4, 1):
            # https://github.com/nengo/nengo/pull/1310
            super(RLS, self).__init__(learning_rate, size_in='post_state')
        else:  # pragma: no cover
            self.error_type = 'decoded'
            super(RLS, self).__init__(learning_rate)

        self.pre_synapse = pre_synapse

    def __repr__(self):
        return "%s(learning_rate=%r, pre_synapse=%r)" % (
            type(self).__name__, self.learning_rate, self.pre_synapse)


class SimRLS(Operator):
    """Nengo backend operator responsible for simulating RLS."""

    def __init__(self, pre_filtered, error, delta, inv_gamma, tag=None):
        super(SimRLS, self).__init__(tag=tag)

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = [delta, inv_gamma]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def inv_gamma(self):
        return self.updates[1]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def error(self):
        return self.reads[1]

    def _descstr(self):
        return 'pre=%s > %s' % (self.pre_filtered, self.delta)

    def make_step(self, signals, dt, rng):
        r = signals[self.pre_filtered]
        delta = signals[self.delta]
        error = signals[self.error]
        P = signals[self.inv_gamma]

        def step_simrls():
            # Note: dt is not used in learning rule
            rP = r.T.dot(P)
            P[...] -= np.outer(P.dot(r), rP) / (1 + rP.dot(r))
            delta[...] = -np.outer(error, P.dot(r))
        return step_simrls


@Builder.register(RLS)
def build_rls(model, rls, rule):
    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]['out']

    pre_filtered = (pre_activities
                    if rls.pre_synapse is None
                    else model.build(rls.pre_synapse, pre_activities))

    # Create input error signal
    error = Signal(np.zeros(rule.size_in), name="RLS:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error

    # Create signal for running estimate of inverse correlation matrix
    assert pre_filtered.ndim == 1
    n_neurons = pre_filtered.shape[0]
    inv_gamma = Signal(np.eye(n_neurons) * rls.learning_rate,
                       name="RLS:inv_gamma")

    model.add_op(SimRLS(pre_filtered=pre_filtered,
                        error=error,
                        delta=model.sig[rule]['delta'],
                        inv_gamma=inv_gamma))

    # expose these for probes
    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['error'] = error
    model.sig[rule]['inv_gamma'] = inv_gamma
