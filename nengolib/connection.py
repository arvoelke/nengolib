import nengo
from nengo import Connection as BaseConnection
from nengo.solvers import LstsqL2

from nengolib.solvers import BiasedSolver

__all__ = ['Connection']


class Connection(BaseConnection):
    """Drop-in replacement for :class:`nengo.Connection`.

    Extends :class:`nengo.Connection` to improve decoding by using a bias.
    If the ``pre`` object is a :class:`nengo.Ensemble`, then a bias will
    be included in its decoders. This automatically improves the decoding of
    functions with a constant offset when the ensemble has few neurons.
    This is equivalent to a change in postsynaptic biases (in effect
    changing the representation to be centered around some constant offset,
    that is discovered optimally).

    Parameters
    ----------
    pre : :class:`nengo.Ensemble` or :class:`nengo.ensemble.Neurons` or :class:`nengo.Node`
        Nengo source object for the connection.
    post : :class:`nengo.Ensemble` or :class:`nengo.ensemble.Neurons` or :class:`nengo.Node` or :class:`nengo.Probe`
        Nengo destination object for the connection.
    solver : :class:`nengo.solvers.Solver`, optional
        Solver to use for decoded bias and connection.
        Defaults to :class:`nengo.solvers.LstsqL2`.
    **kwargs : ``dictionary``, optional
        Additional keyword arguments passed to :class:`nengo.Connection`.

    See Also
    --------
    :class:`nengo.Connection`
    :class:`.Network`
    """  # noqa: E501

    def __init__(self, pre, post, solver=LstsqL2(), **kwargs):
        if isinstance(pre, nengo.Ensemble):
            solver = BiasedSolver(solver)
            bias = nengo.Node(output=solver.bias_function(post.size_in),
                              label="Bias")
            nengo.Connection(bias, post, synapse=None)

        super(Connection, self).__init__(pre, post, solver=solver, **kwargs)
