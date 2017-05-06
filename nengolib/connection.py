import nengo
from nengo import Connection as BaseConnection
from nengo.solvers import LstsqL2

from nengolib.solvers import BiasedSolver


__all__ = ['Connection']


class Connection(BaseConnection):
    """Extends nengo.Connection to improve decoding with a bias."""

    def __init__(self, pre, post, solver=LstsqL2(), **kwargs):
        if isinstance(pre, nengo.Ensemble):
            solver = BiasedSolver(solver)
            nengo.Connection(
                nengo.Node(output=0, label="Bias"), post,
                function=solver.bias_function(post.size_in), synapse=None)

        super(Connection, self).__init__(pre, post, solver=solver, **kwargs)
