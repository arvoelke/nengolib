import nengo
from nengo import Connection as BaseConnection
from nengo.solvers import LstsqL2

from nengolib.solvers import BiasedSolver


__all__ = ['Connection']


class Connection(BaseConnection):
    """Extends nengo.Connection to improve decoding with a bias."""

    def __init__(self, pre, post, solver=LstsqL2(), bias_magnitude=1.0,
                 **kwargs):
        if isinstance(pre, nengo.Ensemble):
            solver = BiasedSolver(solver, magnitude=bias_magnitude)
            nengo.Connection(
                nengo.Node(output=bias_magnitude, label="Bias"), post,
                function=solver.bias_function(post.size_in), synapse=None)

        super(Connection, self).__init__(pre, post, solver=solver, **kwargs)
