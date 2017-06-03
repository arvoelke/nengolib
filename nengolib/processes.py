import numpy as np

from nengo.base import Process
from nengo.exceptions import ValidationError
from nengo.utils.stdlib import checked_call

__all__ = ['Callable']


class Callable(Process):
    """Adapter to convert any callable into a :class:`nengo.Process`.

    Parameters
    ----------
    func : ``callable``
        A function that can be called with a single float argument (time).
    default_dt : ``float``, optional
        Default time-step for the process. Defaults to ``0.001`` seconds.
    seed : ``integer``, optional
        Seed for the process.

    See Also
    --------
    :class:`nengo.Process`

    Examples
    --------
    Making a sine wave process using a lambda:

    >>> from nengolib.processes import Callable
    >>> process = Callable(lambda t: np.sin(2*np.pi*t))

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(process.ntrange(1000), process.run_steps(1000))
    >>> plt.xlabel("Time (s)")
    >>> plt.show()
    """

    def __init__(self, func, default_dt=0.001, seed=None):
        if not callable(func):
            raise ValidationError("func (%s) must be callable" % func,
                                  attr='func', obj=self)
        self.func = func
        value, invoked = checked_call(func, 0.)
        if not invoked:
            raise ValidationError(
                "func (%s) must only take single float argument" % func,
                attr='func', obj=self)
        default_size_out = np.asarray(value).size
        super(Callable, self).__init__(
            default_size_in=0,
            default_size_out=default_size_out,
            default_dt=default_dt,
            seed=seed)

    def __repr__(self):
        return "%s(func=%r, default_dt=%r, seed=%r)" % (
            type(self).__name__, self.func, self.default_dt, self.seed)

    def make_step(self, shape_in, shape_out, dt, rng):
        if shape_in != (0,):
            raise ValidationError("shape_in must be (0,), got %s" % (
                shape_in,), attr='func', obj=self)
        if shape_out != (self.default_size_out,):
            raise ValidationError("shape_out must be (%s,), got %s" % (
                self.default_size_out, shape_out), attr='func', obj=self)

        def step(t):
            return self.func(t)
        return step
