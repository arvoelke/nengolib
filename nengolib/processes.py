import numpy as np

from nengo.base import Process
from nengo.exceptions import ValidationError
from nengo.utils.stdlib import checked_call

__all__ = ['Callable']


class Callable(Process):
    """Wrapper to convert any callable into a Process."""

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
