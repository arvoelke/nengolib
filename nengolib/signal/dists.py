import numpy as np

from nengo.base import Process
from nengo.dists import Choice, Distribution
from nengo.exceptions import ValidationError

from nengolib.signal.system import LinearSystem

__all__ = ['EvalPoints', 'Encoders']


class EvalPoints(Distribution):
    """Samples the output of a LinearSystem given some process."""

    def __init__(self, sys, process, n_steps=10000, dt=None,
                 **run_steps_kwargs):
        super(EvalPoints, self).__init__()
        self.sys = LinearSystem(sys)
        if not isinstance(process, Process):
            raise ValidationError(
                "process (%s) must be a Process" % (process,),
                attr='process', obj=self)
        self.process = process
        self.n_steps = n_steps
        if dt is None:
            dt = self.process.default_dt  # 0.001
        self.dt = dt
        self.run_steps_kwargs = run_steps_kwargs

    def __repr__(self):
        return ("%s(sys=%r, process=%r, n_steps=%r, dt=%r, **%r)" %
                (type(self).__name__, self.sys, self.process, self.n_steps,
                 self.dt, self.run_steps_kwargs))

    def _sample(self, d, rng):
        if self.sys.size_out != d:
            raise ValidationError(
                "sys.size_out (%d) must equal sample d (%s)" %
                (self.sys.size_out, d), attr='sys', obj=self)
        u = self.process.run_steps(
            self.n_steps, d=self.sys.size_in, dt=self.dt, rng=rng,
            **self.run_steps_kwargs)
        return self.sys.filt(u, dt=self.dt)

    def sample(self, num, d=1, rng=np.random):
        y = self._sample(d, rng)
        choices = rng.choice(len(y), size=num, replace=True)
        return y[choices]


class Encoders(EvalPoints):
    """Samples encoders from the maximum radii of a LinearSystem."""

    def sample(self, num, d=1, rng=np.random):
        y = self._sample(d, rng)
        r = 1./np.max(np.abs(y), axis=0)
        assert r.shape == (d,)
        rI = r*np.eye(d)
        return Choice(np.vstack([rI, -rI])).sample(num, d=d, rng=rng)
