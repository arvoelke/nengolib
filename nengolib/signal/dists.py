import numpy as np

from nengo.base import Process
from nengo.dists import Choice, Distribution
from nengo.exceptions import ValidationError

from nengolib.signal.system import LinearSystem

__all__ = ['EvalPoints', 'Encoders']


class EvalPoints(Distribution):
    """Samples the output of a LinearSystem given some input process.

    This can be used to sample the evaluation points according to some
    filtered process. Used by :class:`.RollingWindow`.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    process : :class:`nengo.Process`
       Nengo process to simulate.
    n_steps : ``integer``, optional
       Number of steps to simulate the process. Defaults to ``10000``.
    dt : ``float``, optional
       Process and system simulation time-step.
       Defaults to ``process.default_dt``.
    **run_steps_kwargs : ``dictionary``, optional
       Additional keyword arguments for ``process.run_steps``.

    See Also
    --------
    :class:`.Encoders`
    :class:`.Callable`
    :class:`.RollingWindow`
    :class:`nengo.Ensemble`
    :class:`nengo.dists.Distribution`

    Notes
    -----
    For ideal sampling, the given ``process`` should be aperiodic across the
    interval of time specified by ``n_steps`` and ``dt``, and moreover
    the sampled ``num`` (number of evaluation points) should not
    exceed ``n_steps``.

    Examples
    --------
    >>> from nengolib.signal import EvalPoints

    Sampling from the state-space of an alpha synapse given band-limited
    white noise:

    >>> from nengolib import Alpha
    >>> from nengo.processes import WhiteSignal
    >>> eval_points = EvalPoints(Alpha(.5).X, WhiteSignal(10, high=20))

    >>> import matplotlib.pyplot as plt
    >>> from seaborn import jointplot
    >>> jointplot(*eval_points.sample(1000, 2).T, kind='kde')
    >>> plt.show()
    """

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
        """Samples ``n`` points in ``d`` dimensions."""
        y = self._sample(d, rng)
        choices = rng.choice(len(y), size=num, replace=True)
        return y[choices]


class Encoders(EvalPoints):
    """Samples axis-aligned encoders from the maximum radii of a LinearSystem.

    Given some input process. This is useful when setting ``encoders`` in
    conjunction with ``normalize_encoders=False``, in order to set the
    *effective* radius of each dimension to its maximum absolute value.
    The evaluation points should still be sampled from within this radii.
    Paramters are the same as :class:`.EvalPoints`.
    Used by :class:`.RollingWindow`.

    Parameters
    ----------
    sys : :data:`linear_system_like`
       Linear system representation.
    process : :class:`nengo.Process`
       Nengo process to simulate.
    n_steps : ``integer``, optional
       Number of steps to simulate the process. Defaults to ``10000``.
    dt : ``float``, optional
       Process and system simulation time-step.
       Defaults to ``process.default_dt``.
    **run_steps_kwargs : ``dictionary``, optional
       Additional keyword arguments for ``process.run_steps``.

    See Also
    --------
    :class:`.EvalPoints`
    :class:`.Callable`
    :class:`.RollingWindow`
    :class:`nengo.Ensemble`
    :class:`nengo.dists.Distribution`

    Notes
    -----
    The option ``normalize_encoders=False``, required to make this useful,
    is only available in ``nengo>=2.4.0``. Otherwise these encoders will
    end up being equivalent to ``nengo.dists.Choice(np.vstack([I, -I]))``
    (i.e., unit-length and axis-aligned).
    """

    def sample(self, num, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        y = self._sample(d, rng)
        r = 1./np.max(np.abs(y), axis=0)
        assert r.shape == (d,)
        rI = r*np.eye(d)
        return Choice(np.vstack([rI, -rI])).sample(num, d=d, rng=rng)
