from nengo.utils.compat import is_integer

from nengolib.signal.system import z

__all__ = ['DiscreteDelay', 'BoxFilter']


def DiscreteDelay(steps):
    """A discrete (pure) time-delay: ``z**-steps``.

    Also equivalent to ``(~z)**steps`` or ``1/z**steps``.

    Parameters
    ----------
    steps : ``integer``
        Number of time-steps to delay the input signal.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Digital filter implementing the pure delay exactly.

    See Also
    --------
    :attr:`.z`
    :func:`.PadeDelay`

    Notes
    -----
    A single step of the delay will be removed if using the ``filt`` method.
    This is done for subtle reasons of consistency with Nengo.
    The correct delay will appear when passed to :class:`nengo.Connection`.

    Examples
    --------
    Simulate a Nengo network using a discrete delay of half a second for a
    synapse:

    >>> from nengolib.synapses import DiscreteDelay
    >>> import nengo
    >>> with nengo.Network() as model:
    >>>     stim = nengo.Node(output=lambda t: np.sin(2*np.pi*t))
    >>>     p_stim = nengo.Probe(stim)
    >>>     p_delay = nengo.Probe(stim, synapse=DiscreteDelay(500))
    >>> with nengo.Simulator(model) as sim:
    >>>     sim.run(1.)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sim.trange(), sim.data[p_stim], label="Stimulus")
    >>> plt.plot(sim.trange(), sim.data[p_delay], label="Delayed")
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    if not is_integer(steps) or steps < 0:
        raise ValueError("steps (%s) must be non-negative integer" % (steps,))
    return z**-steps


def BoxFilter(width, normalized=True):
    """A discrete box-filter with a given ``width``, and optionally unit area.

    This filter is also known as a "box blur", and has the effect of
    smoothing out the input signal by taking its rolling mean over a finite
    number of time-steps. Its properties are qualitatively similar to the
    continuous-time :func:`.Lowpass`.

    Parameters
    ----------
    width : ``integer``
        Width of the box-filter (in time-steps).
    normalized : ``boolean``, optional
        If ``True``, then the height of the box-filter is ``1/width``,
        otherwise ``1``. Defaults to ``True``.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Digital system implementing the box-filter.

    See Also
    --------
    :attr:`.z`
    :func:`.Lowpass`

    Examples
    --------
    Simulate a Nengo network using a box filter of 10 ms for a synapse:

    >>> from nengolib.synapses import BoxFilter
    >>> import nengo
    >>> with nengo.Network() as model:
    >>>     stim = nengo.Node(output=lambda _: np.random.randn(1))
    >>>     p_stim = nengo.Probe(stim)
    >>>     p_box = nengo.Probe(stim, synapse=BoxFilter(10))
    >>> with nengo.Simulator(model) as sim:
    >>>     sim.run(.1)

    >>> import matplotlib.pyplot as plt
    >>> plt.step(sim.trange(), sim.data[p_stim], label="Noisy Input", alpha=.5)
    >>> plt.step(sim.trange(), sim.data[p_box], label="Box-Filtered")
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    if not is_integer(width) or width <= 0:
        raise ValueError("width (%s) must be positive integer" % (width,))
    den = DiscreteDelay(width - 1)
    amplitude = 1. / width if normalized else 1.
    # 1 + 1/z + ... + 1/z^(steps) = (z^steps + z^(steps-1) + ... + 1)/z^steps
    return amplitude * sum(z**k for k in range(width)) * den
