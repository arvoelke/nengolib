import numpy as np
from numpy.linalg import matrix_power

from nengolib.signal.system import LinearSystem
from nengolib.signal.discrete import cont2discrete

__all__ = ['ss2sim']


def ss2sim(sys, synapse, dt):
    """Maps a linear system onto a synapse in state-space form.

    This implements a generalization of Principle 3 from the Neural Engineering
    Framework (NEF). [#]_
    Intuitively, this routine compensates for the change in dynamics
    that occurs when the integrator that usually forms the basis for any
    linear system is replaced by the given synapse.
    This is needed because in neural systems we don't have access to a
    perfect integrator; instead the synapse model becomes the
    "dynamical primitive".

    Parameters
    ----------
    sys : :data:`linear_system_like`
        Linear system representation of desired dynamical system.
        Requires ``sys.analog == synapse.analog``.
    synapse : :data:`linear_system_like`
        Linear system representation of the synapse providing the dynamics.
        Requires ``sys.analog == synapse.analog``.
    dt : ``float`` or ``None``
        Time-step of simulation. If not ``None``, then both ``sys`` and
        ``synapse`` are discretized using the ``'zoh'`` method.
        In either case, if ``sys`` is now digital, then the digital
        generalization of Principle 3 will be applied --- otherwise the analog
        version will be applied.

    Returns
    -------
    :class:`.LinearSystem`
        Linear system whose state-space matrices yield the desired
        dynamics when using the synapse model instead of an integrator.

    See Also
    --------
    :class:`.LinearNetwork`
    :class:`.LinearSystem`
    :func:`.cont2discrete`

    Notes
    -----
    This routine is called automatically by :class:`.LinearNetwork`.

    Principle 3 is a special case of this routine when called with
    a continuous :func:`Lowpass` synapse and ``dt=None``. However, specifying
    the ``dt`` (or providing digital systems) will improve the accuracy in
    digital simulation.

    For higher-order synapses, this makes a zero-order hold (ZOH) assumption
    to avoid requiring the input derivatives. In this case, the mapping is
    not perfect. If the input derivatives are known, then the accuracy can be
    made perfect again. See references for details.

    References
    ----------
    .. [#] A. R. Voelker and C. Eliasmith, "Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells",
       2017, Submitted. [`URL <https://github.com/arvoelke/delay2017>`__]

    Examples
    --------
    >>> from nengolib.synapses import ss2sim, PadeDelay

    Map the state of a balanced :func:`PadeDelay` onto a lowpass synapse:

    >>> import nengo
    >>> from nengolib.signal import balance
    >>> sys = balance(PadeDelay(.05, order=6))
    >>> synapse = nengo.Lowpass(.1)
    >>> mapped = ss2sim(sys, synapse, synapse.default_dt)
    >>> assert np.allclose(sys.C, mapped.C)
    >>> assert np.allclose(sys.D, mapped.D)

    Simulate the mapped system directly (without neurons):

    >>> process = nengo.processes.WhiteSignal(1, high=10, y0=0)
    >>> with nengo.Network() as model:
    >>>     stim = nengo.Node(output=process)
    >>>     x = nengo.Node(size_in=len(sys))
    >>>     nengo.Connection(stim, x, transform=mapped.B, synapse=synapse)
    >>>     nengo.Connection(x, x, transform=mapped.A, synapse=synapse)
    >>>     p_stim = nengo.Probe(stim)
    >>>     p_actual = nengo.Probe(x)
    >>> with nengo.Simulator(model) as sim:
    >>>     sim.run(.5)

    The desired dynamics are implemented perfectly:

    >>> target = sys.X.filt(sim.data[p_stim])
    >>> assert np.allclose(target, sim.data[p_actual])

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sim.trange(), target, linestyle='--', lw=4)
    >>> plt.plot(sim.trange(), sim.data[p_actual], alpha=.5)
    >>> plt.legend()
    >>> plt.show()
    """

    synapse = LinearSystem(synapse)
    if synapse.analog and synapse.order_num > 0:
        raise ValueError("analog synapses (%s) must have order zero in the "
                         "numerator" % synapse)

    sys = LinearSystem(sys)
    if sys.analog != synapse.analog:
        raise ValueError("system (%s) and synapse (%s) must both be analog "
                         "or both be digital" % (sys, synapse))

    if dt is not None:
        if not sys.analog:  # sys is digital
            raise ValueError("system (%s) must be analog if dt is not None" %
                             sys)
        sys = cont2discrete(sys, dt=dt)
        synapse = cont2discrete(synapse, dt=dt)

    # If the synapse was discretized, then its numerator may now have multiple
    #   coefficients. By summing them together, we are implicitly assuming that
    #   the output of the synapse will stay constant across
    #   synapse.order_num + 1 time-steps. This is also related to:
    #   http://dsp.stackexchange.com/questions/33510/difference-between-convolving-before-after-discretizing-lti-systems  # noqa: E501
    # For example, if we have H = Lowpass(0.1), then the only difference
    #   between sys1 = cont2discrete(H*H, dt) and
    #           sys2 = cont2discrete(H, dt)*cont2discrete(H, dt), is that
    #   np.sum(sys1.num) == sys2.num (while sys1.den == sys2.den)!
    gain = np.sum(synapse.num)
    c = synapse.den / gain

    A, B, C, D = sys.ss
    k = len(synapse)
    powA = [matrix_power(A, i) for i in range(k + 1)]
    AH = np.sum([c[i] * powA[i] for i in range(k + 1)], axis=0)

    if sys.analog:
        BH = np.dot(
            np.sum([c[i] * powA[i - 1] for i in range(1, k+1)], axis=0), B)

    else:
        BH = np.dot(
            np.sum([c[i] * powA[i - j - 1]
                    for j in range(k) for i in range(j+1, k+1)], axis=0), B)

    return LinearSystem((AH, BH, C, D), analog=sys.analog)
