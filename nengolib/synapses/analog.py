# -*- coding: utf-8 -*-
import numpy as np
import warnings
from scipy.misc import pade, factorial

from nengo.utils.compat import is_integer

from nengolib.signal.system import LinearSystem, s

__all__ = [
    'Lowpass', 'Alpha', 'DoubleExp', 'Bandpass', 'Highpass',
    'pade_delay_error', 'PadeDelay', 'LegendreDelay']


def Lowpass(tau):
    """A first-order lowpass: ``1/(tau*s + 1)``.

    Parameters
    ----------
    tau : ``float``
        Time-constant of exponential decay.

    Returns
    -------
    sys : :class:`.LinearSystem`
        First-order lowpass.

    See Also
    --------
    :class:`nengo.Lowpass`
    :attr:`.s`

    Examples
    --------
    >>> from nengolib import Lowpass
    >>> import matplotlib.pyplot as plt
    >>> taus = np.linspace(.01, .05, 5)
    >>> for tau in taus:
    >>>     sys = Lowpass(tau)
    >>>     plt.plot(sys.ntrange(100), sys.impulse(100),
    >>>              label=r"$\\tau=%s$" % tau)
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    return 1 / (tau*s + 1)


def Alpha(tau):
    """A second-order lowpass: ``1/(tau*s + 1)**2``.

    Equivalent to convolving two identical lowpass synapses together.

    Parameters
    ----------
    tau : ``float``
        Time-constant of exponential decay.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Second-order lowpass with identical time-constants.

    See Also
    --------
    :class:`nengo.Alpha`
    :func:`.Lowpass`
    :func:`.DoubleExp`

    Examples
    --------
    >>> from nengolib import Alpha
    >>> import matplotlib.pyplot as plt
    >>> taus = np.linspace(.01, .05, 5)
    >>> for tau in taus:
    >>>     sys = Alpha(tau)
    >>>     plt.plot(sys.ntrange(100), sys.impulse(100),
    >>>              label=r"$\\tau=%s$" % tau)
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    return DoubleExp(tau, tau)


def DoubleExp(tau1, tau2):
    """A second-order lowpass: ``1/((tau1*s + 1)*(tau2*s + 1))``.

    Equivalent to convolving two lowpass synapses together with potentially
    different time-constants, in either order.

    Parameters
    ----------
    tau1 : ``float``
        Time-constant of one exponential decay.
    tau2 : ``float``
        Time-constant of another exponential decay.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Second-order lowpass with potentially different time-constants.

    See Also
    --------
    :func:`.Lowpass`
    :func:`.Alpha`

    Examples
    --------
    >>> from nengolib import DoubleExp
    >>> import matplotlib.pyplot as plt
    >>> tau1 = .03
    >>> taus = np.linspace(.01, .05, 5)
    >>> plt.title(r"$\\tau_1=%s$" % tau1)
    >>> for tau2 in taus:
    >>>     sys = DoubleExp(tau1, tau2)
    >>>     plt.plot(sys.ntrange(100), sys.impulse(100),
    >>>              label=r"$\\tau_2=%s$" % tau2)
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    return Lowpass(tau1) * Lowpass(tau2)


def Bandpass(freq, Q):
    """A second-order bandpass with given frequency and width.

    Parameters
    ----------
    freq : ``float``
        Frequency (in hertz) of the peak of the bandpass.
    Q : ``float``
        Inversely proportional to width of bandpass.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Second-order lowpass with complex poles.

    See Also
    --------
    :func:`nengo.networks.Oscillator`

    Notes
    -----
    The state of this system is isomorphic to a decaying 2--dimensional
    oscillator with speed given by ``freq`` and decay given by ``Q``.

    References
    ----------
    .. [#] http://www.analog.com/library/analogDialogue/archives/43-09/EDCh%208%20filter.pdf

    Examples
    --------
    Bandpass filters centered around 20 Hz with varying bandwidths:

    >>> from nengolib.synapses import Bandpass
    >>> freq = 20
    >>> Qs = np.linspace(4, 40, 5)

    Evaluate each impulse (time-domain) response:

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(121)
    >>> for Q in Qs:
    >>>     sys = Bandpass(freq, Q)
    >>>     plt.plot(sys.ntrange(1000), sys.impulse(1000),
    >>>              label=r"$Q=%s$" % Q)
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()

    Evaluate each frequency responses:

    >>> plt.subplot(122)
    >>> freqs = np.linspace(0, 40, 100)  # to evaluate
    >>> for Q in Qs:
    >>>     sys = Bandpass(freq, Q)
    >>>     plt.plot(freqs, np.abs(sys.evaluate(freqs)),
    >>>              label=r"$Q=%s$" % Q)
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.legend()
    >>> plt.show()

    Evaluate each state-space impulse (trajectory) after balancing:

    >>> from nengolib.signal import balance
    >>> for Q in Qs:
    >>>     plt.plot(*balance(Bandpass(freq, Q)).X.impulse(1000).T,
    >>>              label=r"$Q=%s$" % Q)
    >>> plt.legend()
    >>> plt.axis('equal')
    >>> plt.show()
    """  # noqa: E501

    w_0 = freq * (2*np.pi)
    return 1 / ((s/w_0)**2 + s/(w_0*Q) + 1)


def Highpass(tau, order=1):
    """A differentiated lowpass of given order: ``(tau*s/(tau*s + 1))**order``.

    Equivalent to differentiating the input, scaling by ``tau``,
    lowpass filtering with time-constant ``tau``, and finally repeating
    this ``order`` times. The lowpass filter is required to make this causal.

    Parameters
    ----------
    tau : ``float``
        Time-constant of the lowpass filter, and highpass gain.
    order : ``integer``, optional
        Dimension of the resulting linear system. Defaults to ``1``.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Highpass filter with time-constant ``tau`` and dimension ``order``.

    See Also
    --------
    :func:`.Lowpass`
    :attr:`.s`

    Examples
    --------
    >>> from nengolib.synapses import Highpass

    Evaluate the highpass in the frequency domain with a time-constant of 10 ms
    and with a variety of orders:

    >>> tau = 1e-2
    >>> orders = list(range(1, 9))
    >>> freqs = np.linspace(0, 50, 100)  # to evaluate

    >>> import matplotlib.pyplot as plt
    >>> plt.title(r"$\\tau=%s$" % tau)
    >>> for order in orders:
    >>>     sys = Highpass(tau, order)
    >>>     assert len(sys) == order
    >>>     plt.plot(freqs, np.abs(sys.evaluate(freqs)),
    >>>              label=r"order=%s" % order)
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.legend()
    >>> plt.show()
    """

    if order < 1 or not is_integer(order):
        raise ValueError("order (%s) must be integer >= 1" % order)
    return (tau*s * Lowpass(tau))**order


def _passthrough_delay(m, c):
    """Analytically derived state-space when p = q = m.

    We use this because it is numerically stable for high m.
    """
    j = np.arange(1, m+1, dtype=np.float64)
    u = (m + j) * (m - j + 1) / (c * j)

    A = np.zeros((m, m))
    B = np.zeros((m, 1))
    C = np.zeros((1, m))
    D = np.zeros((1,))

    A[0, :] = B[0, 0] = -u[0]
    A[1:, :-1][np.diag_indices(m-1)] = u[1:]
    D[0] = (-1)**m
    C[0, np.arange(m) % 2 == 0] = 2*D[0]
    return LinearSystem((A, B, C, D), analog=True)


def _proper_delay(q, c):
    """Analytically derived state-space when p = q - 1.

    We use this because it is numerically stable for high q
    and doesn't have a passthrough.
    """
    j = np.arange(q, dtype=np.float64)
    u = (q + j) * (q - j) / (c * (j + 1))

    A = np.zeros((q, q))
    B = np.zeros((q, 1))
    C = np.zeros((1, q))
    D = np.zeros((1,))

    A[0, :] = -u[0]
    B[0, 0] = u[0]
    A[1:, :-1][np.diag_indices(q-1)] = u[1:]
    C[0, :] = (j + 1) / float(q) * (-1) ** (q - 1 - j)
    return LinearSystem((A, B, C, D), analog=True)


def _pade_delay(p, q, c):
    """Numerically evaluated state-space using Pade approximants.

    This may have numerical issues for large values of p or q.
    """
    i = np.arange(1, p+q+1, dtype=np.float64)
    taylor = np.append([1.0], (-c)**i / factorial(i))
    num, den = pade(taylor, q)
    return LinearSystem((num, den), analog=True)


def pade_delay_error(theta_times_freq, order, p=None):
    """Computes the approximation error in :func:`.PadeDelay`.

    For a given order, the difficulty of the delay is a function of the
    input frequency (:math:`s = 2j \\pi f`) times the delay length
    (:math:`\\theta`).

    Parameters
    ----------
    theta_times_freq : ``array_like``
        A float or array of floats (delay length times frequency) at which to
        evaluate the error.
    order : ``integer``
        ``order`` parameter passed to :func:`.PadeDelay`.
    p : ``integer``, optional
        ``p`` parameter passed to :func:`.PadeDelay`. Defaults to ``None``.

    Returns
    -------
    error : ``np.array`` of ``np.complex``
        Shaped like ``theta_times_freq``, with each element corresponding
        to the complex error term

        .. math::

            F(2j \\pi f) - e^{-\\theta \\times 2j \\pi f}

        where :math:`F(s)` is the transfer function constructed by
        :func:`.PadeDelay` for a delay of length :math:`\\theta`.

    See Also
    --------
    :func:`.PadeDelay`

    Examples
    --------
    >>> from nengolib.synapses import pade_delay_error
    >>> abs(pade_delay_error(1, order=6))
    0.0070350205992081461

    This means that for ``order=6`` and frequencies less than ``1/theta``,
    the approximation error is less than one percent!

    Now visualize the error across a range of frequencies, with various orders:

    >>> import matplotlib.pyplot as plt
    >>> freq_times_theta = np.linspace(0, 5, 1000)
    >>> for order in range(4, 9):
    >>>     plt.plot(freq_times_theta,
    >>>              abs(pade_delay_error(freq_times_theta, order=order)),
    >>>              label="order=%s" % order)
    >>> plt.xlabel(r"Frequency $\\times \\, \\theta$ (Unitless)")
    >>> plt.ylabel("Absolute Error")
    >>> plt.legend()
    >>> plt.show()
    """

    ttf = np.asarray(theta_times_freq)
    # switch to a delay of 1 for simplicity
    # this works due to the substitution of variables: theta*s <-> 1*s'
    sys = PadeDelay(1., order, p=p)
    return sys.evaluate(ttf) - np.exp(-2j*np.pi*ttf)


def _check_order(order):
    if order < 1 or not is_integer(order):
        raise ValueError("order (%s) must be integer >= 1" % order)
    return order


def PadeDelay(theta, order, p=None):
    """A finite-order approximation of a pure time-delay.

    Implements the transfer function:

    .. math::

       F(s) = e^{-\\theta s} + \\mathcal{O}(s^{\\texttt{order}+\\texttt{p}})

    or :math:`y(t) \\approx u(t - \\theta)` in the time-domain (for
    slowly changing inputs).

    This is the optimal approximation for a time-delay system given
    low-frequency inputs implemented using a finite-dimensional state.
    This is achieved via Padé approximants. The state-space of the
    system encodes a rolling window of input history
    (see :class:`.RollingWindow`). This can be used to approximate
    FIR filters and window functions in continuous time.

    Parameters
    ----------
    theta : ``float``
        Length of time-delay in seconds.
    order : ``integer``
        Order of approximation in the denominator
        (dimensionality of resulting system).
    p : ``integer``, optional
        Order of approximation in the numerator. Defaults to ``p=order-1``,
        since this gives the highest-order approximation without a passthrough.
        If ``p=order``, then the system will have a passthrough, which has
        a nonideal step response.

    Returns
    -------
    sys : :class:`.LinearSystem`
        Finite-order approximation of a pure time-delay.

    See Also
    --------
    :func:`.LegendreDelay`
    :func:`.pade_delay_error`
    :class:`.RollingWindow`
    :func:`.DiscreteDelay`
    :func:`scipy.misc.pade`

    Notes
    -----
    Closed-form derivations are found in [#]_.

    References
    ----------
    .. [#] A. R. Voelker and C. Eliasmith, "Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells",
       2017, Submitted. [`URL <https://github.com/arvoelke/delay2017>`__]

    Examples
    --------
    >>> from nengolib.synapses import PadeDelay

    Delay 15 Hz band-limited white noise by 100 ms using various orders of
    approximations:

    >>> from nengolib.signal import z
    >>> from nengo.processes import WhiteSignal
    >>> import matplotlib.pyplot as plt
    >>> process = WhiteSignal(10., high=15, y0=0)
    >>> u = process.run_steps(500)
    >>> t = process.ntrange(len(u))
    >>> plt.plot(t, (z**-100).filt(u), linestyle='--', lw=4, label="Ideal")
    >>> for order in list(range(4, 9)):
    >>>     sys = PadeDelay(.1, order=order)
    >>>     assert len(sys) == order
    >>>     plt.plot(t, sys.filt(u), label="order=%s" % order)
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    q = _check_order(order)
    if p is None:
        p = q - 1

    if p < 1 or not is_integer(p):
        raise ValueError("p (%s) must be integer >= 1" % p)

    if p == q:
        return _passthrough_delay(p, theta)
    elif p == q - 1:
        return _proper_delay(q, theta)
    else:
        if q >= 10:
            warnings.warn("For large values of q (>= 10), p should either be "
                          "None, q - 1, or q.")
        return _pade_delay(p, q, theta)


def LegendreDelay(theta, order):
    """PadeDelay(theta, order) realizing the shifted Legendre basis.

    The transfer function is equivalent to :func:`.PadeDelay`, but its
    canonical state-space realization represents the window of history
    by the shifted Legendre polnomials:

    .. math::

       P_i(2 \\theta' \\theta^{-1} - 1)

    where ``i`` is the zero-based index into the state-vector.

    Parameters
    ----------
    theta : ``float``
        Length of time-delay in seconds.
    order : ``integer``
        Order of approximation in the denominator
        (dimensionality of resulting system).

    Returns
    -------
    sys : :class:`.LinearSystem`
        Finite-order approximation of a pure time-delay.

    See Also
    --------
    :func:`.PadeDelay`
    :func:`.pade_delay_error`
    :class:`.RollingWindow`

    Examples
    --------
    >>> from nengolib.synapses import LegendreDelay

    Delay 15 Hz band-limited white noise by 100 ms using various orders of
    approximations:

    >>> from nengolib.signal import z
    >>> from nengo.processes import WhiteSignal
    >>> import matplotlib.pyplot as plt
    >>> process = WhiteSignal(10., high=15, y0=0)
    >>> u = process.run_steps(500)
    >>> t = process.ntrange(len(u))
    >>> plt.plot(t, (z**-100).filt(u), linestyle='--', lw=4, label="Ideal")
    >>> for order in list(range(4, 9)):
    >>>     sys = LegendreDelay(.1, order=order)
    >>>     assert len(sys) == order
    >>>     plt.plot(t, sys.filt(u), label="order=%s" % order)
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    q = _check_order(order)

    Q = np.arange(q, dtype=np.float64)
    R = (2*Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
    B = (-1.)**Q[:, None] * R
    C = np.ones((1, q))
    D = np.zeros((1,))

    return LinearSystem((A, B, C, D), analog=True)
