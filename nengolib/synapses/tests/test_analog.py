import numpy as np
import pytest

from nengo import LinearFilter as BaseLinearFilter
from nengo import Lowpass as BaseLowpass
from nengo import Alpha as BaseAlpha

from nengolib.synapses.analog import (
    Bandpass, Highpass, pade_delay_error, PadeDelay, Lowpass, Alpha, DoubleExp,
    _pade_delay, _passthrough_delay, _proper_delay)
from nengolib.signal import sys_equal, s, LinearSystem
from nengolib.testing import warns


def test_nengo_analogs():
    assert sys_equal(BaseLinearFilter([1], [1, 0]),
                     LinearSystem(([1], [1, 0])))
    assert sys_equal(BaseLowpass(0.1), Lowpass(0.1))
    assert sys_equal(BaseAlpha(0.1), Alpha(0.1))
    assert sys_equal(BaseAlpha(0.1), DoubleExp(0.1, 0.1))


def test_double_exp():
    tau1 = 0.005
    tau2 = 0.008
    sys = DoubleExp(tau1, tau2)

    assert sys_equal(sys, ([1], [tau1*tau2, tau1 + tau2, 1]))

    assert sys == Lowpass(tau1) * Lowpass(tau2)
    assert sys == 1 / ((tau1*s + 1) * (tau2*s + 1))
    # this equality follows from algebraic manipulation of the above equality
    # however there will be a ZeroDivisionError when tau1 == tau2
    assert sys == (tau1*Lowpass(tau1) - tau2*Lowpass(tau2)) / (tau1 - tau2)


@pytest.mark.parametrize("freq,Q", [(5, 2), (50, 50), (200, 4)])
def test_bandpass(freq, Q):
    sys = Bandpass(freq, Q)

    w_0 = freq * (2*np.pi)
    assert sys_equal(sys, ([1], [1./w_0**2, 1./(w_0*Q), 1]))

    length = 10000
    dt = 0.0001

    response = sys.impulse(length, dt)
    dft = np.fft.rfft(response, axis=0)
    freqs = np.fft.rfftfreq(length, d=dt)
    cp = abs(dft).cumsum()

    # Check that the cumulative power reaches its mean at Q frequency
    np.allclose(freqs[np.where(cp >= cp[-1] / 2)[0][0]], Q)


@pytest.mark.parametrize("tau,order", [(0.01, 1), (0.2, 2), (0.0001, 5)])
def test_highpass(tau, order):
    sys = Highpass(tau, order)

    num, den = map(np.poly1d, ([tau, 0], [tau, 1]))
    assert sys_equal(sys, LinearSystem((num**order, den**order)))

    length = 1000
    dt = 0.001

    response = sys.impulse(length, dt)
    dft = np.fft.rfft(response, axis=0)
    p = abs(dft)

    # Check that the power is monotonically increasing
    assert np.allclose(np.sort(p), p)


@pytest.mark.parametrize("order", [0, 1.5])
def test_invalid_highpass(order):
    with pytest.raises(ValueError):
        Highpass(0.01, order)


@pytest.mark.parametrize("c", [0.1, 0.4, 0.8, 1])
def test_pade_delay(c):
    dt = c / 100.0
    length = int(2*c / dt)

    # Note: the discretization has numerical issues
    # for smaller dt and larger orders, given an impulse.
    sys = PadeDelay(c, order=12)
    response = sys.impulse(length, dt)

    offset = int(0.1*c/dt)  # start at 10% of delay
    atol = int(0.1*c/dt)  # allow 10% margin of error
    assert np.allclose(
        (np.argmax(response[offset:])+offset), int(c/dt), atol=atol)


def test_pade_error(plt):
    assert np.allclose(pade_delay_error(0, order=2), 0)
    assert np.allclose(abs(pade_delay_error(1e5, order=12)), 1)

    # Monotonically decreasing in order
    # Note: these constants appear in examples
    assert np.allclose(abs(pade_delay_error(1, order=5)), 0.066858, atol=1e-6)
    assert np.allclose(abs(pade_delay_error(1, order=6)), 0.007035, atol=1e-6)
    assert np.allclose(abs(pade_delay_error(1, order=7)), 0.000476, atol=1e-6)

    # Monotonically increasing in theta_times_freq
    # (until it oscillates around 1 <-- not part of unit test)
    ttf = np.linspace(0, 2, 100)
    for order in (6, 12, 18):
        error = np.abs(pade_delay_error(ttf, order=order))
        plt.plot(error)
        assert np.all(np.diff(error) > -1e-13), order


@pytest.mark.parametrize("p", [1, 2, 3])
def test_pade_versions(p):
    c = 1
    # make sure all of the delay methods do the same thing
    assert _pade_delay(p, p+1, c) == _proper_delay(p+1, c)
    assert _pade_delay(p, p, c) == _passthrough_delay(p, c)

    # consistency check on each code path within main interface
    assert _proper_delay(p+1, c) == PadeDelay(c, order=p+1)
    assert _passthrough_delay(p, c) == PadeDelay(c, order=p, p=p)
    assert _pade_delay(p, p+2, c) == PadeDelay(c, order=p+2, p=p)


def test_delay_invalid():
    with pytest.raises(ValueError):
        PadeDelay(1, order=0)

    with pytest.raises(ValueError):
        PadeDelay(1, order=1)

    with pytest.raises(ValueError):
        PadeDelay(1, order=2.5)

    with pytest.raises(ValueError):
        PadeDelay(1, order=2, p=0)

    with pytest.raises(ValueError):
        PadeDelay(1, order=2, p=1.5)

    with warns(UserWarning):
        PadeDelay(1, order=10, p=8)


def test_equivalent_defs():
    tau = 0.05

    assert Lowpass(tau) == 1 / (tau*s + 1)
    assert Alpha(tau) == (1 / (tau*s + 1))**2
    assert Highpass(tau, 3) == (tau * s * Lowpass(tau)) ** 3
