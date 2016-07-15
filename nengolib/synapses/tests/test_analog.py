import numpy as np
import pytest

from nengo import LinearFilter as BaseLinearFilter
from nengo import Lowpass as BaseLowpass
from nengo import Alpha as BaseAlpha
from nengo.exceptions import ValidationError
from nengo.utils.testing import warns

from nengolib.synapses.analog import (
    Bandpass, Highpass, PureDelay, LinearFilter, Lowpass, Alpha, DoubleExp,
    _pade_delay, _passthrough_delay, _proper_delay)
from nengolib.signal import impulse, sys_equal, s


def test_nengo_analogs():
    assert sys_equal(BaseLinearFilter([1], [1, 0]), LinearFilter([1], [1, 0]))
    assert sys_equal(BaseLowpass(0.1), Lowpass(0.1))
    assert sys_equal(BaseAlpha(0.1), Alpha(0.1))
    assert sys_equal(BaseAlpha(0.1), DoubleExp(0.1, 0.1))


def test_double_exp():
    tau1 = 0.005
    tau2 = 0.008
    sys = DoubleExp(tau1, tau2)

    assert sys == Lowpass(tau1) * Lowpass(tau2)
    assert sys == 1 / ((tau1*s + 1) * (tau2*s + 1))
    # this equality follows from algebraic manipulation of the above equality
    # however there will be a ZeroDivisionError when tau1 == tau2
    assert sys == (tau1*Lowpass(tau1) - tau2*Lowpass(tau2)) / (tau1 - tau2)


@pytest.mark.parametrize("freq,Q", [(5, 2), (50, 50), (200, 4)])
def test_bandpass(freq, Q):
    sys = Bandpass(freq, Q)

    length = 10000
    dt = 0.0001

    response = impulse(sys, dt, length)
    dft = np.fft.rfft(response, axis=0)
    freqs = np.fft.rfftfreq(length, d=dt)
    cp = abs(dft).cumsum()

    # Check that the cumulative power reaches its mean at Q frequency
    np.allclose(freqs[np.where(cp >= cp[-1] / 2)[0][0]], Q)


@pytest.mark.parametrize("tau,order", [(0.01, 1), (0.2, 2), (0.0001, 5)])
def test_highpass(tau, order):
    sys = Highpass(tau, order)

    length = 1000
    dt = 0.001

    response = impulse(sys, dt, length)
    dft = np.fft.rfft(response, axis=0)
    p = abs(dft)

    # Check that the power is monotonically increasing
    assert np.allclose(np.sort(p), p)


@pytest.mark.parametrize("order", [0, 1.5])
def test_invalid_highpass(order):
    with pytest.raises(ValueError):
        Highpass(0.01, order)


@pytest.mark.parametrize("c", [0.1, 0.4, 0.8])
def test_pade_delay(c):
    dt = 0.001
    length = 1000

    sys = PureDelay(c, order=4)
    response = impulse(sys, dt, length)

    offset = 10
    assert np.allclose(
        (np.argmax(response[offset:])+offset), c*length, atol=100)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_pade_versions(p):
    c = 0.5
    # make sure all of the delay methods do the same thing
    assert _pade_delay(p, p+1, c) == _proper_delay(p+1, c)
    assert _pade_delay(p, p, c) == _passthrough_delay(p, c)

    # consistency check on each code path within main interface
    assert _proper_delay(p+1, c) == PureDelay(c, order=p+1)
    assert _passthrough_delay(p, c) == PureDelay(c, order=p, p=p)
    assert _pade_delay(p, p+2, c) == PureDelay(c, order=p+2, p=p)


def test_delay_invalid():
    with pytest.raises(ValidationError):
        PureDelay(1, order=0)

    with pytest.raises(ValidationError):
        PureDelay(1, order=1)

    with pytest.raises(ValidationError):
        PureDelay(1, order=2.5)

    with pytest.raises(ValidationError):
        PureDelay(1, order=2, p=0)

    with pytest.raises(ValidationError):
        PureDelay(1, order=2, p=1.5)

    with warns(UserWarning):
        PureDelay(1, order=10, p=8)


def test_equivalent_defs():
    tau = 0.05

    assert Lowpass(tau) == 1 / (tau*s + 1)
    assert Alpha(tau) == (1 / (tau*s + 1))**2
    assert Highpass(tau, 3) == (tau * s * Lowpass(tau)) ** 3
