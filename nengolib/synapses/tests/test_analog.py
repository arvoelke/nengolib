import numpy as np
import pytest

from nengo import LinearFilter as BaseLinearFilter
from nengo import Lowpass as BaseLowpass
from nengo import Alpha as BaseAlpha

from nengolib.synapses.analog import (
    Bandpass, Highpass, PadeDelay, LinearFilter, Lowpass, Alpha)
from nengolib.signal import impulse, LinearSystem, sys_equal


def test_nengo_analogs():
    assert sys_equal(BaseLinearFilter([1], [1, 0]), LinearFilter([1], [1, 0]))
    assert sys_equal(BaseLowpass(0.1), Lowpass(0.1))
    assert sys_equal(BaseAlpha(0.1), Alpha(0.1))


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
    assert sys == (Lowpass(tau) * LinearSystem(([tau, 0], [1]))) ** order

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
    p = 3
    q = 4
    dt = 0.001
    length = 1000

    sys = PadeDelay(p, q, c)
    response = impulse(sys, dt, length)

    offset = 10
    assert np.allclose(
        (np.argmax(response[offset:])+offset), c*length, atol=100)
