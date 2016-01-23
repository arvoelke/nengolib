import numpy as np
import pytest

from nengolib.signal import impulse
from nengolib.synapses.analog import bandpass, highpass


@pytest.mark.parametrize("freq,Q", [(5, 2), (50, 50), (200, 4)])
def test_bandpass(freq, Q):
    sys = bandpass(freq, Q)

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
    sys = highpass(tau, order)

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
        highpass(0.01, order)
