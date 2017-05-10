import numpy as np
import pytest

from nengolib.synapses.digital import DiscreteDelay, BoxFilter
from nengolib.signal import impulse, z, pole_zero_cancel


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_pure_delay(steps):
    length = 20
    sys = DiscreteDelay(steps)
    y = impulse(sys, dt=None, length=length)
    assert np.allclose(y, [0]*steps + [1] + [0]*(length - steps - 1))


@pytest.mark.parametrize("width,normalized", [
    (1, True), (1, False), (2, True), (2, False), (5, True), (5, False)])
def test_box_filter(width, normalized):
    length = 20
    sys = BoxFilter(width, normalized)
    y = impulse(sys, dt=None, length=length)

    amplitude = 1.0 / width if normalized else 1.0
    y_ideal = amplitude * np.asarray([1]*width + [0]*(length - width))
    assert np.allclose(y, y_ideal)


def test_bad_steps():
    with pytest.raises(ValueError):
        DiscreteDelay(-1)

    with pytest.raises(ValueError):
        DiscreteDelay(1.5)

    with pytest.raises(ValueError):
        BoxFilter(0)

    with pytest.raises(ValueError):
        BoxFilter(-1)

    with pytest.raises(ValueError):
        BoxFilter(1.5)


def test_equivalent_defs():
    assert DiscreteDelay(5) == (~z)**5
    assert DiscreteDelay(5) == 1 / z**5

    # equivalent to an (undelayed) integrator convolved with a delayed cutoff
    assert BoxFilter(6, normalized=False) == pole_zero_cancel(
        z / (z - 1) * (1 - 1 / z**6))
