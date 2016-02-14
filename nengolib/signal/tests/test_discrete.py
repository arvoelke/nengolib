import pytest

import numpy as np

from nengo.synapses import filt

from nengolib.signal.discrete import cont2discrete, discrete2cont, impulse
from nengolib.signal import s, z
from nengolib.synapses import Lowpass, Alpha, Highpass


@pytest.mark.parametrize(
    "sys", [Lowpass(0.1), Alpha(0.01), Highpass(0.01, order=4)])
def test_discrete(sys):
    dt = 0.001
    alpha = 0.6
    for method in ('gbt', 'bilinear', 'tustin', 'euler', 'forward_diff',
                   'backward_diff', 'zoh'):
        dsys = cont2discrete(sys, dt=dt, method=method, alpha=alpha)
        assert not dsys.analog
        rsys = discrete2cont(dsys, dt=dt, method=method, alpha=alpha)
        assert rsys.analog

        assert np.allclose(sys.ss[0], rsys.ss[0])
        assert np.allclose(sys.ss[1], rsys.ss[1])
        assert np.allclose(sys.ss[2], rsys.ss[2])
        assert np.allclose(sys.ss[3], rsys.ss[3])


def test_invalid_discrete():
    dt = 0.001
    sys = cont2discrete(Lowpass(0.1), dt=dt)

    with pytest.raises(ValueError):
        discrete2cont(sys, dt=dt, method='gbt', alpha=1.1)

    with pytest.raises(ValueError):
        discrete2cont(sys, dt=0)

    with pytest.raises(ValueError):
        discrete2cont(sys, dt=dt, method=None)

    with pytest.raises(ValueError):
        discrete2cont(s, dt=dt)  # already continuous

    with pytest.raises(ValueError):
        cont2discrete(z, dt=dt)  # already discrete


def test_impulse():
    dt = 0.001
    tau = 0.005
    length = 10

    delta = np.zeros(length)  # TODO: turn into a little helper?
    delta[1] = 1.0  # note: start at 1 to compensate for delay removed by nengo

    sys = Lowpass(tau)
    response = impulse(sys, dt, length)
    assert np.allclose(response[0], 0)
    np.allclose(response, filt(delta, sys, dt))

    dss = cont2discrete(sys, dt=dt)
    assert not dss.analog
    np.allclose(response, impulse(dss, dt=None, length=length))


def test_invalid_impulse():
    with pytest.raises(ValueError):
        impulse(s, dt=None, length=10)  # must be digital if dt is None
