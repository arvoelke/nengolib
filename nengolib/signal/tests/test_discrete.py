import pytest

from nengolib.signal.discrete import cont2discrete, discrete2cont
from nengolib.signal import s, z, ss_equal
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

        assert ss_equal(sys, rsys, atol=1e-07)


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
