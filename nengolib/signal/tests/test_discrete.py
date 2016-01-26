import pytest

import numpy as np

from nengolib.signal.discrete import cont2discrete, discrete2cont
from nengolib.synapses import Lowpass, Alpha, Highpass, PadeDelay


@pytest.mark.parametrize(
    "sys", [Lowpass(0.1), Alpha(0.01), Highpass(0.01, order=4),
            PadeDelay(p=3, q=5, c=0.1)])
def test_discrete(sys):
    dt = 0.001
    alpha = 0.6
    for method in ('gbt', 'bilinear', 'tustin', 'euler', 'forward_diff',
                   'backward_diff', 'zoh'):
        dsys = cont2discrete(sys, dt=dt, method=method, alpha=alpha)
        rsys = discrete2cont(dsys, dt=dt, method=method, alpha=alpha)
        assert np.allclose(sys.ss[0], rsys.ss[0], atol=1e-7)
        assert np.allclose(sys.ss[1], rsys.ss[1], atol=1e-7)
        assert np.allclose(sys.ss[2], rsys.ss[2], atol=1e-7)
        assert np.allclose(sys.ss[3], rsys.ss[3], atol=1e-7)

    with pytest.raises(ValueError):
        discrete2cont(dsys, dt=dt, method='gbt', alpha=1.1)

    with pytest.raises(ValueError):
        discrete2cont(dsys, dt=0)

    with pytest.raises(ValueError):
        discrete2cont(dsys, dt=dt, method=None)
