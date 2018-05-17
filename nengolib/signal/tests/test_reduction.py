import numpy as np
import pytest

from nengo.utils.numpy import rmse

from nengolib.signal.reduction import pole_zero_cancel, modred, balred
from nengolib import Lowpass, Alpha
from nengolib.compat import warns
from nengolib.signal import LinearSystem, balanced_transformation, shift


def test_minreal():
    sys1 = Lowpass(0.05)
    sys2 = Lowpass(0.01)
    sys3 = sys2*sys2
    sys4 = LinearSystem(2)

    assert pole_zero_cancel(sys1) == sys1
    assert pole_zero_cancel(sys3) == sys3
    assert pole_zero_cancel(sys3/sys2, tol=1e-4) == sys2  # numpy 1.9.2
    assert (pole_zero_cancel(sys1*sys2/sys3, tol=1e-4) ==
            sys1/sys2)  # numpy 1.9.2

    assert (pole_zero_cancel(sys2/sys3, tol=1e-4) ==
            pole_zero_cancel(~sys2))  # numpy 1.9.2
    assert (pole_zero_cancel(sys3*sys4) ==
            pole_zero_cancel(sys4)*pole_zero_cancel(sys3))


def test_modred(rng):
    dt = 0.001
    isys = Lowpass(0.05)
    noise = 0.5*Lowpass(0.01) + 0.5*Alpha(0.005)
    p = 0.999
    sys = p*isys + (1-p)*noise

    T, Tinv, S = balanced_transformation(sys)
    balsys = sys.transform(T, Tinv)

    # Keeping just the best state should remove the 3 noise dimensions
    # Discarding the lowest state should do at least as well
    for keep_states in (S.argmax(),
                        list(set(range(len(sys))) - set((S.argmin(),)))):
        delsys = modred(balsys, keep_states)
        assert delsys.order_den == np.asarray(keep_states).size

        u = rng.normal(size=2000)

        expected = sys.filt(u, dt)
        actual = delsys.filt(u, dt)
        assert rmse(expected, actual) < 1e-4

        step = np.zeros(2000)
        step[50:] = 1.0
        dcsys = modred(balsys, keep_states, method='dc')
        assert np.allclose(dcsys.dcgain, balsys.dcgain)

        # use of shift related to nengo issue #938
        assert not sys.has_passthrough
        assert dcsys.has_passthrough
        expected = shift(sys.filt(step, dt))
        actual = dcsys.filt(step, dt)
        assert rmse(expected, actual) < 1e-4


def test_invalid_modred():
    with pytest.raises(ValueError):
        modred(Lowpass(0.1), 0, method='zoh')


def test_balred(rng):
    dt = 0.001
    sys = Alpha(0.01) + Lowpass(0.001)

    u = rng.normal(size=2000)
    expected = sys.filt(u, dt)

    def check(order, within, tol, method='del'):
        redsys = balred(sys, order, method=method)
        assert redsys.order_den <= order
        actual = redsys.filt(u, dt)
        assert abs(rmse(expected, actual) - within) < tol

    with warns(UserWarning):
        check(4, 0, 1e-13)
    with warns(UserWarning):
        check(3, 0, 1e-13)
    check(2, 0.03, 0.01)
    check(1, 0.3, 0.1)


def test_invalid_balred():
    with pytest.raises(ValueError):
        balred(Lowpass(0.1), 0)
