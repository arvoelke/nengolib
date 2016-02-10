import numpy as np
import pytest

from nengo.utils.numpy import rmse

from nengolib.signal.reduction import (
    minreal, similarity_transform, balreal, modred, balred)
from nengolib.signal import sys2ss, sys_equal, LinearSystem, apply_filter
from nengolib import Lowpass, Alpha


def test_minreal():
    sys1 = Lowpass(0.05)
    sys2 = Lowpass(0.01)
    sys3 = sys2*sys2
    sys4 = LinearSystem(2)

    assert minreal(sys1) == sys1
    assert minreal(sys3) == sys3
    assert minreal(sys3/sys2, tol=1e-4) == sys2  # numpy 1.9.2
    assert minreal(sys1*sys2/sys3, tol=1e-4) == sys1/sys2  # numpy 1.9.2

    assert minreal(sys2/sys3, tol=1e-4) == minreal(~sys2)  # numpy 1.9.2
    assert minreal(sys3*sys4) == minreal(sys4)*minreal(sys3)


def test_similarity_transform():
    sys = Alpha(0.1)

    A, B, C, D = sys2ss(sys)
    T = np.eye(len(A))
    TA, TB, TC, TD = similarity_transform(A, B, C, D, T)

    assert np.allclose(A, TA)
    assert np.allclose(B, TB)
    assert np.allclose(C, TC)
    assert np.allclose(D, TD)

    T = [[1, 1], [-0.5, 0]]
    TA, TB, TC, TD = similarity_transform(A, B, C, D, T)
    assert sys_equal(sys, (TA, TB, TC, TD))


def test_balreal():
    isys = Lowpass(0.05)
    noise = 0.5*Lowpass(0.01) + 0.5*Alpha(0.005)
    p = 0.8
    sys = p*isys + (1-p)*noise

    balsys, S = balreal(sys)
    assert balsys == sys

    assert np.all(S >= 0)
    assert np.all(S[0] > 0.1)
    assert np.all(S[1:] < 0.01)


def test_modred(rng):
    dt = 0.001
    isys = Lowpass(0.05)
    noise = 0.5*Lowpass(0.01) + 0.5*Alpha(0.005)
    p = 0.999
    sys = p*isys + (1-p)*noise

    balsys, S = balreal(sys)
    delsys = modred(balsys, S.argmax())
    assert delsys.order_den == 1

    u = rng.normal(size=2000)
    expected = apply_filter(u, sys, dt)
    actual = apply_filter(u, delsys, dt)

    assert rmse(expected, actual) < 1e-4

    step = np.zeros(2000)
    step[50:] = 1.0
    dcsys = modred(balsys, S.argmax(), method='dc')
    expected = apply_filter(step, sys, dt)
    actual = apply_filter(step, dcsys, dt)

    assert rmse(expected, actual) < 1e-4


def test_invalid_modred():
    with pytest.raises(ValueError):
        modred(Lowpass(0.1), 0, method='zoh')


def test_balred(rng):
    dt = 0.001
    sys = Alpha(0.01) + Lowpass(0.001)

    u = rng.normal(size=2000)
    expected = apply_filter(u, sys, dt)

    def check(order, within, tol, method='del'):
        redsys = balred(sys, order, method=method)
        actual = apply_filter(u, redsys, dt)
        assert abs(rmse(expected, actual) - within) < tol

    check(4, 0, 1e-13)
    check(3, 0, 1e-13)
    check(2, 0.03, 0.01)
    check(1, 0.3, 0.1)


def test_invalid_balred():
    with pytest.raises(ValueError):
        balred(Lowpass(0.1), 0)
