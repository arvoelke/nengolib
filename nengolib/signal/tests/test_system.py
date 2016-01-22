import numpy as np
import pytest
from scipy.signal import ss2zpk, tf2zpk, cont2discrete

from nengo.synapses import filt

from nengolib import Lowpass, Alpha, LinearFilter
from nengolib.signal import state_norm
from nengolib.signal.system import (
    sys2ss, sys2tf, check_sys_equal, tfmul, impulse, is_exp_stable,
    scale_state)


def test_sys_conversions():
    sys = Alpha(0.1)

    tf = sys2tf(sys)
    ss = sys2ss(sys)

    assert check_sys_equal(sys2ss(tf), ss)
    assert check_sys_equal(sys2ss(ss), ss)  # unchanged
    assert check_sys_equal(sys2tf(tf), tf)  # unchanged
    assert check_sys_equal(sys2tf(ss), tf)

    zpk = ss2zpk(*ss)
    assert check_sys_equal(tf2zpk(*tf), zpk)  # sanity check
    assert check_sys_equal(sys2tf(zpk), tf)
    assert check_sys_equal(sys2ss(zpk), ss)

    with pytest.raises(ValueError):
        sys2ss(np.zeros(5))

    with pytest.raises(ValueError):
        sys2tf(np.zeros(5))


def test_check_sys_equal():
    assert not check_sys_equal(np.zeros(2), np.zeros(3))


def test_tfmul():
    # Check that alpha is just two lowpass multiplied together
    assert check_sys_equal(
        tfmul(Lowpass(0.1), Lowpass(0.1)), sys2tf(Alpha(0.1)))


def test_impulse():
    dt = 0.001
    tau = 0.005
    length = 10

    delta = np.zeros(length)  # TODO: turn into a little helper?
    delta[1] = 1.0

    sys = Lowpass(tau)
    response = impulse(sys, dt, length)
    np.allclose(response, filt(delta, sys, dt))

    dss = cont2discrete(sys2ss(sys), dt=dt)[:-1]
    np.allclose(response, impulse(dss, None, length))


def test_is_exp_stable():
    sys = Lowpass(0.1)
    assert is_exp_stable(sys)

    sys = LinearFilter([1], [1, 0])  # integrator
    assert not is_exp_stable(sys)


def test_scale_state():
    syn = Lowpass(0.1)
    ss = sys2ss(syn)
    scaled = scale_state(*ss, radii=2)

    # Check that it's still the same stystem, even though different matrices
    assert check_sys_equal(ss, scaled)
    assert not np.allclose(ss[1], scaled[1])

    # Check that the state vectors have half the power
    assert np.allclose(
        state_norm(ss, analog=True)/2, state_norm(scaled, analog=True))


def test_invalid_scale_state():
    syn = Lowpass(0.1)
    ss = sys2ss(syn)

    scale_state(*ss, radii=[1])

    with pytest.raises(ValueError):
        scale_state(*ss, radii=[[1]])

    with pytest.raises(ValueError):
        scale_state(*ss, radii=[1, 2])
