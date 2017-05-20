import pytest

import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import inv

from nengo.utils.numpy import norm

from nengolib.signal.lyapunov import (
    _H2P, state_norm, control_gram, observe_gram, balanced_transformation,
    hankel, l1_norm)
from nengolib.signal import sys2ss, cont2discrete, s, z
from nengolib.synapses import Bandpass, PureDelay
from nengolib import Lowpass, Alpha


def test_lyapunov():
    A = np.asarray([[1, 0.5], [1, 0]])
    B = np.asarray([[1, 0], [0, 1]])

    P = _H2P(A, B, analog=True)
    assert np.allclose(np.dot(A, P) + np.dot(P, A.T) + np.dot(B, B.T), 0)

    P = _H2P(A, B, analog=False)
    assert np.allclose(np.dot(A, np.dot(P, A.T)) - P + np.dot(B, B.T), 0)


def test_state_norm(plt):
    # Choose a filter, timestep, and number of simulation timesteps
    sys = Alpha(0.1)
    dt = 0.000001
    length = 2000000
    assert np.allclose(dt*length, 2.0)

    # Check that the power of each state equals the H2-norm of each state
    # The analog case is the same after scaling since dt is approx 0.
    response = sys.X.impulse(length, dt)
    actual = norm(response, axis=0) * dt
    assert np.allclose(actual, state_norm(cont2discrete(sys, dt)))
    assert np.allclose(actual, state_norm(sys) * np.sqrt(dt))

    step = int(0.002/dt)
    plt.figure()
    plt.plot(response[::step, 0], label="$x_0$")
    plt.plot(response[::step, 1], label="$x_1$")
    plt.plot(np.dot(response[::step], sys.C.T), label="$y$")
    plt.legend()


def test_grams():
    sys = 0.6*Alpha(0.01) + 0.4*Lowpass(0.05)

    A, B, C, D = sys2ss(sys)

    P = control_gram(sys)
    assert np.allclose(np.dot(A, P) + np.dot(P, A.T), -np.dot(B, B.T))
    assert matrix_rank(P) == len(P)  # controllable

    Q = observe_gram(sys)
    assert np.allclose(np.dot(A.T, Q) + np.dot(Q, A), -np.dot(C.T, C))
    assert matrix_rank(Q) == len(Q)  # observable


def test_balreal():
    isys = Lowpass(0.05)
    noise = 0.5*Lowpass(0.01) + 0.5*Alpha(0.005)
    p = 0.8
    sys = p*isys + (1-p)*noise

    T, Tinv, S = balanced_transformation(sys)
    assert np.allclose(inv(T), Tinv)
    assert np.allclose(S, hankel(sys))

    balsys = sys.transform(T, Tinv)
    assert balsys == sys

    assert np.all(S >= 0)
    assert np.all(S[0] > 0.3)
    assert np.all(S[1:] < 0.05)
    assert np.allclose(sorted(S, reverse=True), S)

    P = control_gram(balsys)
    Q = observe_gram(balsys)

    diag = np.diag_indices(len(P))
    offdiag = np.ones_like(P, dtype=bool)
    offdiag[diag] = False
    offdiag = np.where(offdiag)

    assert np.allclose(P[diag], S)
    assert np.allclose(P[offdiag], 0)
    assert np.allclose(Q[diag], S)
    assert np.allclose(Q[offdiag], 0)


@pytest.mark.parametrize("sys", [
    PureDelay(0.1, 4), PureDelay(0.2, 5, 5), Alpha(0.2)])
def test_hankel(sys):
    assert np.allclose(hankel(sys), balanced_transformation(sys)[2])


def test_l1_norm_known():
    # Check that Lowpass has a norm of exactly 1
    l1, rtol = l1_norm(Lowpass(0.1))
    assert np.allclose(l1, 1)
    assert np.allclose(rtol, 0)

    # Check that passthrough is handled properly
    assert np.allclose(l1_norm(Lowpass(0.1) + 5)[0], 6)
    assert np.allclose(l1_norm(Lowpass(0.1) - 5)[0], 6)

    # Check that Alpha scaled by a has a norm of approximately abs(a)
    for a in (-2, 3):
        for desired_rtol in (1e-1, 1e-2, 1e-4, 1e-8):
            l1, rtol = l1_norm(a*Alpha(0.1), rtol=desired_rtol)
            assert np.allclose(l1, abs(a), rtol=rtol)
            assert rtol <= desired_rtol


@pytest.mark.parametrize("sys", [
    Bandpass(10, 3), Bandpass(50, 50), PureDelay(0.02, 4),
    PureDelay(0.2, 4, 4)])
def test_l1_norm_unknown(sys):
    # These impulse responses have zero-crossings which makes computing their
    # exact L1-norm infeasible without simulation.
    dt = 0.0001
    length = 500000
    response = sys.impulse(length, dt)
    assert np.allclose(response[-10:], 0)
    l1_est = np.sum(abs(response) * dt)

    desired_rtol = 1e-6
    l1, rtol = l1_norm(sys, rtol=desired_rtol, max_length=2*length)
    assert np.allclose(l1, l1_est, rtol=1e-3)
    assert rtol <= desired_rtol


def test_l1_norm_bad():
    with pytest.raises(ValueError):
        l1_norm(~z)

    with pytest.raises(ValueError):
        l1_norm(~s)
