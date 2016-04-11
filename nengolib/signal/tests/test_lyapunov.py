import numpy as np
import pytest

from nengo.utils.numpy import norm

from nengolib.signal.lyapunov import (
    _H2P, state_norm, control_gram, observe_gram, l1_norm)
from nengolib.signal import sys2ss, impulse, cont2discrete, s, z
from nengolib.synapses import Bandpass, PadeDelay
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

    # Modify the state-space to read out the state vector
    A, B, C, D = sys2ss(sys)
    old_C = C
    C = np.eye(len(A))
    D = np.zeros((len(A), B.shape[1]))

    response = np.empty((length, len(C)))
    for i in range(len(C)):
        # Simulate the state vector
        response[:, i] = impulse((A, B, C[i, :], D[i, :]), dt, length)

    # Check that the power of each state equals the H2-norm of each state
    # The analog case is the same after scaling since dt is approx 0.
    actual = norm(response, axis=0) * dt
    assert np.allclose(actual, state_norm(cont2discrete(sys, dt)))
    assert np.allclose(actual, state_norm(sys) * np.sqrt(dt))

    plt.figure()
    plt.plot(response[:, 0], label="$x_0$")
    plt.plot(response[:, 1], label="$x_1$")
    plt.plot(np.dot(response, old_C.T), label="$y$")
    plt.legend()


def test_grams():
    sys = 0.6*Alpha(0.01) + 0.4*Lowpass(0.05)

    A, B, C, D = sys2ss(sys)

    P = control_gram(sys)
    assert np.allclose(np.dot(A, P) + np.dot(P, A.T), -np.dot(B, B.T))
    assert np.linalg.matrix_rank(P) == len(P)  # controllable

    Q = observe_gram(sys)
    assert np.allclose(np.dot(A.T, Q) + np.dot(Q, A), -np.dot(C.T, C))
    assert np.linalg.matrix_rank(Q) == len(Q)  # observable


def test_l1_norm_known():
    # Check that Lowpass has a norm of exactly 1
    l1, rtol = l1_norm(Lowpass(0.1))
    assert np.allclose(l1, 1)
    assert np.allclose(rtol, 0)

    # Check that Alpha scaled by a has a norm of approximately abs(a)
    for a in (-2, 3):
        for desired_rtol in (1e-1, 1e-2, 1e-4, 1e-8):
            l1, rtol = l1_norm(a*Alpha(0.1), rtol=desired_rtol)
            assert np.allclose(l1, abs(a), rtol=rtol)
            assert rtol <= desired_rtol


@pytest.mark.parametrize("sys", [
    Bandpass(10, 3), Bandpass(50, 50), PadeDelay(3, 4, 0.02),
    PadeDelay(4, 4, 0.2)])
def test_l1_norm_unknown(sys):
    # These impulse responses have zero-crossings which makes computing their
    # exact L1-norm infeasible.
    dt = 0.0001
    length = 500000
    response = impulse(sys, dt=dt, length=length)
    assert np.allclose(response[-10:], 0)
    l1_est = np.sum(abs(response) * dt)

    desired_rtol = 1e-6
    l1, rtol = l1_norm(sys, rtol=desired_rtol, max_length=2*length)
    assert np.allclose(l1, l1_est, rtol=1e-2)
    assert rtol <= desired_rtol


def test_l1_norm_bad():
    with pytest.raises(ValueError):
        l1_norm(~z)

    with pytest.raises(ValueError):
        l1_norm(~s)
