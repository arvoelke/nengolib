import numpy as np
import pytest
from scipy.signal import cont2discrete

from nengo.utils.numpy import norm

from nengolib.signal.lyapunov import _H2P, state_norm
from nengolib.signal import sys2ss, impulse
from nengolib import Alpha


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
    length = 1000000

    # Modify the state-space to read out the state vector
    A, B, C, D = sys2ss(sys)
    old_C = C
    C = np.eye(len(A))
    D = np.zeros((len(A), B.shape[1]))

    discrete = cont2discrete((A, B, C, D), dt)[:-1]

    # Simulate the state vector
    response = impulse(discrete, None, length)

    # Check that the power of each state equals the H2-norm of each state
    # The analog case is the same after scaling since dt is approx 0.
    h2 = state_norm(discrete, analog=False)
    assert np.allclose(h2, norm(response, axis=0))
    assert np.allclose(h2, state_norm(sys, analog=True) / np.sqrt(length))

    plt.figure()
    plt.plot(response[:, 0], label="$x_0$")
    plt.plot(response[:, 1], label="$x_1$")
    plt.plot(np.dot(response, old_C.T), label="$y$")
    plt.legend()


def test_invalid_state_norm():
    with pytest.raises(ValueError):
        state_norm(Alpha(0.1), method=None)
