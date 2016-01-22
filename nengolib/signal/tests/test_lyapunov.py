import numpy as np
import scipy

from nengo import Alpha
from nengo.utils.numpy import norm

from nengolib.signal import stateH2, sys2ss, impulse


def test_stateH2(plt):
    # Choose a filter, timestep, and number of simulation timesteps
    sys = Alpha(0.1)
    dt = 0.000001
    length = 1000000

    # Modify the state-space to read out the state vector
    A, B, C, D = sys2ss(sys)
    old_C = C
    C = np.eye(len(A))
    D = np.zeros((len(A), B.shape[1]))

    discrete = scipy.signal.cont2discrete((A, B, C, D), dt)[:-1]

    # Simulate the state vector
    response = impulse(discrete, None, length)

    # Check that the power of each state equals the H2-norm of each state
    # The analog case is the same after scaling since dt is approx 0.
    h2 = stateH2(discrete, analog=False)
    assert np.allclose(h2, norm(response, axis=0))
    assert np.allclose(h2, stateH2(sys, analog=True) / np.sqrt(length))

    plt.figure()
    plt.plot(response[:, 0], label="$x_0$")
    plt.plot(response[:, 1], label="$x_1$")
    plt.plot(np.dot(response, old_C.T), label="$y$")
    plt.legend()
