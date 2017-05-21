import pytest

import numpy as np

from nengo.exceptions import ValidationError

from nengolib.processes import Callable


def test_callable_scalar():
    n_steps = 1000
    dt = 0.1
    process = Callable(lambda t: t - dt)
    assert isinstance(repr(process), str)
    assert process.default_size_in == 0
    assert process.default_size_out == 1
    assert process.default_dt == 0.001
    assert process.seed is None

    y = process.run_steps(n_steps, dt=dt)
    t = np.arange(n_steps) * dt
    assert np.allclose(y.squeeze(), t)
    assert np.allclose(t, process.ntrange(n_steps, dt=dt) - dt)


def test_callable_multidim():
    n_steps = 1000
    dt = 0.1
    process = Callable(lambda t: [np.sin(t), np.cos(t)], default_dt=dt)
    assert isinstance(repr(process), str)
    assert process.default_size_in == 0
    assert process.default_size_out == 2
    assert process.default_dt == dt
    assert process.seed is None

    y = process.run_steps(n_steps)
    assert y.shape == (n_steps, 2)

    t = process.ntrange(n_steps, dt=dt)
    assert np.allclose(y[:, 0], np.sin(t))
    assert np.allclose(y[:, 1], np.cos(t))


def test_bad_callable():
    with pytest.raises(ValidationError):
        Callable(1)

    with pytest.raises(ValidationError):
        Callable(lambda a, b: a)

    process = Callable(lambda t: t)

    with pytest.raises(ValidationError):
        process.make_step((1,), (1,), dt=1, rng=np.random)

    with pytest.raises(ValidationError):
        process.make_step((0,), (2,), dt=1, rng=np.random)
