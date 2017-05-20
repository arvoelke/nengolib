import numpy as np

from nengo.utils.numpy import rmse, rms

__all__ = ['nrmse', 'shift']


def nrmse(actual, target, axis=None, keepdims=False):
    """Returns RMSE(actual, target) over RMS(target)."""
    actual = np.asarray(actual)
    target = np.asarray(target)
    return (rmse(actual, target, axis=axis, keepdims=keepdims) /
            rms(target, axis=axis, keepdims=keepdims))


def shift(signal, amount=1, value=0):
    """Shift the signal ``amount`` steps, filling in with ``value``."""
    signal = np.asarray(signal)
    shape = (1,) + signal.shape[1:] if signal.ndim > 1 else ()
    value = np.resize(value, shape)
    padded = np.concatenate(
        (np.repeat(value, amount, axis=0),) + (signal,))
    return padded[:len(signal)]
