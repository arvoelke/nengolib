import numpy as np

from nengo.utils.numpy import rmse, rms

__all__ = ['nrmse']


def nrmse(actual, target, axis=None, keepdims=False):
    """Returns RMSE(actual, target) over RMS(target)."""
    actual = np.asarray(actual)
    target = np.asarray(target)
    return (rmse(actual, target, axis=axis, keepdims=keepdims) /
            rms(target, axis=axis, keepdims=keepdims))
