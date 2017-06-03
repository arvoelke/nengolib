import numpy as np

from nengo.utils.numpy import rmse, rms

__all__ = ['nrmse', 'shift']


def nrmse(actual, target, **kwargs):
    """Returns RMSE(actual, target) / RMS(target).

    This gives the normalized RMSE, such that an output of ``0``
    corresponds to ``actual=target`` while an output of ``1``
    corresponds to ``actual=np.zeros_like(target)``.
    This is scale-invariant in the power of the target signal.

    Parameters
    ----------
    actual : ``array_like``
        Actual signal (measured).
    target : ``array_like``
        Target signal (expected).
    **kwargs : ``dictionary``, optional
        Additional keyword arguments passed to
        :func:`nengo.utils.numpy.rmse` and :func:`nengo.utils.numpy.rms`.

    Notes
    -----
    This function is not symmetric. That is, ``nrmse(x, y)`` may differ from
    ``nrmse(y, x)``.

    See Also
    --------
    :func:`nengo.utils.numpy.rmse`
    :func:`nengo.utils.numpy.rms`
    """

    actual = np.asarray(actual)
    target = np.asarray(target)
    return rmse(actual, target, **kwargs) / rms(target, **kwargs)


def shift(signal, amount=1, value=0):
    """Shift the ``signal`` ``amount`` steps, prepending with ``value``.

    Parameters
    ----------
    signal : ``array_like``
        Signal to be shifted.
    amount : ``integer``, optional
        Number of steps to shift. Defaults to ``1``.
    value : ``array_like``, optional
        Value to pad along the first axis. Must be resizable according to the
        remaining axes. Defaults to ``0``.

    Returns
    -------
    ``np.array``
        Shifted signal, shaped according to ``signal``.

    Examples
    --------
    >>> from nengolib.signal import shift
    >>> a = [1, 2, 3]
    >>> shift(a)
    array([0, 1, 2])
    >>> shift(a, 2, 4)
    array([4, 4, 1])

    >>> a = [[1, 2], [3, 4], [5, 6]]
    >>> shift(a)
    array([[0, 0],
           [1, 2],
           [3, 4]])
    >>> shift(a, 2, [0, 7])
    array([[0, 7],
           [0, 7],
           [1, 2]])
    """

    signal = np.asarray(signal)
    shape = (1,) + signal.shape[1:] if signal.ndim > 1 else ()
    value = np.resize(value, shape)
    padded = np.concatenate(
        (np.repeat(value, amount, axis=0),) + (signal,))
    return padded[:len(signal)]
