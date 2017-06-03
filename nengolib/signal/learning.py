import numpy as np

__all__ = ['pes_learning_rate']


def pes_learning_rate(epsilon, activities, t, dt=0.001):
    """Determine the ideal learning rate for PES without noise or filtering.

    This function returns a ``learning_rate`` for use in the PES rule, such
    that after ``t`` seconds (with a simulation timestep of ``dt``) a constant
    input will have error equal to ``epsilon`` times the initial error. [#]_

    Parameters
    ----------
    epsilon : ``float``
        The desired approximation factor. The resulting error will be
        ``epsilon`` times the initial error after time ``t``. If you want the
        error to be at most some constant, then divide ``epsilon`` by the
        largest possible initial error (usually no more than ``2``, when the
        radius is ``1``).
    activities : ``(n,) array_like``
        An array of ``n`` activity rates. Less activity (small :math:`||a||`)
        need a higher learning rate. Pick the activities with the smallest
        :math:`||a||` that you want to learn within ``epsilon``, or make it the
        average firing rate of each neuron.
    t : ``float``
        The amount of simulation time (in seconds) required to obtain the
        desired error.
    dt : ``float``, optional
        The simulation timestep, defaults to ``0.001`` seconds.

    Returns
    -------
    learning_rate : ``float``
        The learning rate to provide to the PES rule.
    gamma : ``float``
        The rate of convergence, such that the error is the initial error
        multiplied by :math:`\\gamma^k` on the ``k``'th timestep.

    References
    ----------
    .. [#] Aaron R. Voelker. A solution to the dynamics of the prescribed error
       sensitivity learning rule. Technical Report, Centre for Theoretical
       Neuroscience, Waterloo, ON, 10 2015. doi:10.13140/RG.2.1.3048.0084.
       [`URL <http://compneuro.uwaterloo.ca/publications/voelker2015.html>`__]
    """

    activities = np.asarray(activities)
    if activities.ndim != 1:
        raise ValueError("activities must be a one-dimensional array")

    n, = activities.shape  # number of neurons
    a_sq = np.dot(activities.T, activities) * dt  # ||a||^2
    k = (t - dt) / dt  # number of simulation timesteps

    gamma = epsilon**(1.0 / k)  # rate of convergence
    kappa = (1 - gamma) / a_sq  # rearrange equation from theorem

    return kappa * n, gamma
