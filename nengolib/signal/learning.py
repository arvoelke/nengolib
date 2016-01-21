import numpy as np

__all__ = ['pes_learning_rate']


def pes_learning_rate(epsilon, activities, t, dt=0.001):
    """Determine the ideal learning rate for PES without noise or filtering.

    This function returns a `learning_rate` for use in the PES rule, such that
    after `t` seconds (with a simulator timestep of `dt`) a constant input
    will have error equal to `epsilon` times the initial error. [1]_

    Parameters
    ----------
    epsilon : float
        The desired approximation factor. The resulting error will be `epsilon`
        times the initial error. If you want the error to be at most some
        constant, then divide `epsilon` by the largest possible initial error
        (usually no more than 2, when the radius is 1).
    activities : array_like (N,)
        An array of N activity rates. Less activity (small ||a||) need a higher
        learning rate. Pick the activities with the smallest ||a|| that you
        want to learn within epsilon, or make it the average firing rate of
        each neuron.
    t : float
        The amount of simulation time (in seconds) required to obtain the
        desired error.
    dt : float (optional)
        The simulation timestep, defaults to 1 ms.

    Returns
    -------
    learning_rate : float
        The learning rate to provide to the PES rule.
    gamma : float
        The rate of convergence, such that the error is the initial error
        multiplied by `gamma ** k` on the k'th timestep.

    References
    ----------
    .. [1] http://compneuro.uwaterloo.ca/publications/voelker2015.html
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
