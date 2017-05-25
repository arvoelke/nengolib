import numpy as np

from scipy.linalg import svd

from nengo.dists import UniformHypersphere

__all__ = ['random_orthogonal']


def random_orthogonal(d, rng=None):
    """Returns a random orthogonal matrix.

    Parameters
    ----------
    d : ``integer``
        Positive dimension of returned matrix.
    rng : :class:`numpy.random.RandomState`, optional
        Random number generator state.

    Returns
    -------
    ``(d, d) np.array``
        Random orthogonal matrix (an orthonormal basis);
        linearly transforms any vector into a uniformly sampled
        vector on the ``d``--ball with the same L2 norm.

    See Also
    --------
    :class:`.ScatteredHypersphere`

    Examples
    --------
    >>> from nengolib.stats import random_orthogonal, ball
    >>> u = np.abs(ball.sample(200, 2))
    >>> v = u.dot(random_orthogonal(2))

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.axis('equal')
    >>> plt.scatter(*u.T, label="u")
    >>> plt.scatter(*v.T, label="v")
    >>> plt.xlim(-1, 1)
    >>> plt.ylim(-1, 1)
    >>> plt.legend()
    >>> plt.show()
    """

    rng = np.random if rng is None else rng
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = svd(m)
    return np.dot(u, v)
