import numpy as np

from scipy.linalg import svd

from nengo.dists import UniformHypersphere

__all__ = ['random_orthogonal']


def random_orthogonal(d, rng=np.random):
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = svd(m)
    return np.dot(u, v)
