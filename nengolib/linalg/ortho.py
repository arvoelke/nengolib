import numpy as np

from nengo.dists import UniformHypersphere

__all__ = ['random_orthogonal']


def random_orthogonal(d, rng=np.random):
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = np.linalg.svd(m)
    return np.dot(u, v)
