import warnings

import numpy as np
from scipy.special import beta, betainc, betaincinv

from nengo.dists import Distribution, UniformHypersphere
from nengo.utils.compat import is_integer

from nengolib.linalg.ortho import random_orthogonal
from nengolib.stats._sobol_seq import i4_sobol_generate

__all__ = [
    'SphericalCoords', 'Sobol', 'ScatteredCube', 'ScatteredHypersphere',
    'cube', 'sphere', 'ball']


class SphericalCoords(Distribution):
    """Spherical coordinates for inverse transform method.

    This is used to map the hypercube onto the hypersphere and hyperball. [1]_

    References
    ----------
    .. [1] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.
    """

    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, num, d=None, rng=np.random):
        shape = self._sample_shape(num, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        return (np.pi * np.sin(np.pi * x) ** (self.m-1) /
                beta(self.m / 2.0, 0.5))

    def cdf(self, x):
        y = 0.5 * betainc(self.m / 2.0, 0.5, np.sin(np.pi * x) ** 2)
        return np.where(x < 0.5, y, 1 - y)

    def ppf(self, y):
        y_reflect = np.where(y < 0.5, y, 1 - y)
        z_sq = betaincinv(self.m / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)


class Sobol(Distribution):
    """Sobol sequence for quasi Monte Carlo sampling the unit hypercube.

    This is like ``np.random.uniform(0, 1, size=(num, d))`` but with each
    ``d``-dimensional point more uniformly scattered.
    """

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, num, d=1, rng=np.random):
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1./num, 1, num)[:, None]
        if d is None or not is_integer(d) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        if d > 40:
            warnings.warn("i4_sobol_generate does not support d > 40; "
                          "falling back to monte-carlo method", UserWarning)
            return np.random.uniform(size=(num, d))
        return i4_sobol_generate(d, num, skip=0)


class ScatteredCube(Distribution):
    """Number-theoretic distribution over the hypercube.

    Transforms quasi Monte carlo samples from the unit hypercube
    to range between ``low`` and ``high``. These bounds may optionally be
    array-like with shape matching the sample dimensionality.
    """

    def __init__(self, low=-1, high=+1, base=Sobol()):
        super(ScatteredCube, self).__init__()
        self.low = np.atleast_1d(low)
        self.high = np.atleast_1d(high)
        self.w = self.high - self.low
        self.base = base

    def __repr__(self):
        return "%s(low=%r, high=%r, base=%r)" % (
            type(self).__name__, self.low, self.high, self.base)

    def sample(self, num, d=1, rng=np.random):
        u = self.base.sample(num, d)

        # shift everything by the same random constant (with wrap-around)
        u = (u + rng.uniform(size=d)[None, :]) % 1.0

        return u * self.w[None, :] + self.low[None, :]


class ScatteredHypersphere(UniformHypersphere):
    """Number-theoretic distribution over the hypersphere and hyperball.

    Applies the inverse transform method [1]_ to some number-theoretic
    sequence (quasi Monte carlo samples from the unit hypercube).

    References
    ----------
    .. [1] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.
    """

    def __init__(self, surface, base=Sobol()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__, self.surface, self.base)

    def sample(self, num, d=1, rng=np.random):
        if d == 1:
            return super(ScatteredHypersphere, self).sample(num, d, rng)

        if self.surface:
            cube = self.base.sample(num, d-1)
            radius = 1.0
        else:
            dcube = self.base.sample(num, d)
            cube, radius = dcube[:, :-1], dcube[:, -1:] ** (1.0 / d)

        # inverse transform method (section 1.5.2)
        for j in range(d-1):
            cube[:, j] = SphericalCoords(d-1-j).ppf(cube[:, j])

        # spherical coordinate transform
        mapped = np.ones((num, d))
        i = np.ones(d-1)
        i[-1] = 2.0
        s = np.sin(i[None, :] * np.pi * cube)
        c = np.cos(i[None, :] * np.pi * cube)
        mapped[:, 1:] = np.cumprod(s, axis=1)
        mapped[:, :-1] *= c

        # radius adjustment for ball versus sphere, and rotate
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)


cube = ScatteredCube()
sphere = ScatteredHypersphere(surface=True)
ball = ScatteredHypersphere(surface=False)
