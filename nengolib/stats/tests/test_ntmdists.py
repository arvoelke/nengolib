import numpy as np
import pytest

from nengo.utils.numpy import norm
from nengo.utils.testing import warns

from nengolib.stats.ntmdists import SphericalCoords, Sobol, sphere, ball


@pytest.mark.parametrize("m", [1, 2, 4, 16])
def test_spherical_coords(m, rng):
    dist = SphericalCoords(m)
    n = 10001
    x = np.linspace(0, 1, n)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)

    # Check that the cdf matches the pdf
    assert np.allclose(pdf.cumsum()/n, cdf, atol=1e-3)

    # Check that the ppf inverts the cdf
    # cutoff needed because the ppf rounds up to 1 when the cdf values
    # get too close 1o 1
    assert np.allclose(dist.ppf(cdf)[x < 0.9], x[x < 0.9])

    # Check that the sampling approximates the cdf
    rvs = dist.sample(n, rng=rng)
    width = 0.1
    for b in np.arange(0, 1, 1.0/width):
        assert np.allclose(np.sum((rvs > b) & (rvs <= b+width)) / float(n),
                           dist.cdf(b+width) - dist.cdf(b), atol=1e-2)


def test_sobol():
    # check derministic
    s1 = Sobol().sample(3, 4)
    s2 = Sobol().sample(3, 4)
    assert np.allclose(s1, s2)

    # check shape
    assert s1.shape == (3, 4)


def test_sobol_invalid_dims():
    with pytest.raises(ValueError):
        Sobol().sample(1, d=0)

    with pytest.raises(ValueError):
        Sobol().sample(1, d=1.5)

    with warns(UserWarning):
        Sobol().sample(2, d=41)


def _furthest(x):
    # returns largest distance from each point
    diffs = x[:, None, :] - x[None, :, :]
    dists = norm(diffs, axis=2)
    return np.max(dists, axis=0)


@pytest.mark.parametrize("d", [1, 2, 4, 16, 64])
def test_ball(d, rng):
    n = 200
    x = ball.sample(n, d, rng)
    assert x.shape == (n, d)

    dist = norm(x, axis=1)
    assert (dist <= 1).all()

    f = _furthest(x)
    assert (f > dist + 0.5).all()


@pytest.mark.parametrize("d", [1, 2, 4, 16, 64])
def test_sphere(d, rng):
    n = 200
    x = sphere.sample(n, d, rng)
    assert x.shape == (n, d)
    assert np.allclose(norm(x, axis=1), 1)

    f = _furthest(x)
    assert (f > 1.5).all()


def test_dist_repr():
    assert repr(SphericalCoords(4)) == "SphericalCoords(4)"
    assert repr(Sobol()) == "Sobol()"
    assert repr(sphere) == "ScatteredHypersphere(surface=True, base=Sobol())"
    assert repr(ball) == "ScatteredHypersphere(surface=False, base=Sobol())"
