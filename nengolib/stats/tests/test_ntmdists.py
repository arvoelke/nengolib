import numpy as np
import pytest

from nengo.dists import Uniform, UniformHypersphere
from nengo.utils.numpy import norm

from nengolib.stats.ntmdists import (
    SphericalCoords, Sobol, Rd,
    ScatteredCube, ScatteredHypersphere, cube, sphere, ball)
from nengolib.testing import warns


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


@pytest.mark.parametrize("ntm", [Sobol(), Rd()])
def test_ntm_properties(ntm):
    # check derministic
    s1 = ntm.sample(3, 4)
    s2 = ntm.sample(3, 4)
    assert np.allclose(s1, s2)

    # check shape
    assert s1.shape == (3, 4)


@pytest.mark.parametrize("ntm", [Sobol(), Rd()])
def test_ntm_invalid_dims(ntm):
    with pytest.raises(ValueError):
        ntm.sample(1, d=0)

    with pytest.raises(ValueError):
        ntm.sample(1, d=1.5)

    if isinstance(ntm, Sobol):
        with warns(UserWarning):
            ntm.sample(2, d=41)


def _furthest(x):
    # returns largest distance from each point
    diffs = x[:, None, :] - x[None, :, :]
    dists = norm(diffs, axis=2)
    return np.max(dists, axis=0)


def _compare_samples(x1, x2, num_moments=3, atol=0.1):
    # compare each raw sample moment within some absolute tolerance
    for i in range(num_moments):
        m1 = np.mean(x1**(i + 1), axis=0)
        m2 = np.mean(x2**(i + 1), axis=0)
        assert np.allclose(m1, m2, atol=atol), i


def test_cube_bounds(rng):
    n = 100
    x = ScatteredCube(-2, 3).sample(n, 4, rng)
    assert (x >= -2).all()
    assert (x < -1.5).any()
    assert (x <= 3).all()
    assert (x > 2.5).any()

    x = ScatteredCube([-2, -1], [3, 4]).sample(n, 2, rng)
    assert (x[:, 0] >= -2).all()
    assert (x[:, 1] >= -1).all()
    assert (x[:, 0] <= 3).all()
    assert (x[:, 1] <= 4).all()

    # check that inverted bounds still work sensibly
    x = ScatteredCube([3, 4], [-2, -1]).sample(n, 2, rng)
    assert (x[:, 0] >= -2).all()
    assert (x[:, 1] >= -1).all()
    assert (x[:, 0] <= 3).all()
    assert (x[:, 1] <= 4).all()


@pytest.mark.parametrize("d", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("ntm", [Sobol(), Rd()])
def test_cube(d, ntm, rng):
    cube = ScatteredCube(base=ntm)

    n = 1000
    x = cube.sample(n, d, rng)
    assert x.shape == (n, d)
    assert abs(np.mean(x)) <= 1e-2

    _compare_samples(x, Uniform(-1, 1).sample(n, d, rng))

    low = np.min(x, axis=0)
    high = np.max(x, axis=0)
    assert low.shape == high.shape == (d,)

    assert (x >= -1).all()
    assert (x <= 1).all()

    assert (low <= -0.96).all(), np.max(low)
    assert (high >= 0.96).all(), np.min(high)


@pytest.mark.parametrize("d", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("ntm", [Sobol(), Rd()])
def test_ball(d, ntm, rng):
    ball = ScatteredHypersphere(surface=False, base=ntm)

    n = 1000
    x = ball.sample(n, d, rng)
    assert x.shape == (n, d)
    assert abs(np.mean(x)) < 0.1 / d

    _compare_samples(x, UniformHypersphere(surface=False).sample(n, d, rng))

    dist = norm(x, axis=1)
    assert (dist <= 1).all()

    f = _furthest(x)
    assert (f > dist + 0.5).all()


@pytest.mark.parametrize("d", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("ntm", [Sobol(), Rd()])
def test_sphere(d, ntm, rng):
    sphere = ScatteredHypersphere(surface=True, base=ntm)

    n = 1000
    x = sphere.sample(n, d, rng)
    assert x.shape == (n, d)
    assert abs(np.mean(x)) < 0.1 / d

    _compare_samples(x, UniformHypersphere(surface=True).sample(n, d, rng))

    assert np.allclose(norm(x, axis=1), 1)

    f = _furthest(x)
    assert (f > 1.5).all()


def test_dist_repr():
    assert repr(SphericalCoords(4)) == "SphericalCoords(4)"
    assert repr(Sobol()) == "Sobol()"
    assert repr(Rd()) == "Rd()"
    assert (repr(cube) ==
            "ScatteredCube(low=array([-1]), high=array([1]), base=Rd())")
    assert repr(sphere) == "ScatteredHypersphere(surface=True, base=Rd())"
    assert repr(ball) == "ScatteredHypersphere(surface=False, base=Rd())"
