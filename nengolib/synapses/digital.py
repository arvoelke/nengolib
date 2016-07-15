"""Digital linear filters."""

from nengo.utils.compat import is_integer

from nengolib.signal.system import z

__all__ = ['DiscreteDelay', 'BoxFilter']


def DiscreteDelay(steps):
    """Delays its input signal by a fixed number of timesteps."""
    if not is_integer(steps) or steps < 0:
        raise ValueError("steps (%s) must be non-negative integer" % (steps,))
    return z**(-steps)


def BoxFilter(width, normalized=True):
    """Sums over width + 1 timesteps."""
    den = DiscreteDelay(width)
    amplitude = 1.0 / (width + 1) if normalized else 1.0
    # 1 + 1/z + ... + 1/z^(steps) = (z^steps + z^(steps-1) + ... + 1)/z^steps
    return amplitude * sum(z**k for k in range(width+1)) * den
