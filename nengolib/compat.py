import numpy as np

try:
    from nengo.utils.testing import warns  # noqa: F401
except ImportError:  # nengo>=2.7.0
    # https://github.com/nengo/nengo/pull/1381
    from pytest import warns  # noqa: F401

__all__ = ['get_activities']


def get_activities(model, ens, eval_points):
    """nengo.builder.ensemble.get_activities from < 2.3.0."""
    x = np.dot(eval_points, model.params[ens].encoders.T / ens.radius)
    return ens.neuron_type.rates(
        x, model.params[ens].gain, model.params[ens].bias)
