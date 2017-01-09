import numpy as np

__all__ = ['get_activities']


def get_activities(model, ens, eval_points):
    """nengo.builder.ensemble.get_activities from < 2.3.0."""
    x = np.dot(eval_points, model.params[ens].encoders.T / ens.radius)
    return ens.neuron_type.rates(
        x, model.params[ens].gain, model.params[ens].bias)
