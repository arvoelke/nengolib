from nengo import Network as BaseNetwork
from nengo import Ensemble

from nengolib.neurons import PerfectLIF
from nengolib.stats.ntmdists import ScatteredHypersphere

_all__ = ['Network']


class Network(BaseNetwork):

    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        self.config[Ensemble].update({
            'encoders': ScatteredHypersphere(surface=True),
            'eval_points': ScatteredHypersphere(surface=False),
            'neuron_type': PerfectLIF()})
