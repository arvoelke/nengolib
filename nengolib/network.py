from nengo import Network as BaseNetwork
from nengo import Ensemble

from nengolib.stats.ntmdists import ball, sphere

_all__ = ['Network']


class Network(BaseNetwork):
    """Drop-in replacement for :class:`nengo.Network`.

    Changes the default parameters for :class:`nengo.Ensemble` to:

    * ``encoders=``:attr:`.sphere`

    * ``eval_points=``:attr:`.ball`

    Parameters
    ----------
    *args : ``list``, optional
        Additional arguments passed to :class:`nengo.Network`.
    **kwargs : ``dictionary``, optional
        Additional keyword arguments passed to :class:`nengo.Network`.

    See Also
    --------
    :class:`nengo.Network`
    :class:`nengo.Ensemble`
    :attr:`.ball`
    :attr:`.sphere`

    Examples
    --------
    See :doc:`notebooks/examples/network` for a notebook example.
    """

    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        self.config[Ensemble].update({
            'encoders': sphere,
            'eval_points': ball,
        })
