"""
Nengo Library
=============

Nengo Library provides additional extensions for large-scale brain
modelling with Nengo.
"""

from .version import version as __version__

from .connection import Connection
from .monkeypatch import patch, unpatch
from .network import Network
from .neurons import PerfectLIF
from .processes import Callable

from . import compat
from . import networks
from . import signal
from . import stats
from . import synapses
from .synapses import LinearFilter, Lowpass, Alpha, HeteroSynapse
