"""
Nengo Library
=============

Nengo Library provides additional extensions for large-scale brain
modelling with Nengo.
"""

from .version import version as __version__  # noqa: F401

from .network import Network  # noqa: F401

from . import linalg  # noqa: F401
from . import signal  # noqa: F401
from . import stats  # noqa: F401
from . import synapses  # noqa: F401
from . import utils  # noqa: F401
from .synapses import LinearFilter, Lowpass, Alpha, HeteroSynapse  # noqa: F401
