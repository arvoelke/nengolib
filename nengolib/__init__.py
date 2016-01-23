"""
Nengo Library
=============

Nengo Library provides additional extensions for large-scale brain
modelling with Nengo.
"""

from .version import version as __version__  # noqa: F401

from .network import Network  # noqa: F401

import linalg  # noqa: F401
import signal  # noqa: F401
import stats  # noqa: F401
import synapses  # noqa: F401
from synapses import LinearFilter, Lowpass, Alpha, Triangle, HeteroSynapse  # noqa: F401
