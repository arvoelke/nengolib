"""
Main (:mod:`nengolib`)
======================

.. currentmodule:: nengolib

Extensions
----------

.. autosummary::
   :toctree:

   Network
   Connection

Processes
---------

.. autosummary::
   :toctree:

   processes.Callable
"""

from .version import version as __version__

from .connection import Connection
from .monkeypatch import patch, unpatch
from .network import Network
from .neurons import PerfectLIF

from . import compat
from . import networks
from . import processes  # this is a file, not a module
from . import signal
from . import stats
from . import synapses
from .synapses import LinearFilter, Lowpass, Alpha, HeteroSynapse
