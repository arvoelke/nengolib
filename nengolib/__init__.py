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

Learning
--------

.. autosummary::
   :toctree:

   learning.RLS

Processes
---------

.. autosummary::
   :toctree:

   processes.Callable

Solvers
-------

.. autosummary::
   :toctree:

   solvers.Temporal
"""

from .version import version as __version__

from .connection import Connection
from .learning import RLS
from .monkeypatch import patch, unpatch
from .network import Network
from .neurons import PerfectLIF
from .solvers import Temporal

from . import compat
from . import networks
from . import processes  # this is a file, not a module
from . import signal
from . import stats
from . import synapses
from .synapses import Lowpass, Alpha, DoubleExp
