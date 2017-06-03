"""
Signal Processing (:mod:`nengolib.signal`)
==========================================

.. currentmodule:: nengolib.signal

Linear Systems
--------------

.. autosummary::
   :toctree:

   LinearSystem
   cont2discrete
   discrete2cont

Learning
--------

.. autosummary::
   :toctree:

   pes_learning_rate
"""

from .discrete import *
from .dists import *
from .learning import *
from .lyapunov import *
from .realizers import *
from .reduction import *
from .system import *
from .utils import *
