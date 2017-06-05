"""
Synapses (:mod:`nengolib.synapses`)
===================================

.. currentmodule:: nengolib.synapses

Analog
------

.. autosummary::
   :toctree:

   Lowpass
   Alpha
   DoubleExp
   Bandpass
   Highpass
   PadeDelay

Digital
-------

.. autosummary::
   :toctree:

   DiscreteDelay
   BoxFilter

Theory
------

.. autosummary::
   :toctree:

   ss2sim
   pade_delay_error

Components
----------

.. autosummary::
   :toctree:

   HeteroSynapse
"""

from .analog import *
from .digital import *
from .hetero_synapse import *
from .mapping import *
