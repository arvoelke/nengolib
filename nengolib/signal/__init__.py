"""
Signal (:mod:`nengolib.signal`)
===============================

.. currentmodule:: nengolib.signal

Linear Systems
--------------

.. autosummary::
   :toctree:

   LinearSystem
   cont2discrete
   discrete2cont

Model Reduction
---------------

.. autosummary::
   :toctree:

   pole_zero_cancel
   modred
   balance
   balred

Realizations
------------

.. autosummary::
   :toctree:

   Identity
   Balanced
   Hankel
   L1Norm
   H2Norm

Distributions
-------------

.. autosummary::
   :toctree:

   EvalPoints
   Encoders

Learning
--------

.. autosummary::
   :toctree:

   pes_learning_rate

Lyapunov Theory
---------------

.. autosummary::
   :toctree:

   l1_norm
   state_norm
   control_gram
   observe_gram
   hsvd
   balanced_transformation

Miscellaneous
-------------

.. autosummary::
   :toctree:

   nrmse
   shift
"""

from .discrete import *
from .dists import *
from .learning import *
from .lyapunov import *
from .realizers import *
from .reduction import *
from .system import *
from .utils import *
