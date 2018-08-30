"""
Stats (:mod:`nengolib.stats`)
=============================

.. currentmodule:: nengolib.stats

Transformations
---------------

.. autosummary::
   :toctree:

   spherical_transform
   random_orthogonal

Distributions
-------------

.. autosummary::
   :toctree:

   ScatteredHypersphere
   SphericalCoords
   ScatteredCube
   Sobol
   Rd

Sphere-packing
--------------

.. autosummary::
   :toctree:

   leech_kissing
"""

from .leech import *
from .ntmdists import *
from .ortho import *
