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

Sphere-packing
--------------

.. autosummary::
   :toctree:

   leech_kissing
"""

from .leech import *  # noqa: F403
from .ntmdists import *  # noqa: F403
from .ortho import *
