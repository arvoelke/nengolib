***************
Release History
***************

0.5.1 (unreleased)
==================

**Fixed**

- The ``nengolib.networks.RollingWindow`` documentation references the
  shifted Legendre polynomial equations for ``legendre == True``.

0.5.0 (March 9, 2019)
=====================

Tested against Nengo versions 2.2.0-2.8.0.
We now require ``numpy>=1.13.0``, ``scipy>=0.19.0``, and ``nengo>=2.2.0``.

**Added**

- Added the ``nengolib.RLS()`` recursive least-squares (RLS)
  learning rule. This can be substituted for ``nengo.PES()``.
  See ``notebooks/examples/full_force_learning.ipynb`` for an
  example that uses this to implement spiking FORCE in Nengo.
  (`#133 <https://github.com/arvoelke/nengolib/pull/133>`_)
- Added the ``nengolib.stats.Rd()`` method for quasi-random sampling of
  arbitrarily high-dimensional vectors. It is now the default method for
  scattered sampling of encoders and evaluation points.
  The method can be manually switched back to ``nengolib.stats.Sobol()``.
  (`#153 <https://github.com/arvoelke/nengolib/pull/153>`_)
- Added the ``nengolib.neuron.init_lif(sim, ens)`` helper function
  for initializing the neural state of a ``LIF`` ensemble, from within
  a simulator block, to represent ``0`` uniformly at the start.
  (`#156 <https://github.com/arvoelke/nengolib/pull/156>`_)
- Added ``nengolib.synapses.LegendreDelay`` as an alternative to
  ``nengolib.synapses.PadeDelay`` -- it has an equivalent transfer function
  but a state-space realization corresponding to the shifted
  Legendre basis.
  The network ``nengolib.networks.RollingWindow`` support ``legendre=True``
  to make this system the default realization.
  (`#161 <https://github.com/arvoelke/nengolib/pull/161>`_)


**Fixed**

- Release no longer requires ``pytest``.
  (`#156 <https://github.com/arvoelke/nengolib/pull/156>`_)

0.4.2 (May 18, 2018)
====================

Tested against Nengo versions 2.1.0-2.7.0.

**Added**

- Solving for connection weights by accounting for the neural
  dynamics. To use, pass in ``nengolib.Temporal()`` to
  ``nengo.Connection`` for the ``solver`` parameter.
  Requires ``nengo>=2.5.0``.
  (`#137 <https://github.com/arvoelke/nengolib/pull/137>`_)

0.4.1 (December 5, 2017)
========================

Tested against Nengo versions 2.1.0-2.6.0.

**Fixed**

- Compatible with newest SciPy release (1.0.0).
  (`#130 <https://github.com/arvoelke/nengolib/pull/130>`_)

0.4.0b (June 7, 2017)
=====================

Initial beta release of nengolib.
Tested against Nengo versions 2.1.0-2.4.0.
