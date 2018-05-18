***************
Release History
***************

0.4.3 (unreleased)
==================

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
