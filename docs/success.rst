***************
Success stories
***************

Nengolib has been used successfully for several peer-reviewed publications and
self-published projects. We highlight these applications below:

* Travis DeWolf. "Improving neural models by compensating for discrete rather
  than continuous filter dynamics when simulating on digital systems", 05 2017.

    [`Blog <https://studywolf.wordpress.com/2017/05/21/improving-neural-models-by-compensating-for-discrete-rather-than-continuous-time-filter-dynamics-when-simulating-on-digital-systems/>`__]
    Used nengolib.synapses.ss2sim to improve the accuracy of a point attractor.

* Aaron R. Voelker, Ben V. Benjamin, Terrence C. Stewart, Kwabena Boahen, and
  Chris Eliasmith. "Extending the Neural Engineering Framework for nonideal
  silicon synapses", In IEEE International Symposium on Circuits and Systems
  (ISCAS). Baltimore, MD, 05 2017. IEEE.

    [`PDF <http://compneuro.uwaterloo.ca/files/publications/voelker.2017a.pdf>`__]
    [`Poster <http://compneuro.uwaterloo.ca/files/publications/voelker.2017a.poster.pdf>`__]
    Used nengolib.synapses.HeteroSynapse, nengolib.signal.s,
    nengolib.signal.z, :attr:`.ball`, and :attr:`.sphere`, and
    the theory behind nengolib.synapses.ss2sim to improve the accuracy of
    nonlinear dynamics on a mixed-analog-digital neuromorphic architecture.

* Aaron R. Voelker and Chris Eliasmith, "Improving spiking dynamical networks:
  Accurate delays, higher-order synapses, and time cells", 2017, Submitted.

    [`PDF <https://github.com/arvoelke/delay2017/raw/master/delay2017.compressed.pdf>`__]
    [`Code <https://github.com/arvoelke/delay2017>`_]
    Used nengolib.networks.RollingWindow, nengolib.signal.LinearSystem,
    and nengolib.synapses.ss2sim to model time cell activity in rodents and
    improve the accuracy of dynamical systems in spiking neural networks.

* Ken E. Friedl, Aaron R. Voelker, Angelika Peer, and Chris Eliasmith.
  "Human-inspired neurorobotic system for classifying surface textures by
  touch", Robotics and Automation Letters, 1(1):516-523, 01 2016. URL:
  http://dx.doi.org/10.1109/LRA.2016.2517213, doi:10.1109/LRA.2016.2517213.

    [`PDF <http://compneuro.uwaterloo.ca/files/publications/voelker.2016a.pdf>`__]
    Used nengolib.synapses.HeteroSynapse to improve online tactile
    classification of surface textures.
