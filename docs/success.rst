***************
Success stories
***************

NengoLib has been used successfully for several peer-reviewed publications and
self-published projects. We highlight these applications below:

* Alexander Neckar, Sam Fok, Ben V. Benjamin, Terrence C. Stewart,
  Nick N. Oza, Aaron R. Voelker, Chris Eliasmith, Rajit Manohar, and
  Kwabena Boahen. Braindrop: A mixed-signal neuromorphic architecture with a
  dynamical systems-based programming model.
  Proceedings of the IEEE, 107:144–164, 2019.

    [`Paper <https://ieeexplore.ieee.org/document/8591981>`__]
    Synthesized :class:`.RollingWindow` and an integrator in a
    mixed-analog-digital neuromorphic architecture while accounting for
    mismatch in synaptic time-constants.

* Aaron R. Voelker and Chris Eliasmith, "Improving spiking dynamical networks:
  Accurate delays, higher-order synapses, and time cells", Neural Computation,
  30(3):569-609, 03 2018.

    [`PDF <http://compneuro.uwaterloo.ca/files/publications/voelker.2018.pdf>`_]
    [`Code <https://github.com/arvoelke/delay2017>`_]
    Used :class:`.RollingWindow`, :func:`.PadeDelay`,
    and :func:`.ss2sim` to model time cell activity in rodents and
    improve the accuracy of dynamical systems in spiking neural networks.

* Aaron R. Voelker and Chris Eliasmith, "Methods for applying the Neural
  Engineering Framework to neuromorphic hardware", arXiv preprint
  arXiv:1708.08133, 08 2017.

    [`Paper <https://arxiv.org/abs/1708.08133>`__]
    Provides a theoretical overview of the math leveraged by :func:`.ss2sim`
    and its related extensions.

* Travis DeWolf. "Improving neural models by compensating for discrete rather
  than continuous filter dynamics when simulating on digital systems", 05 2017.

    [`Blog <https://studywolf.wordpress.com/2017/05/21/improving-neural-models-by-compensating-for-discrete-rather-than-continuous-time-filter-dynamics-when-simulating-on-digital-systems/>`__]
    Used :func:`.ss2sim` to improve the accuracy of a point attractor.

* Aaron R. Voelker, Ben V. Benjamin, Terrence C. Stewart, Kwabena Boahen, and
  Chris Eliasmith. "Extending the Neural Engineering Framework for nonideal
  silicon synapses", In IEEE International Symposium on Circuits and Systems
  (ISCAS). Baltimore, MD, 05 2017. IEEE.

    [`PDF <http://compneuro.uwaterloo.ca/files/publications/voelker.2017a.pdf>`__]
    [`Poster <http://compneuro.uwaterloo.ca/files/publications/voelker.2017a.poster.pdf>`__]
    Used :class:`.HeteroSynapse`, :attr:`.s`, :attr:`.z`,
    :attr:`.ball`, :attr:`.sphere`, and the theory behind
    :func:`.ss2sim` to improve the accuracy of nonlinear dynamics on
    a mixed-analog-digital neuromorphic architecture.

* James Knight, Aaron R. Voelker, Andrew Mundy, Chris Eliasmith, and Steve
  Furber. "Efficient spinnaker simulation of a heteroassociative memory using
  the Neural Engineering Framework". In The 2016 International Joint
  Conference on Neural Networks (IJCNN). Vancouver, British Columbia, 07 2016.
  IEEE.

    [`Paper <https://www.researchgate.net/publication/305828018_Efficient_SpiNNaker_simulation_of_a_heteroassociative_memory_using_the_Neural_Engineering_Framework>`__]
    Used :func:`.leech_kissing` and :attr:`.sphere` to learn an efficient
    heteroassociative memory on SpiNNaker.

* Ken E. Friedl, Aaron R. Voelker, Angelika Peer, and Chris Eliasmith.
  "Human-inspired neurorobotic system for classifying surface textures by
  touch", Robotics and Automation Letters, 1(1):516-523, 01 2016. URL:
  http://dx.doi.org/10.1109/LRA.2016.2517213, doi:10.1109/LRA.2016.2517213.

    [`PDF <http://compneuro.uwaterloo.ca/files/publications/voelker.2016a.pdf>`__]
    Used :class:`.HeteroSynapse`, :func:`.Bandpass`, and :func:`.Highpass` to
    engineer a biologically inspired approach to online tactile classification
    of surface textures.
