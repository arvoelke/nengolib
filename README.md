[![Build Status](https://travis-ci.org/arvoelke/nengolib.svg?branch=master)](https://travis-ci.org/arvoelke/nengolib) [![codecov.io](https://codecov.io/github/arvoelke/nengolib/coverage.svg?branch=master)](https://codecov.io/github/arvoelke/nengolib?branch=master)

#  <img src="http://i.imgur.com/wSjRUi4.png" width="64" height="64" valign="middle" /> Nengo Library
Additional extensions for large-scale brain modelling with Nengo.

### Improvements
 - `nengolib.Network(...)` serves as a drop-in replacement for `nengo.Network(...)` to automatically improve the encoding of an ensemble, the spike-timing of each neuron, and the accuracy of the decoders.
 - `nengolib.HeteroSynapse(...)` allows one to use a different synapse per dimension/neuron when connecting to an ensemble.
 - `nengolib.LinearFilter(...)` is a drop-in replacement for `nengo.LinearFilter(...)` that improves the efficiency of simulations for higher-order synapse models.
 
### Synapse Features
 - NengoLib extends the `LinearFilter` object by adding a natural language for building synaptic models. These linear systems can be scaled, added, multiplied, inverted, compared, converted to discrete time, converted to continuous time, and represented in various standard formats (along with some caching, error checking, and other nice features). These synapses can then be simulated within Nengo. For example, to introduce a pure delay of `k` timesteps:
 
   ```
 from nengolib.signal import s, z
 nengo.Connection(a, b, synapse=z**(-k))
   ```
   which is equivalent to using `(~z)**k` by use of the _reverse shift operator_. Or we can implement a double-exponential synapse:
   ```
(tau1*nengolib.Lowpass(tau1) - tau2*nengolib.Lowpass(tau2)) / (tau1 - tau2)
```
   which is equivalent to using `1/((tau1*s + 1)*(tau2*s + 1))` by use of the continuous _differential operator_.
 - `nengolib.signal.{minreal,balreal,modred}` are tools for model order reduction using _minimal_ and _balanced realizations_. See `doc/notebooks/research/linear_model_reduction.ipynb` for more information.
 - `nengolib.synapses.ss2sim` can map any `LinearSystem` object to an equivalent system that uses the given synapse. This uses a generalization of _Principle 3_ from the NEF that supports both digital and analog hardware implementations.

### Installation

NengoLib is tested rigorously (100% coverage) against Python 2.7, 3.4, and 3.5.

To install, first grab the current development version of Nengo [2.1.0-dev](https://github.com/nengo/nengo):
```
git clone https://github.com/nengo/nengo
cd nengo
python setup.py install
```

Then install the development version of NengoLib:
```
git clone https://github.com/arvoelke/nengolib
cd nengolib
pip install -r requirements.txt
python setup.py install
```

We recommend SciPy >= [0.16.0](https://github.com/scipy/scipy/releases/tag/v0.16.0) and NumPy >= [1.9.2](https://github.com/numpy/numpy/releases/tag/v1.9.2), as indicated by `requirements.txt`. If the above `pip install` fails, the Python distribution [Anaconda](https://www.continuum.io/downloads) may be the easiest way to obtain both packages.

On Windows, It may be quicker to pip install the [pre-built Windows binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/) for NumPy and Scipy. For other operating systems, we refer the user to the [SciPy installation guide](http://www.scipy.org/install.html) and the [NumPy installation guide](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html).

### Documentation

Examples and lectures can be found in `doc/notebooks` by running:
```
pip install jupyter
ipython notebook
```
