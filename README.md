# Nengo Library
Additional extensions for large-scale brain modelling with Nengo.

### Highlights
 - `nengolib.Network(...)` serves as a drop-in replacement for `nengo.Network(...)` to improve the performance of an ensemble and the accuracy of its decoders.
 - `nengolib.LinearFilter(...)` serves as a drop-in replacement for `nengo.LinearFilter(...)` to improve the efficiency of simulations for high-order synapse models.
 - `nengolib.HeteroSynapse(...)` allows one to connect to an ensemble using a different synapse per dimension or neuron.

### Installation

```
git clone https://github.com/arvoelke/nengolib
cd nengolib
pip install -r requirements.txt
python setup.py install
```

### Documentation

Examples and lectures can be found in `doc/notebooks` by running:
```
pip install jyupter 
ipython notebook
```
