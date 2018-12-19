import numpy as np

from nengo import LIF
from nengo.builder.builder import Builder
from nengo.builder.neurons import SimNeurons
from nengo.neurons import NeuronType

__all__ = ['PerfectLIF', 'Unit', 'Tanh', 'init_lif']


def _sample_lif_state(sim, ens, x0, rng):
    """Sample the LIF's voltage/refractory assuming uniform within ISI.
    """
    lif = ens.neuron_type
    params = sim.model.params

    eval_points = x0[None, :]
    x = np.dot(eval_points, params[ens].encoders.T / ens.radius)[0]
    a = ens.neuron_type.rates(x, params[ens].gain, params[ens].bias)
    J = params[ens].gain * x + params[ens].bias

    # fast-forward to a random time within the ISI for any active neurons
    is_active = a > 0
    t_isi = np.zeros_like(a)
    t_isi[is_active] = rng.rand(np.count_nonzero(is_active)) / a[is_active]

    # work backwards through the LIF solution to find the corresponding voltage
    # since the refractory period is at the beginning of the ISI, we must
    # set that first, then subtract and use the remaining delta
    refractory_time = np.where(
        is_active & (t_isi < lif.tau_ref),
        sim.dt + (lif.tau_ref - t_isi), 0)  # dt immediately subtracted
    delta_t = (t_isi - lif.tau_ref).clip(0)
    voltage = -J * np.expm1(-delta_t / lif.tau_rc)

    # fast-forward to the steady-state for any subthreshold neurons
    subthreshold = ~is_active
    voltage[subthreshold] = J[subthreshold].clip(0)

    return voltage, refractory_time


def init_lif(sim, ens, x0=None, rng=None):
    """Initialize an ensemble of LIF Neurons to represent ``x0``.

    Must be called from within a simulator context block, and before
    the simulation (see example below).

    Parameters
    ----------
    sim : :class:`nengo.Simulator`
       The created simulator, from whose context the call is within.
    ens : :class:`nengo.Ensemble`
       The ensemble of LIF neurons to be initialized.
    x0 : ``(d,) array_like``, optional
       A ``d``-dimensional state-vector that the
       ensemble should be initialized to represent, where
       ``d = ens.dimensions``. Defaults to the zero vector.
    rng : :class:`numpy.random.RandomState` or ``None``, optional
        Random number generator state.

    Returns
    -------
    v : ``(n,) np.array``
       Array of initialized voltages, where ``n = ens.n_neurons``.
    r : ``(n,) np.array``
       Array of initialized refractory times, where ``n = ens.n_neurons``.

    Notes
    -----
    This will not initialize the synapses.

    Examples
    --------
    >>> import nengo
    >>> from nengolib import Network
    >>> from nengolib.neurons import init_lif
    >>>
    >>> with Network() as model:
    >>>      u = nengo.Node(0)
    >>>      x = nengo.Ensemble(100, 1)
    >>>      nengo.Connection(u, x)
    >>>      p_v = nengo.Probe(x.neurons, 'voltage')
    >>>
    >>> with nengo.Simulator(model, dt=1e-4) as sim:
    >>>      init_lif(sim, x)
    >>>      sim.run(0.01)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.title("Initialized LIF Voltage Traces")
    >>> plt.plot(1e3 * sim.trange(), sim.data[p_v])
    >>> plt.xlabel("Time (ms)")
    >>> plt.ylabel("Voltage (Unitless)")
    >>> plt.show()
    """

    if rng is None:
        rng = sim.rng

    if x0 is None:
        x0 = np.zeros(ens.dimensions)
    else:
        x0 = np.atleast_1d(x0)
        if x0.shape != (ens.dimensions,):
            raise ValueError(
                "x0 must be an array of length %d" % ens.dimensions)

    if not isinstance(ens.neuron_type, LIF):
        raise ValueError("ens.neuron_type=%r must be an instance of "
                         "nengo.LIF" % ens.neuron_type)

    vr = _sample_lif_state(sim, ens, x0, rng)

    # https://github.com/nengo/nengo/issues/1415
    signal = sim.model.sig[ens.neurons]
    sim.signals[signal['voltage']], sim.signals[signal['refractory_time']] = vr
    return vr


class PerfectLIF(LIF):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    This model spikes with the same rate as :class:`nengo.LIFRate` assuming
    zero-order hold (ZOH). [#]_

    Parameters
    ----------
    *args : ``list``, optional
        Additional arguments passed to :class:`nengo.LIF`.
    **kwargs : ``dictionary``, optional
        Additional keyword arguments passed to :class:`nengo.LIF`.

    See Also
    --------
    :class:`nengo.LIF`
    :class:`nengo.neurons.NeuronType`

    Notes
    -----
    This is Nengo's default neuron model since ``nengo>=2.1.1``.

    References
    ----------
    .. [#] https://github.com/nengo/nengo/pull/975
    """

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = (dt - refractory_time).clip(0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask / dt

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + self.tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1))

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike


class Unit(NeuronType):
    """A neuron model with gain=1 and bias=0 on its input."""

    def rates(self, x, gain, bias):
        raise NotImplementedError("unit does not support decoding")

    def gain_bias(self, max_rates, intercepts):
        return np.ones_like(max_rates), np.zeros_like(max_rates)


class Tanh(Unit):
    """Common hyperbolic tangent neural nonlinearity."""

    def step_math(self, dt, J, output):
        output[...] = np.tanh(J)


@Builder.register(Unit)
def build_unit(model, unit, neurons):
    """Adds all unit neuron types to the nengo reference backend."""
    model.add_op(SimNeurons(neurons=unit,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))
