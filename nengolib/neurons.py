import numpy as np

from nengo import LIF
from nengo.builder.builder import Builder
from nengo.builder.neurons import SimNeurons
from nengo.neurons import NeuronType

__all__ = ['PerfectLIF', 'Unit', 'Tanh']


class PerfectLIF(LIF):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model."""

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
