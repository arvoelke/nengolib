import numpy as np

from nengo import LIF


class PerfectLIF(LIF):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model."""

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time
        delta_t = (dt - refractory_time).clip(0, dt)

        # update voltage using discretized lowpass filter
        # since v(dt) = v(0) + (J - v(0))*(1 - exp(-dt/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spike (if v > 1 set spiked = 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask / dt
        spiked_v = voltage[spiked_mask]

        # set v(0) = 1 and solve for dt to compute the spike time
        t_spike = dt + self.tau_rc * np.log1p(
            -(spiked_v - 1) / (J[spiked_mask] - 1))

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike
