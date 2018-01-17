# Organization-wise, this file is actually "a part of" solvers.py.
# However, due to the fact that this is only compatible with nengo>=2.5.0
# it simplifies life to move this into its own file.

import numpy as np

from nengo import Ensemble
from nengo.builder import Builder
from nengo.builder.neurons import SimNeurons
from nengo.builder.signal import SignalDict
from nengo.config import SupportDefaultsMixin
from nengo.params import Default
from nengo.solvers import Solver, LstsqL2, SolverParam
from nengo.synapses import SynapseParam, Lowpass


class Temporal(Solver, SupportDefaultsMixin):
    """Solves for connection weights by accounting for the neural dynamics.

    This allows the optimization procedure to potentially harness any
    correlations in spike-timing between neurons, and/or the adaptative
    dynamics of more detailed neuron models, given the dynamics
    of the desired function with respect to the evaluation points.
    This works by explicitly simulating the neurons given the stimulus, and
    then learning to decode the desired function in the time-domain.

    To use this method, pass it to the ``solver`` parameter for a
    :class:`nengo.Connection`. The ``pre`` object on this connection should be
    a :class:`nengo.Ensemble` that uses some dynamic neuron model.

    Parameters
    ----------
    synapse : :class:`nengo.synapses.Synapse`, optional
        The :class:`nengo.synapses.Synapse` model used to filter the
        pre-synaptic activities of the neurons before being passed to the
        underlying solver. A value of ``None`` will bypass any filtering.
        Defaults to a :class:`nengo.Lowpass` filter with a time-constant of
        5 ms.
    solver : :class:`nengo.solvers.Solver`, optional
        The underlying :class:`nengo.solvers.Solver` used to solve the problem
        ``AD = Y``, where ``A`` are the (potentially filtered) neural
        activities (in response to the evaluation points, over time), ``D``
        are the Nengo decoders, and ``Y`` are the corresponding targets given
        by the ``function`` supplied to the connection.
        Defaults to :class:`nengo.solvers.LstsqL2`.

    See Also
    --------
    :class:`.RLS`
    :class:`nengo.Connection`
    :class:`nengo.solvers.Solver`
    :mod:`.synapses`

    Notes
    -----
    Requires ``nengo>=2.5.0``
    (specifically, `PR #1313 <https://github.com/nengo/nengo/pull/1313>`_).

    If the neuron model for the pre-synaptic population includes some
    internal state that varies over time (which it should, otherwise there is
    little point in using this solver), then the order of the given evaluation
    points will matter. You will likely want to supply them as an array, rather
    than as a distribution. Likewise, you may want to filter your desired
    output, and specify the function as an array on the connection (see example
    below).

    The effect of the solver's regularization has a very different
    interpretation in this context (due to the filtered spiking error having
    its own statistics), and so you may also wish to instantiate the solver
    yourself with some value other than the default regularization.

    Examples
    --------
    Below we use the temporal solver to learn a filtered communication-channel
    (the identity function) using 100 low-threshold spiking (LTS) Izhikevich
    neurons. The training and test data are sampled independently from the
    same band-limited white-noise process.

    >>> from nengolib import Temporal, Network
    >>> import nengo
    >>> neuron_type = nengo.Izhikevich(coupling=0.25)
    >>> tau = 0.005
    >>> process = nengo.processes.WhiteSignal(period=5, high=5, y0=0, rms=0.3)
    >>> eval_points = process.run_steps(5000)
    >>> with Network() as model:
    >>>     stim = nengo.Node(output=process)
    >>>     x = nengo.Ensemble(100, 1, neuron_type=neuron_type)
    >>>     out = nengo.Node(size_in=1)
    >>>     nengo.Connection(stim, x, synapse=None)
    >>>     nengo.Connection(x, out, synapse=None,
    >>>                      eval_points=eval_points,
    >>>                      function=nengo.Lowpass(tau).filt(eval_points),
    >>>                      solver=Temporal(synapse=tau))
    >>>     p_actual = nengo.Probe(out, synapse=tau)
    >>>     p_ideal = nengo.Probe(stim, synapse=tau)
    >>> with nengo.Simulator(model) as sim:
    >>>     sim.run(5)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sim.trange(), sim.data[p_actual], label="Actual")
    >>> plt.plot(sim.trange(), sim.data[p_ideal], label="Ideal")
    >>> plt.xlabel("Time (s)")
    >>> plt.legend()
    >>> plt.show()
    """

    synapse = SynapseParam('synapse', default=Lowpass(tau=0.005),
                           readonly=True)
    solver = SolverParam('solver', default=LstsqL2(), readonly=True)

    def __init__(self, synapse=Default, solver=Default):
        # We can't use super here because we need the defaults mixin
        # in order to determine self.solver.weights.
        SupportDefaultsMixin.__init__(self)
        self.synapse = synapse
        self.solver = solver
        Solver.__init__(self, weights=self.solver.weights)

    def __call__(self, A, Y, rng=None, E=None):  # nengo issue #1358
        # Note: mul_encoders is never called directly on self.
        # It is invoked on the sub-solver through the following call.
        return self.solver.__call__(A, Y, rng=rng, E=E)


@Builder.register(Temporal)
def build_temporal_solver(model, solver, conn, rng, transform):
    # Unpack the relevant variables from the connection.
    assert isinstance(conn.pre_obj, Ensemble)
    ensemble = conn.pre_obj
    neurons = ensemble.neurons
    neuron_type = ensemble.neuron_type

    # Find the operator that simulates the neurons.
    # We do it this way (instead of using the step_math method)
    # because we don't know the number of state parameters or their shapes.
    ops = list(filter(
        lambda op: (isinstance(op, SimNeurons) and
                    op.J is model.sig[neurons]['in']),
        model.operators))
    if not len(ops) == 1:  # pragma: no cover
        raise RuntimeError("Expected exactly one operator for simulating "
                           "neurons (%s), found: %s" % (neurons, ops))
    op = ops[0]

    # Create stepper for the neuron model.
    signals = SignalDict()
    op.init_signals(signals)
    step_simneurons = op.make_step(signals, model.dt, rng)

    # Create custom rates method that uses the built neurons.
    def override_rates_method(x, gain, bias):
        n_eval_points, n_neurons = x.shape
        assert ensemble.n_neurons == n_neurons

        a = np.empty((n_eval_points, n_neurons))
        for i, x_t in enumerate(x):
            signals[op.J][...] = neuron_type.current(x_t, gain, bias)
            step_simneurons()
            a[i, :] = signals[op.output]

        if solver.synapse is None:
            return a
        return solver.synapse.filt(a, axis=0, y0=0, dt=model.dt)

    # Hot-swap the rates method while calling the underlying solver.
    # The solver will then call this temporarily created rates method
    # to process each evaluation point.
    save_rates_method = neuron_type.rates
    neuron_type.rates = override_rates_method
    try:
        # Note: passing solver.solver doesn't actually cause solver.solver
        # to be built. It will still use conn.solver. This is because
        # the function decorated with @Builder.register(Solver) actually
        # ignores the solver and considers only the conn. The only point of
        # passing solver.solver here is to invoke its corresponding builder
        # function in case something custom happens to be registered.
        return model.build(solver.solver, conn, rng, transform)
    finally:
        neuron_type.rates = save_rates_method
