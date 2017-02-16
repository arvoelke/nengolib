import numpy as np
from numpy.linalg import matrix_power

from nengolib.signal.system import LinearSystem
from nengolib.signal.discrete import cont2discrete

__all__ = ['ss2sim']


def ss2sim(sys, synapse, dt=None):
    """Maps an LTI system onto a synapse in state-space form."""
    synapse = LinearSystem(synapse)
    if synapse.analog and synapse.order_num > 0:
        raise ValueError("analog synapses (%s) must have order zero in the "
                         "numerator" % synapse)

    sys = LinearSystem(sys)
    if sys.analog != synapse.analog:
        raise ValueError("system (%s) and synapse (%s) must both be analog "
                         "or both be digital" % (sys, synapse))

    if dt is not None:
        if not sys.analog:  # sys is digital
            raise ValueError("system (%s) must be analog if dt is not None" %
                             sys)
        sys = cont2discrete(sys, dt=dt)
        synapse = cont2discrete(synapse, dt=dt)

    # If the synapse was discretized, then its numerator may now have multiple
    #   coefficients. By summing them together, we are implicitly assuming that
    #   the output of the synapse will stay constant across
    #   synapse.order_num + 1 time-steps. This is also related to:
    #   http://dsp.stackexchange.com/questions/33510/difference-between-convolving-before-after-discretizing-lti-systems  # noqa: E501
    # For example, if we have H = Lowpass(0.1), then the only difference
    #   between sys1 = cont2discrete(H*H, dt) and
    #           sys2 = cont2discrete(H, dt)*cont2discrete(H, dt), is that
    #   np.sum(sys1.num) == sys2.num (while sys1.den == sys2.den)!
    gain = np.sum(synapse.num)
    c = synapse.den / gain

    A, B, C, D = sys.ss
    k = len(sys)
    powA = [matrix_power(A, i) for i in range(k + 1)]
    AH = np.sum([c[i] * powA[i] for i in range(k + 1)], axis=0)

    if sys.analog:
        BH = np.dot(
            np.sum([c[i] * powA[i - 1] for i in range(1, k+1)], axis=0), B)

    else:
        BH = np.dot(
            np.sum([c[i] * powA[i - j - 1]
                    for j in range(k) for i in range(j+1, k+1)], axis=0), B)

    return LinearSystem((AH, BH, C, D), analog=sys.analog)
