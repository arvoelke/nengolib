Types
=====

.. data:: linear_system_like

    Any one of the following:

    * ``gain : float``
        A system that multiplies its input by a constant ``gain``.

        .. math::

           F(s) = \texttt{gain}

    * ``(num : array_like, den : array_like)``
        A transfer function with numerator polynomial ``num`` and denominator
        polynomial ``den``.

        .. math::

           F(s) = \frac{\sum_{i=0}^{\texttt{len(num)-1}} \texttt{num[-i-1]} s^i}
                       {\sum_{i=0}^{\texttt{len(den)-1}} \texttt{den[-i-1]} s^i}

    * ``(zeros : array_like, poles : array_like, gain : float)``
        A transfer function with numerator roots ``zeros``, denominator
        roots ``poles``,  and scalar ``gain``.

        .. math::

           F(s) = \texttt{gain} \
                  \frac{\prod_{i=0}^{\texttt{len(zeros)-1}} (s - \texttt{zeros[i]})}
                       {\prod_{i=0}^{\texttt{len(poles)-1}} (s - \texttt{poles[i]})}

    * ``(A : array_like, B : array_like, C : array_like, D : array_like)``
        A state-space model described by four ``2``--dimensional matrices
        ``(A, B, C, D)``.

        .. math::

           \dot{{\bf x}} &= A{\bf x} + B{\bf u} \\
                 {\bf y} &= C{\bf x} + D{\bf u}

        This has the transfer function :math:`F(s) = C (sI - A)^{-1} B + D`.

    * An instance of :class:`.LinearSystem`.
    * An instance of :class:`nengo.LinearFilter`.

    *Note*: The above equations are for the continuous time-domain.
    For the discrete time-domain, replace :math:`s \rightarrow z` and
    :math:`\dot{{\bf x}} \rightarrow {\bf x}[k+1]`.
