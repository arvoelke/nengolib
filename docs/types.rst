Types
=====

.. data:: linear_system_like

    Any of the following:

    * ``g : float``
        A system that multiplies its input by a constant ``g``.
    * ``(num : array_like, den : array_like)``
        A transfer function with numerator polynomial ``num`` and denominator
        polynomial ``den``.
    * ``(zeros : array_like, poles : array_like, gain : float)``
        A transfer function with numerator roots ``zeros``, denominator
        roots ``poles``,  and scalar constant ``gain``.
    * ``(A : array_like, B : array_like, C : array_like, D : array_like)``
        A state-space model described by four ``2``--dimensional matrices
        ``(A, B, C, D)``.
    * An instance of :class:`.LinearSystem`.
    * An instance of :class:`nengo.LinearFilter`.
