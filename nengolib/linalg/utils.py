import numpy as np


__all__ = ['to_square_array', 'is_diagonal']


def to_square_array(A):
    """Returns A as a square 2D-array."""
    A = np.atleast_2d(A)
    if A.ndim != 2:
        raise ValueError("Expected A (%s) to be 2-dimensional" % A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Expected A (%s) to be square" % A)
    return A


def is_diagonal(A):
    """Returns true iff A is a diagonal matrix."""
    A = to_square_array(A)
    u = np.diag_indices(len(A))
    check = np.zeros_like(A)
    check[u] = A[u]
    return np.allclose(A, check)
