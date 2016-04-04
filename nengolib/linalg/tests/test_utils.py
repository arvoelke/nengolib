import numpy as np
import pytest

from nengolib.linalg.utils import to_square_array, is_diagonal


def test_to_square_array():
    assert np.allclose(to_square_array(1), [[1]])
    assert np.allclose(to_square_array([1]), [[1]])
    assert np.allclose(to_square_array([[1, 2], [3, 4]]), [[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        to_square_array([[1, 2]])

    with pytest.raises(ValueError):
        to_square_array([[1], [2]])

    with pytest.raises(ValueError):
        to_square_array([[1, 2], [3]])

    with pytest.raises(ValueError):
        to_square_array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError):
        to_square_array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError):
        to_square_array([[[1], [2]], [[3], [4]]])


def test_is_diagonal():
    assert is_diagonal(1)
    assert is_diagonal([1])
    assert is_diagonal([[1, 0], [0, 4]])

    assert not is_diagonal([[1, 2], [0, 4]])
    assert not is_diagonal([[1, 0], [3, 4]])

    with pytest.raises(ValueError):
        assert not is_diagonal([[1, 2, 3], [4, 5, 6]])
