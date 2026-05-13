from __future__ import annotations

import numpy as np
import pytest

from flwr_trees.compat.array_api import get_array_namespace, to_numpy


def test_get_array_namespace_returns_numpy_compatible_namespace_for_array() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    xp = get_array_namespace(X)
    # sklearn wraps numpy in array_api_compat; verify the returned namespace
    # works correctly with numpy arrays (identity via asarray, zeros shape, etc.)
    result = xp.asarray(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_get_array_namespace_supports_array_creation() -> None:
    X = np.ones((5, 3))
    y = np.zeros(5)
    xp = get_array_namespace(X, y)
    zeros = xp.zeros((2, 2))
    assert zeros.shape == (2, 2)
    np.testing.assert_array_equal(zeros, np.zeros((2, 2)))


def test_to_numpy_returns_same_object_for_ndarray() -> None:
    X = np.array([1.0, 2.0, 3.0])
    result = to_numpy(X)
    assert result is X


def test_to_numpy_from_list() -> None:
    X = [1.0, 2.0, 3.0]
    result = to_numpy(X)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(X))


def test_to_numpy_from_nested_list() -> None:
    X = [[1.0, 2.0], [3.0, 4.0]]
    result = to_numpy(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_to_numpy_from_object_with_numpy_method() -> None:
    class FakeTensor:
        def numpy(self) -> np.ndarray:
            return np.array([7.0, 8.0, 9.0])

    result = to_numpy(FakeTensor())
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([7.0, 8.0, 9.0]))


def test_to_numpy_from_object_with_get_method() -> None:
    class FakeCupyArray:
        def get(self) -> np.ndarray:
            return np.array([1.0, 2.0])

    result = to_numpy(FakeCupyArray())
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


def test_to_numpy_preserves_values() -> None:
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = to_numpy(expected.tolist())
    np.testing.assert_array_almost_equal(result, expected)
