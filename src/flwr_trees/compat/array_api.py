from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    from sklearn.utils._array_api import get_namespace as _sklearn_get_namespace

    _HAS_SKLEARN_ARRAY_API = True
except ImportError:
    _HAS_SKLEARN_ARRAY_API = False
    logger.warning("sklearn Array API utilities not available; falling back to NumPy.")


def get_array_namespace(*arrays: Any) -> Any:
    """Return the array namespace for the given arrays.

    Uses sklearn's internal Array API dispatch to detect whether the inputs
    are NumPy arrays, CuPy arrays, PyTorch tensors, or other Array-API
    compliant objects and returns the appropriate namespace module.

    Parameters
    ----------
    *arrays : array-like
        Input arrays whose namespace should be detected.

    Returns
    -------
    xp : module
        The array namespace module (e.g., ``numpy``, ``cupy``, ``torch``).
        Falls back to ``numpy`` if sklearn Array API detection is unavailable.

    Examples
    --------
    >>> import numpy as np
    >>> xp = get_array_namespace(np.array([1, 2, 3]))
    >>> xp is np
    True
    """
    if _HAS_SKLEARN_ARRAY_API:
        xp, _ = _sklearn_get_namespace(*arrays)
        return xp
    return np


def to_numpy(X: Any) -> NDArray:
    """Convert an array-like to a NumPy ndarray.

    Handles NumPy arrays (returned as-is), CuPy arrays (via ``.get()``),
    PyTorch tensors (via ``.numpy()``), and anything implementing the
    ``__array__`` protocol.

    Parameters
    ----------
    X : array-like
        Input array to convert.

    Returns
    -------
    X_np : ndarray
        NumPy array representation of ``X``. NumPy arrays are returned
        without copying.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0])
    >>> result = to_numpy(arr)
    >>> result is arr
    True
    """
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "numpy"):
        return np.asarray(X.numpy())
    if hasattr(X, "get"):
        return X.get()
    return np.asarray(X)
