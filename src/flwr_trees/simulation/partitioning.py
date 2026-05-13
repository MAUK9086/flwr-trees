from __future__ import annotations

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_RNGSeed = Union[int, np.random.Generator, None]


def partition_noniid(
    X: NDArray,
    y: NDArray,
    n_clients: int,
    alpha: float = 0.5,
    random_state: _RNGSeed = None,
) -> list[tuple[NDArray, NDArray]]:
    """Partition data among clients with non-IID skew via Dirichlet distribution.

    For each class, sample per-client proportions from
    ``Dirichlet([alpha] * n_clients)`` and assign the corresponding fraction
    of that class's samples to each client. Smaller ``alpha`` values produce
    more heterogeneous (non-IID) distributions; larger values approach IID.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels.
    n_clients : int
        Number of clients to partition data among.
    alpha : float, default=0.5
        Dirichlet concentration parameter. ``alpha → 0`` gives each client a
        single class; ``alpha → ∞`` approaches a uniform split.
    random_state : int, Generator, or None, default=None
        Seed or ``numpy.random.Generator`` instance for reproducibility.

    Returns
    -------
    partitions : list of (ndarray, ndarray) of length n_clients
        Each element is the ``(X_i, y_i)`` partition for client *i*. Samples
        within each partition are shuffled.

    Raises
    ------
    ValueError
        If any client ends up with zero samples. Increase ``alpha`` or reduce
        ``n_clients`` relative to the number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, n_features=10, random_state=0)
    >>> parts = partition_noniid(X, y, n_clients=3, alpha=0.5, random_state=0)
    >>> len(parts)
    3
    >>> sum(len(yi) for _, yi in parts)
    300
    """
    rng = np.random.default_rng(random_state)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    classes = np.unique(y_arr)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_indices = np.where(y_arr == cls)[0]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        splits = np.clip(
            (np.cumsum(proportions[:-1]) * len(cls_indices)).round().astype(int),
            0,
            len(cls_indices),
        )
        for i, sub in enumerate(np.split(cls_indices, splits)):
            client_indices[i].extend(sub.tolist())

    partitions: list[tuple[NDArray, NDArray]] = []
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            raise ValueError(
                f"Client {i} received 0 samples. "
                "Try increasing alpha or reducing n_clients."
            )
        idx = np.array(indices)
        rng.shuffle(idx)
        partitions.append((X_arr[idx], y_arr[idx]))

    return partitions


def simulate_clients(
    X: NDArray,
    y: NDArray,
    n_clients: int,
    iid: bool = True,
    alpha: float = 0.5,
    random_state: _RNGSeed = None,
) -> list[tuple[NDArray, NDArray]]:
    """Split data into per-client partitions for federated simulation.

    Produces either an IID uniform split or a Dirichlet-based non-IID split
    depending on the ``iid`` flag.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels.
    n_clients : int
        Number of clients.
    iid : bool, default=True
        If ``True``, samples are split uniformly at random (IID). If
        ``False``, :func:`partition_noniid` is used with ``alpha``.
    alpha : float, default=0.5
        Dirichlet concentration parameter; only used when ``iid=False``.
    random_state : int, Generator, or None, default=None
        Seed or ``numpy.random.Generator`` instance for reproducibility.

    Returns
    -------
    partitions : list of (ndarray, ndarray) of length n_clients
        Each element is the ``(X_i, y_i)`` partition for client *i*.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, n_features=10, random_state=0)
    >>> parts = simulate_clients(X, y, n_clients=3, iid=True, random_state=0)
    >>> len(parts)
    3
    >>> sum(len(yi) for _, yi in parts)
    300
    """
    rng = np.random.default_rng(random_state)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if not iid:
        return partition_noniid(X_arr, y_arr, n_clients, alpha=alpha, random_state=rng)

    indices = np.arange(len(y_arr))
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    return [(X_arr[idx], y_arr[idx]) for idx in splits]
