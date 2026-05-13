from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from flwr_trees.simulation.partitioning import partition_noniid, simulate_clients


def _make_data(
    n_samples: int = 300,
    n_features: int = 10,
    n_classes: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=5,
        n_redundant=2,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# simulate_clients — IID
# ---------------------------------------------------------------------------


def test_simulate_clients_iid_returns_correct_number_of_partitions() -> None:
    X, y = _make_data()
    parts = simulate_clients(X, y, n_clients=3, iid=True, random_state=0)
    assert len(parts) == 3


def test_simulate_clients_iid_covers_all_samples() -> None:
    X, y = _make_data(n_samples=300)
    parts = simulate_clients(X, y, n_clients=3, iid=True, random_state=0)
    total = sum(len(yi) for _, yi in parts)
    assert total == 300


def test_simulate_clients_iid_shapes_consistent() -> None:
    X, y = _make_data(n_samples=300)
    parts = simulate_clients(X, y, n_clients=3, iid=True, random_state=0)
    for Xi, yi in parts:
        assert Xi.ndim == 2
        assert yi.ndim == 1
        assert len(Xi) == len(yi)
        assert Xi.shape[1] == X.shape[1]


def test_simulate_clients_iid_no_overlap() -> None:
    X, y = _make_data(n_samples=300)
    parts = simulate_clients(X, y, n_clients=3, iid=True, random_state=0)
    all_indices: list[int] = []
    for Xi, _ in parts:
        for row in Xi:
            matches = np.where((X == row).all(axis=1))[0]
            all_indices.extend(matches.tolist())
    assert len(all_indices) == len(set(all_indices)), "IID split has overlapping samples"


def test_simulate_clients_iid_reproducible() -> None:
    X, y = _make_data()
    p1 = simulate_clients(X, y, n_clients=3, iid=True, random_state=7)
    p2 = simulate_clients(X, y, n_clients=3, iid=True, random_state=7)
    for (_, y1), (_, y2) in zip(p1, p2):
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# simulate_clients — non-IID
# ---------------------------------------------------------------------------


def test_simulate_clients_noniid_returns_correct_number_of_partitions() -> None:
    X, y = _make_data()
    parts = simulate_clients(X, y, n_clients=3, iid=False, alpha=0.5, random_state=0)
    assert len(parts) == 3


def test_simulate_clients_noniid_covers_all_samples() -> None:
    X, y = _make_data(n_samples=300)
    parts = simulate_clients(X, y, n_clients=3, iid=False, alpha=0.5, random_state=0)
    total = sum(len(yi) for _, yi in parts)
    assert total == 300


def test_simulate_clients_noniid_shapes_consistent() -> None:
    X, y = _make_data(n_samples=300)
    parts = simulate_clients(X, y, n_clients=3, iid=False, alpha=0.5, random_state=0)
    for Xi, yi in parts:
        assert Xi.ndim == 2
        assert yi.ndim == 1
        assert len(Xi) == len(yi)


# ---------------------------------------------------------------------------
# partition_noniid
# ---------------------------------------------------------------------------


def test_partition_noniid_correct_count() -> None:
    X, y = _make_data()
    parts = partition_noniid(X, y, n_clients=5, alpha=1.0, random_state=0)
    assert len(parts) == 5


def test_partition_noniid_covers_all_samples() -> None:
    X, y = _make_data(n_samples=300)
    parts = partition_noniid(X, y, n_clients=3, alpha=0.5, random_state=0)
    total = sum(len(yi) for _, yi in parts)
    assert total == 300


def test_partition_noniid_reproducible() -> None:
    X, y = _make_data()
    p1 = partition_noniid(X, y, n_clients=3, alpha=0.5, random_state=42)
    p2 = partition_noniid(X, y, n_clients=3, alpha=0.5, random_state=42)
    for (_, y1), (_, y2) in zip(p1, p2):
        np.testing.assert_array_equal(y1, y2)


def test_partition_noniid_high_alpha_more_uniform_than_low() -> None:
    """High alpha => more uniform class distribution within each client."""
    X, y = _make_data(n_samples=600, n_classes=3)

    def mean_class_std(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
        stds = []
        for _, yi in parts:
            _, counts = np.unique(yi, return_counts=True)
            stds.append(counts.std())
        return float(np.mean(stds))

    high = partition_noniid(X, y, n_clients=3, alpha=10.0, random_state=0)
    low = partition_noniid(X, y, n_clients=3, alpha=0.1, random_state=0)
    assert mean_class_std(high) <= mean_class_std(low)


def test_partition_noniid_raises_on_zero_samples() -> None:
    X, y = _make_data(n_samples=30, n_classes=2)
    with pytest.raises(ValueError, match="0 samples"):
        partition_noniid(X, y, n_clients=200, alpha=0.001, random_state=0)


def test_partition_noniid_binary_classification() -> None:
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    parts = partition_noniid(X, y, n_clients=4, alpha=0.5, random_state=1)
    assert len(parts) == 4
    assert sum(len(yi) for _, yi in parts) == 200


def test_simulate_clients_passes_alpha_to_noniid() -> None:
    X, y = _make_data(n_samples=600)
    parts = simulate_clients(X, y, n_clients=3, iid=False, alpha=5.0, random_state=99)
    assert len(parts) == 3
    assert sum(len(yi) for _, yi in parts) == 600
