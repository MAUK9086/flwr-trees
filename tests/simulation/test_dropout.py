from __future__ import annotations

import numpy as np
import pytest

from flwr_trees.simulation.dropout import ClientDropoutWrapper


def _make_partitions(n: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.ones((5, 3)), np.ones(5)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


def test_sample_returns_at_least_min_clients() -> None:
    parts = _make_partitions(10)
    wrapper = ClientDropoutWrapper(parts, dropout_rate=0.99, min_clients=2, random_state=0)
    for round_idx in range(20):
        surviving = wrapper.sample(round_idx)
        assert len(surviving) >= 2


def test_sample_never_exceeds_total_partitions() -> None:
    parts = _make_partitions(10)
    wrapper = ClientDropoutWrapper(parts, dropout_rate=0.0, random_state=1)
    assert len(wrapper.sample(0)) == 10


def test_sample_full_return_when_rate_zero() -> None:
    parts = _make_partitions(8)
    wrapper = ClientDropoutWrapper(parts, dropout_rate=0.0, random_state=42)
    for round_idx in range(5):
        assert len(wrapper.sample(round_idx)) == 8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_sample_reproducible_with_same_round_idx() -> None:
    parts = _make_partitions(10)
    w1 = ClientDropoutWrapper(parts, dropout_rate=0.5, random_state=7)
    w2 = ClientDropoutWrapper(parts, dropout_rate=0.5, random_state=7)
    assert len(w1.sample(3)) == len(w2.sample(3))


def test_sample_varies_across_rounds() -> None:
    parts = _make_partitions(10)
    wrapper = ClientDropoutWrapper(parts, dropout_rate=0.4, random_state=99)
    counts = {wrapper.sample(r).__len__() for r in range(10)}
    # With dropout_rate=0.4 and 10 clients, sizes should vary across rounds
    assert len(counts) > 1


def test_sample_no_reproducibility_when_random_state_none() -> None:
    parts = _make_partitions(10)
    w1 = ClientDropoutWrapper(parts, dropout_rate=0.5, random_state=None)
    w2 = ClientDropoutWrapper(parts, dropout_rate=0.5, random_state=None)
    # With None seed, just verify the call succeeds (can't assert values)
    assert len(w1.sample(0)) >= 2
    assert len(w2.sample(0)) >= 2


# ---------------------------------------------------------------------------
# Statistical dropout rate
# ---------------------------------------------------------------------------


def test_sample_approximately_correct_dropout_rate() -> None:
    n_partitions = 20
    n_rounds = 200
    dropout_rate = 0.3
    parts = _make_partitions(n_partitions)
    wrapper = ClientDropoutWrapper(
        parts, dropout_rate=dropout_rate, min_clients=1, random_state=0
    )
    total_survivors = sum(len(wrapper.sample(r)) for r in range(n_rounds))
    mean_survivors = total_survivors / n_rounds
    expected = n_partitions * (1 - dropout_rate)
    # Allow 10% relative tolerance
    assert abs(mean_survivors - expected) / expected < 0.1, (
        f"Mean survivors {mean_survivors:.2f} far from expected {expected:.2f}"
    )


# ---------------------------------------------------------------------------
# Min-clients fill-in
# ---------------------------------------------------------------------------


def test_min_clients_forces_minimum_even_with_max_dropout() -> None:
    parts = _make_partitions(5)
    min_clients = 3
    wrapper = ClientDropoutWrapper(
        parts, dropout_rate=1.0, min_clients=min_clients, random_state=0
    )
    # dropout_rate=1.0 means all clients should drop → filled back up to min_clients
    surviving = wrapper.sample(0)
    assert len(surviving) == min_clients
