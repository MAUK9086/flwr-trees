from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from flwr_trees.privacy.dp import DPTreeWrapper, NoisyHistogram


def _fitted_tree() -> tuple[DecisionTreeClassifier, np.ndarray]:
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)
    return tree, X


# ---------------------------------------------------------------------------
# NoisyHistogram
# ---------------------------------------------------------------------------


def test_noisy_histogram_same_shape() -> None:
    counts = np.array([5.0, 10.0, 3.0, 8.0])
    noisy = NoisyHistogram(epsilon=1.0, random_state=0).apply(counts)
    assert noisy.shape == counts.shape


def test_noisy_histogram_counts_nonnegative() -> None:
    counts = np.array([0.0, 1.0, 2.0, 50.0, 100.0])
    noisy = NoisyHistogram(epsilon=0.1, random_state=42).apply(counts)
    assert (noisy >= 0).all()


def test_noisy_histogram_different_seeds_give_different_noise() -> None:
    counts = np.ones(20) * 50.0
    n1 = NoisyHistogram(epsilon=1.0, random_state=0).apply(counts)
    n2 = NoisyHistogram(epsilon=1.0, random_state=1).apply(counts)
    assert not np.allclose(n1, n2)


def test_noisy_histogram_same_seed_reproducible() -> None:
    counts = np.arange(10, dtype=float)
    n1 = NoisyHistogram(epsilon=2.0, random_state=99).apply(counts)
    n2 = NoisyHistogram(epsilon=2.0, random_state=99).apply(counts)
    np.testing.assert_array_equal(n1, n2)


def test_noisy_histogram_higher_epsilon_less_noise() -> None:
    counts = np.ones(1000) * 100.0
    low_eps = NoisyHistogram(epsilon=0.1, random_state=0).apply(counts)
    high_eps = NoisyHistogram(epsilon=10.0, random_state=0).apply(counts)
    assert low_eps.std() > high_eps.std()


# ---------------------------------------------------------------------------
# DPTreeWrapper
# ---------------------------------------------------------------------------


def test_dp_tree_wrapper_predict_proba_shape() -> None:
    tree, X = _fitted_tree()
    wrapper = DPTreeWrapper(tree, epsilon=1.0, random_state=0)
    proba = wrapper.predict_proba(X)
    assert proba.shape == (len(X), len(tree.classes_))


def test_dp_tree_wrapper_predict_proba_sums_to_one() -> None:
    tree, X = _fitted_tree()
    wrapper = DPTreeWrapper(tree, epsilon=1.0, random_state=0)
    proba = wrapper.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_dp_tree_wrapper_predict_proba_nonnegative() -> None:
    tree, X = _fitted_tree()
    wrapper = DPTreeWrapper(tree, epsilon=1.0, random_state=0)
    proba = wrapper.predict_proba(X)
    assert (proba >= 0).all()


def test_dp_tree_wrapper_predict_returns_valid_labels() -> None:
    tree, X = _fitted_tree()
    wrapper = DPTreeWrapper(tree, epsilon=1.0, random_state=0)
    preds = wrapper.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset(set(range(len(tree.classes_))))


def test_dp_tree_wrapper_different_seeds_give_different_results() -> None:
    tree, X = _fitted_tree()
    p1 = DPTreeWrapper(tree, epsilon=1.0, random_state=0).predict_proba(X)
    p2 = DPTreeWrapper(tree, epsilon=1.0, random_state=1).predict_proba(X)
    assert not np.allclose(p1, p2)


def test_dp_tree_wrapper_same_seed_reproducible() -> None:
    tree, X = _fitted_tree()
    p1 = DPTreeWrapper(tree, epsilon=2.0, random_state=7).predict_proba(X)
    p2 = DPTreeWrapper(tree, epsilon=2.0, random_state=7).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)
