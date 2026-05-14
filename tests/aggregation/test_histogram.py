from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

from sklearn.datasets import make_regression

from flwr_trees.aggregation.histogram import FedHistogramAggregation
from flwr_trees.estimators.hist_rf import FederatedHistogramRFClassifier, FederatedHistogramRFRegressor
from flwr_trees.estimators.rf import FederatedRandomForestClassifier

_N_ESTIMATORS = 5
_N_CLIENTS = 3
_N_BINS = 8
_HIST_KWARGS: dict = dict(
    n_estimators=_N_ESTIMATORS,
    n_clients=_N_CLIENTS,
    n_rounds=1,
    n_bins=_N_BINS,
    use_flower=True,
    random_state=42,
)
_BAGGING_KWARGS: dict = dict(
    n_estimators=_N_ESTIMATORS,
    n_clients=_N_CLIENTS,
    n_rounds=1,
    use_flower=True,
    random_state=42,
)


def _make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=2,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# Basic fitted-attribute tests
# ---------------------------------------------------------------------------


def test_histogram_fit_returns_self() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    assert clf.fit(X_tr, y_tr) is clf


def test_histogram_sets_classes_after_fit() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "classes_")
    assert set(clf.classes_) == set(np.unique(y_tr))


def test_histogram_sets_estimators_after_fit() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "estimators_")
    assert len(clf.estimators_) == _N_CLIENTS * _N_ESTIMATORS
    assert all(isinstance(t, DecisionTreeClassifier) for t in clf.estimators_)


def test_histogram_sets_thresholds_after_fit() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "thresholds_")
    assert len(clf.thresholds_) == X_tr.shape[1]
    for edges in clf.thresholds_:
        assert edges.shape == (_N_BINS + 1,)


def test_histogram_sets_strategy_attribute() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "strategy_")
    assert isinstance(clf.strategy_, FedHistogramAggregation)


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


def test_histogram_accuracy_above_chance() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) > 0.5


def test_histogram_accuracy_within_5pct_of_bagging() -> None:
    X_tr, X_te, y_tr, y_te = _make_data()

    clf_hist = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf_hist.fit(X_tr, y_tr)
    acc_hist = clf_hist.score(X_te, y_te)

    clf_bag = FederatedRandomForestClassifier(**_BAGGING_KWARGS)
    clf_bag.fit(X_tr, y_tr)
    acc_bag = clf_bag.score(X_te, y_te)

    assert abs(acc_hist - acc_bag) < 0.05, (
        f"Histogram accuracy {acc_hist:.4f} deviates more than 5% "
        f"from bagging accuracy {acc_bag:.4f}"
    )


# ---------------------------------------------------------------------------
# Communication-cost tests
# ---------------------------------------------------------------------------


def test_histogram_bytes_round1_smaller_than_bagging() -> None:
    X_tr, _, y_tr, _ = _make_data()

    clf_hist = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf_hist.fit(X_tr, y_tr)
    # Index [1] = histogram round (index [0] is now metadata/Round 0)
    histogram_bytes = clf_hist.strategy_.bytes_sent_per_round[1]

    clf_bag = FederatedRandomForestClassifier(**_BAGGING_KWARGS)
    clf_bag.fit(X_tr, y_tr)
    bagging_bytes = clf_bag.strategy_.bytes_sent_per_round[0]

    assert histogram_bytes < bagging_bytes, (
        f"Histogram round-1 bytes ({histogram_bytes}) should be less than "
        f"bagging round-1 bytes ({bagging_bytes})"
    )


def test_histogram_bytes_saved_vs_bagging_positive() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert len(clf.strategy_.bytes_saved_vs_bagging) >= 1
    assert all(b > 0 for b in clf.strategy_.bytes_saved_vs_bagging), (
        f"Expected positive savings, got {clf.strategy_.bytes_saved_vs_bagging}"
    )


def test_histogram_bytes_sent_per_round_has_three_entries() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    # Round 0 (metadata) + Round 1 (histogram) + Round 2 (trees) = 3 entries
    assert len(clf.strategy_.bytes_sent_per_round) == 3
    assert all(isinstance(b, int) and b > 0 for b in clf.strategy_.bytes_sent_per_round)


# ---------------------------------------------------------------------------
# sklearn compliance
# ---------------------------------------------------------------------------


def test_histogram_check_estimator() -> None:
    check_estimator(FederatedHistogramRFClassifier(n_estimators=5))


# ---------------------------------------------------------------------------
# Phase 7B: feature_ranges (Option C) and DP metadata round (Option A) tests
# ---------------------------------------------------------------------------


def test_option_c_uses_provided_ranges() -> None:
    X_tr, _, y_tr, _ = _make_data()
    feature_ranges = np.column_stack([X_tr.min(axis=0), X_tr.max(axis=0)])
    clf = FederatedHistogramRFClassifier(
        **{**_HIST_KWARGS, "feature_ranges": feature_ranges}
    )
    clf.fit(X_tr, y_tr)
    for f, edges in enumerate(clf.thresholds_):
        assert edges[0] >= feature_ranges[f, 0] - 1e-9
        assert edges[-1] <= feature_ranges[f, 1] + 1e-9


def test_option_a_runs_without_feature_ranges() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(**_HIST_KWARGS)
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "thresholds_")
    assert len(clf.thresholds_) == X_tr.shape[1]


def test_option_c_no_round_0() -> None:
    X_tr, _, y_tr, _ = _make_data()
    feature_ranges = np.column_stack([X_tr.min(axis=0), X_tr.max(axis=0)])
    clf = FederatedHistogramRFClassifier(
        n_estimators=_N_ESTIMATORS,
        n_clients=_N_CLIENTS,
        n_rounds=1,
        n_bins=_N_BINS,
        use_flower=True,
        feature_ranges=feature_ranges,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    # Option C: no metadata round → n_rounds + 1 entries (histogram + trees)
    assert len(clf.strategy_.bytes_sent_per_round) == 2


def test_option_a_has_round_0() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf = FederatedHistogramRFClassifier(
        n_estimators=_N_ESTIMATORS,
        n_clients=_N_CLIENTS,
        n_rounds=1,
        n_bins=_N_BINS,
        use_flower=True,
        feature_ranges=None,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    # Option A: metadata + histogram + trees = n_rounds + 2 entries
    assert len(clf.strategy_.bytes_sent_per_round) == 3


def test_metadata_epsilon_affects_edges() -> None:
    X_tr, _, y_tr, _ = _make_data()
    clf_low = FederatedHistogramRFClassifier(
        **{**_HIST_KWARGS, "metadata_epsilon": 0.1}
    )
    clf_high = FederatedHistogramRFClassifier(
        **{**_HIST_KWARGS, "metadata_epsilon": 100.0}
    )
    clf_low.fit(X_tr, y_tr)
    clf_high.fit(X_tr, y_tr)
    # Low epsilon adds more noise → different edges from high epsilon
    edges_low = np.concatenate(clf_low.thresholds_)
    edges_high = np.concatenate(clf_high.thresholds_)
    assert not np.allclose(edges_low, edges_high)


def test_histogram_regressor_flower_path() -> None:
    X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    reg = FederatedHistogramRFRegressor(
        n_estimators=5,
        n_clients=3,
        n_rounds=1,
        n_bins=8,
        use_flower=True,
        random_state=42,
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (200,)
    assert preds.dtype == np.float64
