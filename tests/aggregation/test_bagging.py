from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from flwr_trees.estimators.rf import FederatedRandomForestClassifier


def _make_data(
    n_samples: int = 500,
    n_features: int = 10,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


_FLOWER_KWARGS: dict = dict(
    n_clients=5,
    n_rounds=3,
    n_estimators=10,
    use_flower=True,
    random_state=42,
)


def test_flower_path_accuracy_above_threshold() -> None:
    X_train, X_test, y_train, y_test = _make_data()
    clf = FederatedRandomForestClassifier(**_FLOWER_KWARGS)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    assert acc > 0.7, f"Expected accuracy > 0.7, got {acc:.4f}"


def test_flower_path_bytes_sent_per_round_length() -> None:
    X_train, X_test, y_train, y_test = _make_data()
    clf = FederatedRandomForestClassifier(**_FLOWER_KWARGS)
    clf.fit(X_train, y_train)
    assert len(clf.strategy_.bytes_sent_per_round) == 3


def test_flower_path_bytes_sent_per_round_positive() -> None:
    X_train, X_test, y_train, y_test = _make_data()
    clf = FederatedRandomForestClassifier(**_FLOWER_KWARGS)
    clf.fit(X_train, y_train)
    for b in clf.strategy_.bytes_sent_per_round:
        assert isinstance(b, int)
        assert b > 0


def test_flower_path_strategy_attribute_set() -> None:
    from flwr_trees.aggregation.bagging import FedForestBagging

    X_train, X_test, y_train, y_test = _make_data()
    clf = FederatedRandomForestClassifier(**_FLOWER_KWARGS)
    clf.fit(X_train, y_train)
    assert hasattr(clf, "strategy_")
    assert isinstance(clf.strategy_, FedForestBagging)


def test_local_path_unaffected_by_use_flower_param() -> None:
    X_train, X_test, y_train, y_test = _make_data()
    clf = FederatedRandomForestClassifier(
        n_clients=5,
        n_rounds=1,
        n_estimators=10,
        use_flower=False,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    assert len(clf.estimators_) == 5 * 10
    assert not hasattr(clf, "strategy_")
