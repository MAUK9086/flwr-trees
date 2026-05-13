from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import check_estimator

from flwr_trees.estimators.xgb import FederatedXGBClassifier, FederatedXGBRegressor

# Small values keep each test well under 30 seconds.
_CLF_KWARGS: dict = dict(n_estimators=5, n_clients=3, n_rounds=1, random_state=0)
_REG_KWARGS: dict = dict(n_estimators=5, n_clients=3, n_rounds=1, random_state=0)


def _clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_classes=3,
        n_informative=5,
        n_redundant=2,
        random_state=0,
    )
    return X, y


def _reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=150, n_features=10, random_state=0)
    return X, y


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


def test_classifier_fit_returns_self() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    result = clf.fit(X, y)
    assert result is clf


def test_classifier_sets_classes_after_fit() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert set(clf.classes_) == set(np.unique(y))


def test_classifier_sets_n_features_in_after_fit() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert clf.n_features_in_ == X.shape[1]


def test_classifier_sets_model_after_fit() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert hasattr(clf, "booster_")
    assert isinstance(clf.booster_, xgb.Booster)


def test_classifier_predict_shape() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)


def test_classifier_predict_labels_in_classes() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset(set(clf.classes_))


def test_classifier_predict_before_fit_raises() -> None:
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    X, _ = _clf_data()
    with pytest.raises(Exception):
        clf.predict(X)


def test_classifier_predict_proba_shape() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(clf.classes_))


def test_classifier_predict_proba_sums_to_one() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), atol=1e-5)


def test_classifier_predict_consistent_with_proba() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    preds_from_proba = clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(clf.predict(X), preds_from_proba)


def test_classifier_score_above_chance() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    # Chance level ≈ 1/3 for 3-class; expect at least 0.5
    assert clf.score(X, y) > 0.5


def test_classifier_binary_works() -> None:
    X, y = make_classification(n_samples=80, n_features=5, n_classes=2, random_state=1)
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert clf.predict_proba(X).shape == (80, 2)
    assert set(clf.predict(X)).issubset({0, 1})


def test_classifier_does_not_mutate_input() -> None:
    X, y = _clf_data()
    X_copy = X.copy()
    y_copy = y.copy()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_classifier_n_features_mismatch_raises() -> None:
    X, y = _clf_data()
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    with pytest.raises(Exception):
        clf.predict(X[:, :5])


def test_classifier_get_params_round_trips() -> None:
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    params = clf.get_params()
    clf2 = FederatedXGBClassifier(**params)
    assert clf2.get_params() == params


def test_classifier_set_params_works() -> None:
    clf = FederatedXGBClassifier(**_CLF_KWARGS)
    clf.set_params(max_depth=3, learning_rate=0.05)
    assert clf.max_depth == 3
    assert clf.learning_rate == 0.05


def test_classifier_check_estimator() -> None:
    check_estimator(FederatedXGBClassifier(n_estimators=5))


def test_classifier_flower_path() -> None:
    from flwr_trees.aggregation.cyclic import FedForestCyclic

    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=3,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    clf = FederatedXGBClassifier(
        n_estimators=5,
        n_clients=3,
        n_rounds=2,
        use_flower=True,
        random_state=42,
    )
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc > 0.6, f"Expected accuracy > 0.6, got {acc:.4f}"
    assert hasattr(clf, "strategy_")
    assert isinstance(clf.strategy_, FedForestCyclic)
    # n_clients=3, n_rounds=2 → 6 total cyclic steps
    assert len(clf.strategy_.bytes_sent_per_round) == 6
    assert all(isinstance(b, int) and b > 0 for b in clf.strategy_.bytes_sent_per_round)


# ---------------------------------------------------------------------------
# Regressor tests
# ---------------------------------------------------------------------------


def test_regressor_fit_returns_self() -> None:
    X, y = _reg_data()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    result = reg.fit(X, y)
    assert result is reg


def test_regressor_sets_n_features_in_after_fit() -> None:
    X, y = _reg_data()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    assert reg.n_features_in_ == X.shape[1]


def test_regressor_sets_model_after_fit() -> None:
    X, y = _reg_data()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    assert hasattr(reg, "booster_")
    assert isinstance(reg.booster_, xgb.Booster)


def test_regressor_predict_shape() -> None:
    X, y = _reg_data()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (X.shape[0],)


def test_regressor_predict_before_fit_raises() -> None:
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    X, _ = _reg_data()
    with pytest.raises(Exception):
        reg.predict(X)


def test_regressor_score_better_than_constant() -> None:
    X, y = _reg_data()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.0


def test_regressor_does_not_mutate_input() -> None:
    X, y = _reg_data()
    X_copy = X.copy()
    y_copy = y.copy()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_regressor_n_features_mismatch_raises() -> None:
    X, y = _reg_data()
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    with pytest.raises(Exception):
        reg.predict(X[:, :5])


def test_regressor_get_params_round_trips() -> None:
    reg = FederatedXGBRegressor(**_REG_KWARGS)
    params = reg.get_params()
    reg2 = FederatedXGBRegressor(**params)
    assert reg2.get_params() == params


def test_regressor_check_estimator() -> None:
    check_estimator(FederatedXGBRegressor(n_estimators=5))
