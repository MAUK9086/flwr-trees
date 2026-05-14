from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import check_estimator

from flwr_trees.estimators.gbt import FederatedGBTClassifier, FederatedGBTRegressor

_CLF_KWARGS: dict = dict(n_estimators=5, n_clients=3, random_state=42)
_REG_KWARGS: dict = dict(n_estimators=5, n_clients=3, random_state=42)


def _clf_data(
    n_samples: int = 150,
    n_classes: int = 3,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n_samples,
        n_features=10,
        n_classes=n_classes,
        n_informative=5,
        n_redundant=2,
        random_state=random_state,
    )


def _reg_data(
    n_samples: int = 150,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    return make_regression(
        n_samples=n_samples,
        n_features=10,
        noise=0.1,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# FederatedGBTClassifier — basic API
# ---------------------------------------------------------------------------


def test_classifier_fit_returns_self() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    assert clf.fit(X, y) is clf


def test_classifier_sets_classes_after_fit() -> None:
    X, y = _clf_data(n_classes=3)
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    np.testing.assert_array_equal(np.sort(clf.classes_), np.unique(y))


def test_classifier_sets_n_features_in_after_fit() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert clf.n_features_in_ == X.shape[1]


def test_classifier_sets_estimators_after_fit() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    assert hasattr(clf, "estimators_")
    # one full GBT per active client
    assert len(clf.estimators_) == 3
    from sklearn.ensemble import GradientBoostingClassifier
    assert all(isinstance(m, GradientBoostingClassifier) for m in clf.estimators_)


def test_classifier_predict_shape() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)


def test_classifier_predict_labels_in_classes() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset(set(clf.classes_))


def test_classifier_predict_before_fit_raises() -> None:
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    X, _ = _clf_data()
    with pytest.raises(Exception):
        clf.predict(X)


def test_classifier_predict_proba_shape() -> None:
    X, y = _clf_data(n_classes=3)
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), len(clf.classes_))


def test_classifier_predict_proba_sums_to_one() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)), atol=1e-6)


def test_classifier_predict_consistent_with_proba() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    expected = clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(clf.predict(X), expected)


def test_classifier_score_above_chance() -> None:
    X, y = _clf_data(n_classes=2)
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    score = clf.score(X, y)
    majority_baseline = np.bincount(y).max() / len(y)
    assert score > majority_baseline


def test_classifier_binary_works() -> None:
    X, y = _clf_data(n_classes=2)
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classifier_noniid_partition() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(
        n_estimators=5, n_clients=3, iid=False, alpha=1.0, random_state=7
    )
    clf.fit(X, y)
    assert hasattr(clf, "estimators_")
    assert len(clf.estimators_) > 0


def test_classifier_does_not_mutate_input() -> None:
    X, y = _clf_data()
    X_copy, y_copy = X.copy(), y.copy()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_classifier_n_features_mismatch_raises() -> None:
    X, y = _clf_data()
    clf = FederatedGBTClassifier(**_CLF_KWARGS)
    clf.fit(X, y)
    with pytest.raises(Exception):
        clf.predict(X[:, :-1])


def test_classifier_get_params_round_trips() -> None:
    clf = FederatedGBTClassifier(n_estimators=7, n_clients=4, random_state=1)
    params = clf.get_params()
    assert params["n_estimators"] == 7
    assert params["n_clients"] == 4
    assert params["random_state"] == 1


def test_classifier_set_params_works() -> None:
    clf = FederatedGBTClassifier()
    clf.set_params(n_estimators=3, n_clients=2)
    assert clf.n_estimators == 3
    assert clf.n_clients == 2


def test_classifier_reproducible_with_random_state() -> None:
    X, y = _clf_data()
    clf1 = FederatedGBTClassifier(n_estimators=5, n_clients=3, random_state=0)
    clf2 = FederatedGBTClassifier(n_estimators=5, n_clients=3, random_state=0)
    clf1.fit(X, y)
    clf2.fit(X, y)
    np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


def test_classifier_handles_n_clients_larger_than_n_samples() -> None:
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    clf = FederatedGBTClassifier(n_estimators=2, n_clients=100, random_state=0)
    clf.fit(X, y)
    assert hasattr(clf, "estimators_")


def test_classifier_check_estimator() -> None:
    check_estimator(FederatedGBTClassifier(n_estimators=5))


# ---------------------------------------------------------------------------
# FederatedGBTRegressor — basic API
# ---------------------------------------------------------------------------


def test_regressor_fit_returns_self() -> None:
    X, y = _reg_data()
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    assert reg.fit(X, y) is reg


def test_regressor_sets_n_features_in_after_fit() -> None:
    X, y = _reg_data()
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    assert reg.n_features_in_ == X.shape[1]


def test_regressor_sets_estimators_after_fit() -> None:
    X, y = _reg_data()
    reg = FederatedGBTRegressor(n_estimators=5, n_clients=3, random_state=0)
    reg.fit(X, y)
    from sklearn.ensemble import GradientBoostingRegressor
    assert len(reg.estimators_) == 3
    assert all(isinstance(m, GradientBoostingRegressor) for m in reg.estimators_)


def test_regressor_predict_shape() -> None:
    X, y = _reg_data()
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (len(y),)


def test_regressor_predict_before_fit_raises() -> None:
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    X, _ = _reg_data()
    with pytest.raises(Exception):
        reg.predict(X)


def test_regressor_score_better_than_constant() -> None:
    X, y = _reg_data()
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.0


def test_regressor_does_not_mutate_input() -> None:
    X, y = _reg_data()
    X_copy, y_copy = X.copy(), y.copy()
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(y, y_copy)


def test_regressor_n_features_mismatch_raises() -> None:
    X, y = _reg_data()
    reg = FederatedGBTRegressor(**_REG_KWARGS)
    reg.fit(X, y)
    with pytest.raises(Exception):
        reg.predict(X[:, :-1])


def test_regressor_get_params_round_trips() -> None:
    reg = FederatedGBTRegressor(n_estimators=7, n_clients=4)
    params = reg.get_params()
    assert params["n_estimators"] == 7
    assert params["n_clients"] == 4


def test_regressor_reproducible_with_random_state() -> None:
    X, y = _reg_data()
    reg1 = FederatedGBTRegressor(n_estimators=5, n_clients=3, random_state=0)
    reg2 = FederatedGBTRegressor(n_estimators=5, n_clients=3, random_state=0)
    reg1.fit(X, y)
    reg2.fit(X, y)
    np.testing.assert_array_equal(reg1.predict(X), reg2.predict(X))


def test_regressor_check_estimator() -> None:
    # n_clients=1 ensures the single client sees all training data so
    # check_regressors_train's training-score assertion (R² > 0.5) passes.
    check_estimator(FederatedGBTRegressor(n_estimators=5, n_clients=1))
