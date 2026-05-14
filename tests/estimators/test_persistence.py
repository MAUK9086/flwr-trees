from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from flwr_trees.estimators.gbt import FederatedGBTClassifier
from flwr_trees.estimators.hist_rf import FederatedHistogramRFClassifier
from flwr_trees.estimators.rf import FederatedRandomForestClassifier, FederatedRandomForestRegressor
from flwr_trees.estimators.xgb import FederatedXGBClassifier

_X_CLF, _Y_CLF = make_classification(n_samples=200, n_features=10, random_state=0)
_X_REG, _Y_REG = make_regression(n_samples=200, n_features=10, random_state=0)


def test_rf_classifier_picklable() -> None:
    clf = FederatedRandomForestClassifier(n_estimators=5, n_clients=2, random_state=0)
    clf.fit(_X_CLF, _Y_CLF)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(_X_CLF), clf.predict(_X_CLF))


def test_rf_regressor_picklable() -> None:
    reg = FederatedRandomForestRegressor(n_estimators=5, n_clients=2, random_state=0)
    reg.fit(_X_REG, _Y_REG)
    restored = pickle.loads(pickle.dumps(reg))
    np.testing.assert_array_almost_equal(restored.predict(_X_REG), reg.predict(_X_REG))


def test_xgb_classifier_picklable() -> None:
    clf = FederatedXGBClassifier(n_estimators=5, n_clients=2, random_state=0)
    clf.fit(_X_CLF, _Y_CLF)
    restored = pickle.loads(pickle.dumps(clf))
    assert hasattr(restored, "booster_")
    np.testing.assert_array_equal(restored.predict(_X_CLF), clf.predict(_X_CLF))


def test_gbt_classifier_picklable() -> None:
    clf = FederatedGBTClassifier(n_estimators=5, n_clients=2, random_state=0)
    clf.fit(_X_CLF, _Y_CLF)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(_X_CLF), clf.predict(_X_CLF))


def test_histogram_rf_picklable() -> None:
    clf = FederatedHistogramRFClassifier(
        n_estimators=5, n_clients=3, n_rounds=1, n_bins=8, use_flower=True, random_state=42
    )
    clf.fit(_X_CLF, _Y_CLF)
    data = pickle.dumps(clf)
    restored = pickle.loads(data)
    assert hasattr(restored, "estimators_")
    assert hasattr(restored, "thresholds_")
    assert not hasattr(restored, "strategy_")


def test_pickle_predict_matches_original() -> None:
    clf = FederatedRandomForestClassifier(n_estimators=5, n_clients=2, random_state=0)
    clf.fit(_X_CLF, _Y_CLF)
    original_preds = clf.predict(_X_CLF)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(_X_CLF), original_preds)
