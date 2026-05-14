from __future__ import annotations

import pytest
from sklearn.datasets import make_classification

from flwr_trees.estimators.xgb import FederatedXGBClassifier, _xgb_params


def _cuda_available() -> bool:
    try:
        import xgboost as xgb
        import numpy as np
        X = np.random.default_rng(0).random((10, 2)).astype(np.float32)
        y = np.zeros(10, dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y)
        xgb.train({"device": "cuda", "verbosity": 0}, dtrain, num_boost_round=1)
        return True
    except Exception:
        return False


def test_device_cpu_default() -> None:
    clf = FederatedXGBClassifier()
    assert clf.device == "cpu"


def test_device_param_stored() -> None:
    clf = FederatedXGBClassifier(device="cuda")
    assert clf.device == "cuda"


def test_device_passed_to_xgb_params() -> None:
    params = _xgb_params(
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        seed=0,
        objective="binary:logistic",
        device="cuda",
    )
    assert params["device"] == "cuda"


def test_device_cpu_not_in_params_by_default() -> None:
    params = _xgb_params(
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        seed=0,
        objective="binary:logistic",
    )
    assert "device" not in params


def test_device_cpu_trains_normally() -> None:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    clf = FederatedXGBClassifier(n_estimators=5, n_clients=2, device="cpu", random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_cuda_trains_normally() -> None:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    clf = FederatedXGBClassifier(n_estimators=5, n_clients=2, device="cuda", random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
