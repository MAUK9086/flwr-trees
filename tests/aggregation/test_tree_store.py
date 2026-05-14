from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from flwr_trees.aggregation._tree_store import DiskTreeStore
from flwr_trees.aggregation.bagging import FedForestBagging
from flwr_trees.estimators.rf import FederatedRandomForestClassifier


def _make_trees(n: int = 3) -> list[DecisionTreeClassifier]:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    trees = []
    for i in range(n):
        t = DecisionTreeClassifier(max_depth=2, random_state=i).fit(X, y)
        trees.append(t)
    return trees


def test_store_append_and_len() -> None:
    store = DiskTreeStore()
    trees = _make_trees(3)
    for t in trees:
        store.append(t)
    assert len(store) == 3
    store.close()


def test_store_extend_and_iter() -> None:
    store = DiskTreeStore()
    trees = _make_trees(4)
    store.extend(trees)
    loaded = list(store)
    assert len(loaded) == 4
    for orig, ld in zip(trees, loaded):
        assert ld.get_n_leaves() == orig.get_n_leaves()
    store.close()


def test_store_getitem() -> None:
    store = DiskTreeStore()
    trees = _make_trees(3)
    store.extend(trees)
    assert store[0].get_n_leaves() == trees[0].get_n_leaves()
    assert store[2].get_n_leaves() == trees[2].get_n_leaves()
    with pytest.raises(IndexError):
        _ = store[5]
    store.close()


def test_store_cleanup_on_close() -> None:
    store = DiskTreeStore()
    store.extend(_make_trees(2))
    tmp_dir = store._dir
    assert os.path.isdir(tmp_dir)
    store.close()
    assert not os.path.isdir(tmp_dir)


def test_store_does_not_load_all_on_iter(monkeypatch: pytest.MonkeyPatch) -> None:
    import joblib

    call_count = {"n": 0}
    original_load = joblib.load

    def counting_load(path: str) -> object:
        call_count["n"] += 1
        return original_load(path)

    store = DiskTreeStore()
    store.extend(_make_trees(10))

    monkeypatch.setattr(joblib, "load", counting_load)

    total = 0
    for _ in store:
        total += 1
    assert total == 10
    # Each tree loaded exactly once (lazy, not all upfront)
    assert call_count["n"] == 10
    store.close()


def test_bagging_strategy_uses_disk_store() -> None:
    from flwr_trees.aggregation._tree_store import DiskTreeStore
    strategy = FedForestBagging()
    assert isinstance(strategy.trees_, DiskTreeStore)


def test_large_ensemble_memory() -> None:
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = FederatedRandomForestClassifier(
        n_estimators=5, n_clients=10, n_rounds=1, use_flower=True, random_state=0
    )
    clf.fit(X_tr, y_tr)
    assert len(clf.estimators_) == 50
    # Predict still works via lazy iteration
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)
