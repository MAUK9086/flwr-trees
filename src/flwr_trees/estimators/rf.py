from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from flwr_trees.estimators.base import BaseFederatedTreeEstimator
from flwr_trees.simulation.partitioning import simulate_clients

logger = logging.getLogger(__name__)


class FederatedRandomForestClassifier(ClassifierMixin, BaseFederatedTreeEstimator):
    """Federated Random Forest classifier using local simulation.

    Partitions training data among ``n_clients`` simulated clients, trains
    one :class:`~sklearn.ensemble.RandomForestClassifier` per client, then
    bags all resulting decision trees into a single ensemble stored in
    ``self.estimators_``.  Prediction is done by averaging class
    probabilities across all trees (consistent with sklearn's RF).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees *per client*.  Total trees = ``n_estimators * n_clients``.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Number of federated training rounds.
    iid : bool, default=True
        If ``True``, data is partitioned uniformly at random (IID).  If
        ``False``, uses Dirichlet-based non-IID partitioning.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int or None, default=None
        Maximum tree depth passed to each client's RF.
    max_features : str, int, or float, default="sqrt"
        Number of features to consider at each split.
    min_samples_split : int or float, default=2
        Minimum samples to split an internal node.
    min_samples_leaf : int or float, default=1
        Minimum samples required to be at a leaf node.
    use_flower : bool, default=False
        If ``False``, run the plain local training loop (fast; used by
        ``check_estimator`` and unit tests).  If ``True``, drive the FL loop
        via Flower ``Strategy`` / ``NumPyClient`` types using an in-process
        orchestrator (no Ray required).  Sets ``self.strategy_`` after fit.
    random_state : int or None, default=None
        Seed for data partitioning and each client RF.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        All decision trees collected from every client RF after ``fit()``.
    classes_ : ndarray of shape (n_classes,)
        Global class labels seen during ``fit()``.
    n_features_in_ : int
        Number of features seen during ``fit()``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_clients: int = 5,
        n_rounds: int = 1,
        iid: bool = True,
        alpha: float = 0.5,
        max_depth: int | None = None,
        max_features: Union[str, int, float] = "sqrt",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        use_flower: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_clients=n_clients,
            n_rounds=n_rounds,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.iid = iid
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_flower = use_flower

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedRandomForestClassifier:
        """Fit the federated random forest on partitioned client data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : FederatedRandomForestClassifier
            Fitted estimator.
        """
        X_np, y_np = self._validate_data_federated(X, y, reset=True)
        check_classification_targets(y_np)

        # Encode labels to contiguous integers so that each client RF's
        # tree.classes_ is a predictable integer array regardless of the
        # original y dtype (int, float, string, …).  self.classes_ retains
        # the original labels for decode-on-predict.
        le = LabelEncoder()
        le.fit_transform(y_np)  # fit le; y_enc not needed here
        self.classes_ = le.classes_

        n_samples = X_np.shape[0]
        effective_clients = max(1, min(self.n_clients, n_samples))

        rng = np.random.default_rng(self.random_state)
        partitions = simulate_clients(
            X_np,
            y_np,
            n_clients=effective_clients,
            iid=self.iid,
            alpha=self.alpha,
            random_state=rng,
        )

        if self.use_flower:
            self._run_flower_fit(partitions, le, rng)
        else:
            self._run_local_fit(partitions, le, rng)

        return self

    def _run_local_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        le: LabelEncoder,
        rng: np.random.Generator,
    ) -> None:
        """Train one RF per client partition and bag all trees locally."""
        self.estimators_: list[DecisionTreeClassifier] = []
        for client_idx, (X_i, y_i_orig) in enumerate(partitions):
            y_i = le.transform(y_i_orig)
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=client_seed,
            )
            rf.fit(X_i, y_i)
            self.estimators_.extend(rf.estimators_)
            logger.debug(
                "Client %d: trained %d trees on %d samples",
                client_idx,
                len(rf.estimators_),
                len(y_i),
            )

    def _run_flower_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        le: LabelEncoder,
        rng: np.random.Generator,
    ) -> None:
        """Drive the FL loop using Flower Strategy + NumPyClient types.

        Builds one ``FedForestBaggingClient`` and ``_LocalClientProxy`` per
        partition, then runs ``self.n_rounds`` rounds by directly calling
        ``proxy.fit()`` and ``strategy.aggregate_fit()``.  No Ray or network
        I/O is required; all execution is in-process.

        After completion sets ``self.estimators_`` and ``self.strategy_``.

        Parameters
        ----------
        partitions : list of (NDArray, NDArray)
            Per-client (X_i, y_i_orig) pairs with original labels.
        le : LabelEncoder
            Fitted encoder used to transform partition labels to integers.
        rng : np.random.Generator
            RNG for reproducible per-client seeds.
        """
        from flwr.common import FitIns, ndarrays_to_parameters

        from flwr_trees.aggregation.bagging import (
            FedForestBagging,
            FedForestBaggingClient,
            _LocalClientProxy,
        )

        strategy = FedForestBagging()

        rf_params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

        proxies: list[_LocalClientProxy] = []
        for client_idx, (X_i, y_i_orig) in enumerate(partitions):
            y_i = le.transform(y_i_orig)
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            client = FedForestBaggingClient(
                X_i=X_i,
                y_i=y_i,
                rf_params={**rf_params, "random_state": client_seed},
            )
            proxies.append(_LocalClientProxy(cid=str(client_idx), client=client))

        empty_params = ndarrays_to_parameters([])

        for round_idx in range(1, self.n_rounds + 1):
            fit_ins = FitIns(
                parameters=empty_params,
                config={"round": round_idx},
            )
            results = [
                (proxy, proxy.fit(fit_ins, timeout=None, group_id=None))
                for proxy in proxies
            ]
            strategy.aggregate_fit(
                server_round=round_idx,
                results=results,
                failures=[],
            )
            logger.info(
                "Flower round %d/%d: %d total trees accumulated",
                round_idx,
                self.n_rounds,
                len(strategy.trees_),
            )

        self.estimators_: list[DecisionTreeClassifier] = strategy.trees_
        self.strategy_ = strategy

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities averaged across all client trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Averaged probability estimates.
        """
        check_is_fitted(self)
        X_np = self._validate_data_federated(X, reset=False)
        n_samples = X_np.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=np.float64)

        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X_np)
            # tree.classes_ are encoded integers (0..n_classes-1), so they
            # directly index into the global proba array.
            for j, cls_idx in enumerate(tree.classes_):
                proba[:, int(cls_idx)] += tree_proba[:, j]

        proba /= len(self.estimators_)
        return proba

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels as argmax of averaged probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class FederatedRandomForestRegressor(RegressorMixin, BaseFederatedTreeEstimator):
    """Federated Random Forest regressor using local simulation.

    Partitions training data among ``n_clients`` simulated clients, trains
    one :class:`~sklearn.ensemble.RandomForestRegressor` per client, then
    bags all resulting decision trees into a single ensemble stored in
    ``self.estimators_``.  Prediction is the mean across all trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees *per client*.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Number of federated training rounds.
    iid : bool, default=True
        IID data partitioning when ``True``; Dirichlet non-IID when ``False``.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int or None, default=None
        Maximum tree depth.
    max_features : str, int, or float, default=1.0
        Features to consider at each split (sklearn default for regressors).
    min_samples_split : int or float, default=2
        Minimum samples to split an internal node.
    min_samples_leaf : int or float, default=1
        Minimum samples at a leaf node.
    random_state : int or None, default=None
        Seed for data partitioning and each client RF.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        All decision trees collected from every client RF after ``fit()``.
    n_features_in_ : int
        Number of features seen during ``fit()``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_clients: int = 5,
        n_rounds: int = 1,
        iid: bool = True,
        alpha: float = 0.5,
        max_depth: int | None = None,
        max_features: Union[str, int, float] = 1.0,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_clients=n_clients,
            n_rounds=n_rounds,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.iid = iid
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedRandomForestRegressor:
        """Fit the federated random forest regressor on partitioned client data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FederatedRandomForestRegressor
            Fitted estimator.
        """
        X_np, y_np = self._validate_data_federated(X, y, reset=True)

        n_samples = X_np.shape[0]
        effective_clients = max(1, min(self.n_clients, n_samples))

        rng = np.random.default_rng(self.random_state)
        partitions = simulate_clients(
            X_np,
            y_np,
            n_clients=effective_clients,
            iid=self.iid,
            alpha=self.alpha,
            random_state=rng,
        )

        self.estimators_: list[DecisionTreeRegressor] = []
        for client_idx, (X_i, y_i) in enumerate(partitions):
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=client_seed,
            )
            rf.fit(X_i, y_i)
            self.estimators_.extend(rf.estimators_)
            logger.debug(
                "Client %d: trained %d trees on %d samples",
                client_idx,
                len(rf.estimators_),
                len(y_i),
            )

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict regression targets as the mean across all client trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self)
        X_np = self._validate_data_federated(X, reset=False)
        all_preds = np.array(
            [tree.predict(X_np) for tree in self.estimators_], dtype=np.float64
        )
        return np.mean(all_preds, axis=0)
