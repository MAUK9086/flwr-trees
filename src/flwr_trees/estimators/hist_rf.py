from __future__ import annotations

import logging

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


def _apply_bins(X_np: NDArray, thresholds: list[NDArray]) -> NDArray:
    """Discretise features using pre-computed global bin edges.

    Parameters
    ----------
    X_np : ndarray of shape (n_samples, n_features)
        Raw feature matrix.
    thresholds : list of NDArray
        Global bin edges per feature, each of shape ``(n_bins + 1,)``.

    Returns
    -------
    X_binned : ndarray of shape (n_samples, n_features), dtype float64
        Integer bin indices clipped to ``[1, n_bins]`` so out-of-range
        test samples remain within the training grid.
    """
    return np.column_stack(
        [
            np.digitize(X_np[:, f], thresholds[f]).clip(1, len(thresholds[f]) - 1)
            for f in range(X_np.shape[1])
        ]
    ).astype(np.float64)


class FederatedHistogramRFClassifier(ClassifierMixin, BaseFederatedTreeEstimator):
    """Federated Random Forest classifier using histogram-based feature discretisation.

    Reduces FL communication cost compared to
    :class:`~flwr_trees.estimators.rf.FederatedRandomForestClassifier` by
    exchanging compact per-feature histograms (round 1) instead of full trees.
    The server aggregates histograms to derive global split thresholds, which
    all clients then use to discretise their features before training local
    Random Forests (round 2+).

    This is the primary novel contribution of the *flwr-trees* research paper.
    Communication savings are tracked in ``self.strategy_.bytes_saved_vs_bagging``
    when ``use_flower=True``.

    Parameters
    ----------
    n_estimators : int, default=100
        Trees *per client* per training round.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Number of tree-training rounds *after* the histogram exchange.
        Total Flower rounds = ``1 + n_rounds``.
    iid : bool, default=True
        IID data partitioning; Dirichlet non-IID when ``False``.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int or None, default=None
        Maximum tree depth.
    n_bins : int, default=32
        Number of histogram bins per feature.  More bins → finer
        discretisation → smaller communication savings.
    use_flower : bool, default=False
        If ``True``, drive the protocol via Flower Strategy / NumPyClient
        types; sets ``self.strategy_`` after fit.
    random_state : int or None, default=None
        Seed for data partitioning and per-client RF training.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        All fitted trees from all clients and all training rounds.
    classes_ : ndarray of shape (n_classes,)
        Original class labels seen during ``fit()``.
    thresholds_ : list of NDArray
        Global bin edges per feature used for discretisation at predict time.
    n_features_in_ : int
        Number of features seen during ``fit()``.
    strategy_ : FedHistogramAggregation
        The Flower strategy instance; set only when ``use_flower=True``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_clients: int = 5,
        n_rounds: int = 1,
        iid: bool = True,
        alpha: float = 0.5,
        max_depth: int | None = None,
        n_bins: int = 32,
        use_flower: bool = False,
        feature_ranges: np.ndarray | None = None,
        metadata_epsilon: float = 5.0,
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
        self.n_bins = n_bins
        self.use_flower = use_flower
        self.feature_ranges = feature_ranges
        self.metadata_epsilon = metadata_epsilon

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedHistogramRFClassifier:
        """Fit via federated histogram aggregation followed by bagging.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : FederatedHistogramRFClassifier
            Fitted estimator.
        """
        X_np, y_np = self._validate_data_federated(X, y, reset=True)
        check_classification_targets(y_np)

        le = LabelEncoder()
        le.fit(y_np)
        self.classes_ = le.classes_

        n_samples, n_features = X_np.shape
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

        if self.feature_ranges is not None:
            ranges = np.asarray(self.feature_ranges)
            global_edges: list[NDArray] | None = [
                np.linspace(ranges[f, 0], ranges[f, 1], self.n_bins + 1)
                for f in range(n_features)
            ]
        else:
            global_edges = None  # Option A: DP Round 0 will compute edges

        if self.use_flower:
            self._run_flower_fit(partitions, le, rng, global_edges, n_features)
        else:
            self._run_local_fit(partitions, le, rng, global_edges, n_features)

        return self

    def _run_local_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        le: LabelEncoder,
        rng: np.random.Generator,
        global_edges: list[NDArray] | None,
        n_features: int,
    ) -> None:
        """Train locally: aggregate histograms in memory, then train RF per partition."""
        if global_edges is None:
            # Option A: DP-noised range exchange across partitions
            rng_dp = np.random.default_rng(self.random_state)
            noise_scale = 1.0 / self.metadata_epsilon
            all_mins = np.array([[X_i[:, f].min() for f in range(n_features)] for X_i, _ in partitions])
            all_maxs = np.array([[X_i[:, f].max() for f in range(n_features)] for X_i, _ in partitions])
            noisy_mins = all_mins + rng_dp.laplace(0, noise_scale, all_mins.shape)
            noisy_maxs = all_maxs + rng_dp.laplace(0, noise_scale, all_maxs.shape)
            global_min = noisy_mins.min(axis=0)
            global_max = noisy_maxs.max(axis=0)
            global_max = np.maximum(global_max, global_min + 1e-6)
            global_edges = [
                np.linspace(global_min[f], global_max[f], self.n_bins + 1)
                for f in range(n_features)
            ]

        self.thresholds_ = global_edges
        self.estimators_: list[DecisionTreeClassifier] = []

        for round_idx in range(self.n_rounds):
            for client_idx, (X_i, y_i_orig) in enumerate(partitions):
                y_i = le.transform(y_i_orig)
                client_seed = (
                    None
                    if self.random_state is None
                    else int(rng.integers(0, 2**31))
                )
                X_binned = _apply_bins(X_i, global_edges)
                rf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=client_seed,
                )
                rf.fit(X_binned, y_i)
                self.estimators_.extend(rf.estimators_)
                logger.debug(
                    "Round %d, client %d: %d trees trained on %d samples",
                    round_idx + 1,
                    client_idx,
                    len(rf.estimators_),
                    len(y_i),
                )

    def _run_flower_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        le: LabelEncoder,
        rng: np.random.Generator,
        global_edges: list[NDArray] | None,
        n_features: int,
    ) -> None:
        """Drive the FL loop via Flower Strategy + NumPyClient types.

        Option C (feature_ranges provided): 2-round protocol (histogram + tree rounds).
        Option A (feature_ranges=None): 3-round protocol (metadata + histogram + tree rounds).
        """
        from flwr.common import FitIns, ndarrays_to_parameters

        from flwr_trees.aggregation.histogram import (
            FedHistogramAggregation,
            HistogramClient,
            _LocalHistogramProxy,
        )

        strategy = FedHistogramAggregation(n_bins=self.n_bins)

        proxies: list[_LocalHistogramProxy] = []
        for client_idx, (X_i, y_i_orig) in enumerate(partitions):
            y_i = le.transform(y_i_orig)
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            client = HistogramClient(
                X_i=X_i,
                y_i=y_i,
                n_bins=self.n_bins,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=client_seed,
                global_edges=global_edges,
                task="classify",
                metadata_epsilon=self.metadata_epsilon,
            )
            proxies.append(_LocalHistogramProxy(cid=str(client_idx), client=client))

        if global_edges is not None:
            # Option C: edges pre-agreed, skip Round 0
            edge_params = ndarrays_to_parameters(global_edges)
        else:
            # Option A: Round 0 — DP metadata exchange to determine global edges
            edge_params, _ = strategy.aggregate_fit(
                server_round=0,
                results=[
                    (
                        proxy,
                        proxy.fit(
                            FitIns(
                                parameters=ndarrays_to_parameters([]),
                                config={"phase": "metadata"},
                            ),
                            timeout=None,
                            group_id=None,
                        ),
                    )
                    for proxy in proxies
                ],
                failures=[],
            )
            # Update client global_edges with server-computed edges
            server_edges = strategy.global_edges_
            for proxy in proxies:
                proxy._client.global_edges = server_edges

        # Round 1: histogram exchange
        threshold_params, _ = strategy.aggregate_fit(
            server_round=1,
            results=[
                (
                    proxy,
                    proxy.fit(
                        FitIns(
                            parameters=edge_params,
                            config={"phase": "histogram"},
                        ),
                        timeout=None,
                        group_id=None,
                    ),
                )
                for proxy in proxies
            ],
            failures=[],
        )

        # Round 2+ (n_rounds cycles): always re-pass threshold_params from round 1
        for _ in range(self.n_rounds):
            strategy.aggregate_fit(
                server_round=2,
                results=[
                    (
                        proxy,
                        proxy.fit(
                            FitIns(
                                parameters=threshold_params,
                                config={"phase": "train"},
                            ),
                            timeout=None,
                            group_id=None,
                        ),
                    )
                    for proxy in proxies
                ],
                failures=[],
            )

        self.estimators_ = strategy.trees_  # type: ignore[assignment]
        self.thresholds_ = strategy.thresholds_
        self.strategy_ = strategy

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities on discretised features.

        Applies global feature discretisation (same thresholds used during
        training) before querying each tree, then averages probabilities with
        sparse-class accumulation to handle non-IID trees.

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
        X_binned = _apply_bins(X_np, self.thresholds_)
        n_classes = len(self.classes_)
        proba = np.zeros((X_np.shape[0], n_classes), dtype=np.float64)

        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X_binned)
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


class FederatedHistogramRFRegressor(RegressorMixin, BaseFederatedTreeEstimator):
    """Federated Random Forest regressor using histogram-based feature discretisation.

    Regression counterpart of :class:`FederatedHistogramRFClassifier`.  Uses
    the same two-round protocol (histogram exchange → constrained tree
    training) but with
    :class:`~sklearn.ensemble.RandomForestRegressor` and mean prediction.

    Parameters
    ----------
    n_estimators : int, default=100
        Trees per client per training round.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Number of tree-training rounds after the histogram exchange.
    iid : bool, default=True
        IID data partitioning; Dirichlet non-IID when ``False``.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int or None, default=None
        Maximum tree depth.
    n_bins : int, default=32
        Number of histogram bins per feature.
    use_flower : bool, default=False
        If ``True``, use Flower Strategy / NumPyClient types; sets
        ``self.strategy_`` after fit.
    random_state : int or None, default=None
        Seed for data partitioning and per-client RF training.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        All fitted trees from all clients and all training rounds.
    thresholds_ : list of NDArray
        Global bin edges per feature used for discretisation at predict time.
    n_features_in_ : int
        Number of features seen during ``fit()``.
    strategy_ : FedHistogramAggregation
        The Flower strategy instance; set only when ``use_flower=True``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_clients: int = 5,
        n_rounds: int = 1,
        iid: bool = True,
        alpha: float = 0.5,
        max_depth: int | None = None,
        n_bins: int = 32,
        use_flower: bool = False,
        feature_ranges: np.ndarray | None = None,
        metadata_epsilon: float = 5.0,
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
        self.n_bins = n_bins
        self.use_flower = use_flower
        self.feature_ranges = feature_ranges
        self.metadata_epsilon = metadata_epsilon

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedHistogramRFRegressor:
        """Fit via federated histogram aggregation followed by bagging.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FederatedHistogramRFRegressor
            Fitted estimator.
        """
        X_np, y_np = self._validate_data_federated(X, y, reset=True)

        n_samples, n_features = X_np.shape
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

        if self.feature_ranges is not None:
            ranges = np.asarray(self.feature_ranges)
            global_edges: list[NDArray] | None = [
                np.linspace(ranges[f, 0], ranges[f, 1], self.n_bins + 1)
                for f in range(n_features)
            ]
        else:
            global_edges = None  # Option A: DP Round 0 will compute edges

        if self.use_flower:
            self._run_flower_fit(partitions, rng, global_edges, n_features)
        else:
            self._run_local_fit(partitions, rng, global_edges, n_features)

        return self

    def _run_local_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        rng: np.random.Generator,
        global_edges: list[NDArray] | None,
        n_features: int,
    ) -> None:
        """Train locally: bin features, train RF per partition."""
        if global_edges is None:
            # Option A: DP-noised range exchange across partitions
            rng_dp = np.random.default_rng(self.random_state)
            noise_scale = 1.0 / self.metadata_epsilon
            all_mins = np.array([[X_i[:, f].min() for f in range(n_features)] for X_i, _ in partitions])
            all_maxs = np.array([[X_i[:, f].max() for f in range(n_features)] for X_i, _ in partitions])
            noisy_mins = all_mins + rng_dp.laplace(0, noise_scale, all_mins.shape)
            noisy_maxs = all_maxs + rng_dp.laplace(0, noise_scale, all_maxs.shape)
            global_min = noisy_mins.min(axis=0)
            global_max = noisy_maxs.max(axis=0)
            global_max = np.maximum(global_max, global_min + 1e-6)
            global_edges = [
                np.linspace(global_min[f], global_max[f], self.n_bins + 1)
                for f in range(n_features)
            ]

        self.thresholds_ = global_edges
        self.estimators_: list[DecisionTreeRegressor] = []

        for round_idx in range(self.n_rounds):
            for client_idx, (X_i, y_i) in enumerate(partitions):
                client_seed = (
                    None
                    if self.random_state is None
                    else int(rng.integers(0, 2**31))
                )
                X_binned = _apply_bins(X_i, global_edges)
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=client_seed,
                )
                rf.fit(X_binned, y_i)
                self.estimators_.extend(rf.estimators_)
                logger.debug(
                    "Round %d, client %d: %d trees trained on %d samples",
                    round_idx + 1,
                    client_idx,
                    len(rf.estimators_),
                    len(y_i),
                )

    def _run_flower_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        rng: np.random.Generator,
        global_edges: list[NDArray] | None,
        n_features: int,
    ) -> None:
        """Drive the FL loop via Flower Strategy + NumPyClient types.

        Option C (feature_ranges provided): 2-round protocol (histogram + tree rounds).
        Option A (feature_ranges=None): 3-round protocol (metadata + histogram + tree rounds).
        """
        from flwr.common import FitIns, ndarrays_to_parameters

        from flwr_trees.aggregation.histogram import (
            FedHistogramAggregation,
            HistogramClient,
            _LocalHistogramProxy,
        )

        strategy = FedHistogramAggregation(n_bins=self.n_bins)

        proxies: list[_LocalHistogramProxy] = []
        for client_idx, (X_i, y_i) in enumerate(partitions):
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            client = HistogramClient(
                X_i=X_i,
                y_i=y_i.astype(np.float64),
                n_bins=self.n_bins,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=client_seed,
                global_edges=global_edges,
                task="regress",
                metadata_epsilon=self.metadata_epsilon,
            )
            proxies.append(_LocalHistogramProxy(cid=str(client_idx), client=client))

        if global_edges is not None:
            # Option C: edges pre-agreed, skip Round 0
            edge_params = ndarrays_to_parameters(global_edges)
        else:
            # Option A: Round 0 — DP metadata exchange to determine global edges
            edge_params, _ = strategy.aggregate_fit(
                server_round=0,
                results=[
                    (
                        proxy,
                        proxy.fit(
                            FitIns(
                                parameters=ndarrays_to_parameters([]),
                                config={"phase": "metadata"},
                            ),
                            timeout=None,
                            group_id=None,
                        ),
                    )
                    for proxy in proxies
                ],
                failures=[],
            )
            # Update client global_edges with server-computed edges
            server_edges = strategy.global_edges_
            for proxy in proxies:
                proxy._client.global_edges = server_edges

        # Round 1: histogram exchange
        threshold_params, _ = strategy.aggregate_fit(
            server_round=1,
            results=[
                (
                    proxy,
                    proxy.fit(
                        FitIns(
                            parameters=edge_params,
                            config={"phase": "histogram"},
                        ),
                        timeout=None,
                        group_id=None,
                    ),
                )
                for proxy in proxies
            ],
            failures=[],
        )

        # Round 2+ (n_rounds cycles): always re-pass threshold_params from round 1
        for _ in range(self.n_rounds):
            strategy.aggregate_fit(
                server_round=2,
                results=[
                    (
                        proxy,
                        proxy.fit(
                            FitIns(
                                parameters=threshold_params,
                                config={"phase": "train"},
                            ),
                            timeout=None,
                            group_id=None,
                        ),
                    )
                    for proxy in proxies
                ],
                failures=[],
            )

        self.estimators_ = strategy.trees_  # type: ignore[assignment]
        self.thresholds_ = strategy.thresholds_
        self.strategy_ = strategy

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict regression targets as mean of tree predictions on binned features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Mean predicted target values across all trees.
        """
        check_is_fitted(self)
        X_np = self._validate_data_federated(X, reset=False)
        X_binned = _apply_bins(X_np, self.thresholds_)
        predictions = np.array([tree.predict(X_binned) for tree in self.estimators_])
        return predictions.mean(axis=0).astype(np.float64)
