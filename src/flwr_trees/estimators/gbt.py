from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from flwr_trees.estimators.base import BaseFederatedTreeEstimator
from flwr_trees.simulation.partitioning import simulate_clients

logger = logging.getLogger(__name__)


class FederatedGBTClassifier(ClassifierMixin, BaseFederatedTreeEstimator):
    """Federated Gradient Boosted Trees classifier using local simulation.

    Partitions training data among ``n_clients`` simulated clients, trains
    one :class:`~sklearn.ensemble.GradientBoostingClassifier` per client, then
    averages class probabilities across all client models for prediction.

    Unlike :class:`FederatedRandomForestClassifier`, which bags individual
    trees, each client here produces a complete gradient-boosted ensemble.
    Trees within a GBT are not independent (each corrects the residuals of
    the previous), so they cannot be meaningfully extracted and recombined
    without their learning-rate context.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds *per client*.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Accepted for API compatibility with other federated estimators; not
        used in GBT training (boosting rounds are controlled by ``n_estimators``).
    iid : bool, default=True
        If ``True``, data is partitioned uniformly at random (IID). If
        ``False``, uses Dirichlet-based non-IID partitioning.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int, default=3
        Maximum tree depth for each boosting round.
    learning_rate : float, default=0.1
        Step size shrinkage to prevent over-fitting.
    subsample : float, default=1.0
        Fraction of training samples used for fitting each base learner.
    random_state : int or None, default=None
        Seed for data partitioning and each client GBT.

    Attributes
    ----------
    estimators_ : list of GradientBoostingClassifier
        One fitted GBT model per active client after ``fit()``.
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
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
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
        self.learning_rate = learning_rate
        self.subsample = subsample

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedGBTClassifier:
        """Fit one GradientBoostingClassifier per client and collect all models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : FederatedGBTClassifier
            Fitted estimator.
        """
        X_np, y_np = self._validate_data_federated(X, y, reset=True)
        check_classification_targets(y_np)

        le = LabelEncoder()
        le.fit(y_np)
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

        self.estimators_: list[GradientBoostingClassifier] = []
        for client_idx, (X_i, y_i_orig) in enumerate(partitions):
            y_i = le.transform(y_i_orig)
            if len(np.unique(y_i)) < 2:
                logger.warning(
                    "Client %d has only %d unique class(es) — skipping GBT training",
                    client_idx,
                    len(np.unique(y_i)),
                )
                continue
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            gbt = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=client_seed,
            )
            gbt.fit(X_i, y_i)
            self.estimators_.append(gbt)
            logger.debug(
                "Client %d: trained GBT with %d rounds on %d samples",
                client_idx,
                self.n_estimators,
                len(y_i),
            )

        if len(self.estimators_) == 0:
            # Fallback: train directly on the full dataset.
            # Covers extreme cases (n_clients >> n_samples, all partitions single-class).
            # If the full dataset also has one class (e.g. n_samples=1), GBT raises its
            # own "got 1 class" ValueError which matches sklearn check_estimator patterns.
            logger.warning(
                "All %d client partitions were skipped (one class each); "
                "falling back to training on the full dataset.",
                len(partitions),
            )
            fallback_seed = (
                None if self.random_state is None else int(rng.integers(0, 2**31))
            )
            gbt = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=fallback_seed,
            )
            gbt.fit(X_np, le.transform(y_np))
            self.estimators_.append(gbt)

        return self

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities averaged across all client GBT models.

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

        for gbt in self.estimators_:
            gbt_proba = gbt.predict_proba(X_np)
            for j, cls in enumerate(gbt.classes_):
                proba[:, int(cls)] += gbt_proba[:, j]

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


class FederatedGBTRegressor(RegressorMixin, BaseFederatedTreeEstimator):
    """Federated Gradient Boosted Trees regressor using local simulation.

    Partitions training data among ``n_clients`` simulated clients, trains
    one :class:`~sklearn.ensemble.GradientBoostingRegressor` per client, then
    averages predictions across all client models.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds *per client*.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Accepted for API compatibility; not used in GBT training.
    iid : bool, default=True
        If ``True``, data is partitioned uniformly at random (IID). If
        ``False``, uses Dirichlet-based non-IID partitioning.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int, default=3
        Maximum tree depth for each boosting round.
    learning_rate : float, default=0.1
        Step size shrinkage to prevent over-fitting.
    subsample : float, default=1.0
        Fraction of training samples used for fitting each base learner.
    random_state : int or None, default=None
        Seed for data partitioning and each client GBT.

    Attributes
    ----------
    estimators_ : list of GradientBoostingRegressor
        One fitted GBT model per client after ``fit()``.
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
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
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
        self.learning_rate = learning_rate
        self.subsample = subsample

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedGBTRegressor:
        """Fit one GradientBoostingRegressor per client and collect all models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FederatedGBTRegressor
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

        self.estimators_: list[GradientBoostingRegressor] = []
        for client_idx, (X_i, y_i) in enumerate(partitions):
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            gbt = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=client_seed,
            )
            gbt.fit(X_i, y_i)
            self.estimators_.append(gbt)
            logger.debug(
                "Client %d: trained GBT regressor with %d rounds on %d samples",
                client_idx,
                self.n_estimators,
                len(y_i),
            )

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict targets as the mean across all client GBT predictions.

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
            [gbt.predict(X_np) for gbt in self.estimators_], dtype=np.float64
        )
        return np.mean(all_preds, axis=0)
