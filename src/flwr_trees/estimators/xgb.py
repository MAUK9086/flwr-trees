from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from flwr_trees.estimators.base import BaseFederatedTreeEstimator
from flwr_trees.simulation.partitioning import simulate_clients

logger = logging.getLogger(__name__)


def _xgb_params(
    max_depth: int,
    learning_rate: float,
    subsample: float,
    seed: int | None,
    objective: str,
    num_class: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build a native XGBoost params dict for ``xgb.train()``."""
    p: dict[str, Any] = {
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "objective": objective,
        "verbosity": 0,
    }
    if seed is not None:
        p["seed"] = seed
    if num_class is not None:
        p["num_class"] = num_class
    if device != "cpu":
        p["device"] = device
    return p


class FederatedXGBClassifier(ClassifierMixin, BaseFederatedTreeEstimator):
    """Federated XGBoost classifier using cyclic boosting.

    Partitions training data among ``n_clients`` simulated clients and
    trains a single XGBoost Booster by passing it cyclically through all
    clients, one per step.  Each client adds ``n_estimators`` boosting
    rounds to the shared Booster.  This mirrors the semantics of
    ``flwr.server.strategy.FedXgbCyclic``.

    Unlike :class:`FederatedRandomForestClassifier`, the result is a single
    ``xgboost.Booster`` stored in ``self.booster_``.

    Parameters
    ----------
    n_estimators : int, default=100
        Boosting rounds *added per client per step*.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Number of complete cycles through all clients.
    iid : bool, default=True
        IID data partitioning when ``True``; Dirichlet non-IID when
        ``False``.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int, default=6
        Maximum XGBoost tree depth.
    learning_rate : float, default=0.1
        Boosting learning rate (eta).
    subsample : float, default=1.0
        Subsample ratio of the training samples.
    use_flower : bool, default=False
        If ``False``, run the plain local cyclic loop (fast, used by
        ``check_estimator`` and unit tests).  If ``True``, drive the FL
        loop via Flower ``Strategy`` / ``NumPyClient`` types using an
        in-process orchestrator; sets ``self.strategy_`` after fit.
    random_state : int or None, default=None
        Seed for data partitioning and per-client XGBoost training.

    Attributes
    ----------
    booster_ : xgboost.Booster
        The final Booster after all cyclic rounds.
    classes_ : ndarray of shape (n_classes,)
        Original class labels seen during ``fit()``.
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
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        use_flower: bool = False,
        device: str = "cpu",
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
        self.use_flower = use_flower
        self.device = device

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedXGBClassifier:
        """Fit via cyclic federated XGBoost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : FederatedXGBClassifier
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
        """Train via cyclic boosting using xgb.train() directly.

        Uses the native XGBoost API to avoid sklearn wrapper class-validation
        errors when individual partitions contain a subset of global classes.
        """
        n_classes = len(le.classes_)
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        num_class = n_classes if n_classes > 2 else None

        booster: xgb.Booster | None = None

        for _ in range(self.n_rounds):
            for client_idx, (X_i, y_i_orig) in enumerate(partitions):
                y_i = le.transform(y_i_orig).astype(np.float32)
                client_seed = (
                    None
                    if self.random_state is None
                    else int(rng.integers(0, 2**31))
                )
                params = _xgb_params(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    seed=client_seed,
                    objective=objective,
                    num_class=num_class,
                    device=self.device,
                )
                dtrain = xgb.DMatrix(X_i, label=y_i)
                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.n_estimators,
                    xgb_model=booster,
                    verbose_eval=False,
                )
                logger.debug(
                    "Client %d: %d boosting rounds added", client_idx, self.n_estimators
                )

        self.booster_: xgb.Booster = booster  # type: ignore[assignment]

    def _run_flower_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        le: LabelEncoder,
        rng: np.random.Generator,
    ) -> None:
        """Drive the cyclic FL loop via Flower Strategy + NumPyClient types.

        Builds one ``XGBCyclicClient`` and ``_LocalCyclicProxy`` per
        partition, then cycles through all clients for ``self.n_rounds``
        rounds.  After completion sets ``self.booster_`` and
        ``self.strategy_``.
        """
        from flwr.common import FitIns, ndarrays_to_parameters

        from flwr_trees.aggregation.cyclic import (
            FedForestCyclic,
            XGBCyclicClient,
            _LocalCyclicProxy,
        )

        n_classes = len(le.classes_)
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        num_class = n_classes if n_classes > 2 else None

        proxies: list[_LocalCyclicProxy] = []
        for client_idx, (X_i, y_i_orig) in enumerate(partitions):
            y_i = le.transform(y_i_orig).astype(np.float32)
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            xgb_params = _xgb_params(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                seed=client_seed,
                objective=objective,
                num_class=num_class,
                device=self.device,
            )
            xgb_params["_task"] = "classify"
            xgb_params["_num_boost_round"] = self.n_estimators
            client = XGBCyclicClient(X_i=X_i, y_i=y_i, xgb_params=xgb_params)
            proxies.append(_LocalCyclicProxy(cid=str(client_idx), client=client))

        strategy = FedForestCyclic()
        current_params = ndarrays_to_parameters([])

        for round_idx in range(1, self.n_rounds + 1):
            for client_idx, proxy in enumerate(proxies):
                step = (round_idx - 1) * len(proxies) + client_idx + 1
                fit_ins = FitIns(
                    parameters=current_params,
                    config={"round": step},
                )
                fit_res = proxy.fit(fit_ins, timeout=None, group_id=None)
                current_params, _ = strategy.aggregate_fit(
                    server_round=step,
                    results=[(proxy, fit_res)],
                    failures=[],
                )
                logger.info(
                    "Flower cyclic step %d (round %d/%d, client %d/%d)",
                    step,
                    round_idx,
                    self.n_rounds,
                    client_idx + 1,
                    len(proxies),
                )

        self.booster_ = proxies[-1]._client.booster_  # type: ignore[assignment]
        self.strategy_ = strategy

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities using the final Booster.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability estimates.
        """
        check_is_fitted(self)
        X_np = self._validate_data_federated(X, reset=False)
        dtest = xgb.DMatrix(X_np)
        raw = self.booster_.predict(dtest)
        n_classes = len(self.classes_)
        if n_classes > 2:
            # multi:softprob returns a flat array of length n_samples * n_classes
            proba: NDArray = raw.reshape(X_np.shape[0], n_classes)
        else:
            # binary:logistic returns probability of the positive class
            proba = np.column_stack([1.0 - raw, raw])
        return proba.astype(np.float64)

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels as argmax of class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels decoded to original label space.
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class FederatedXGBRegressor(RegressorMixin, BaseFederatedTreeEstimator):
    """Federated XGBoost regressor using cyclic boosting.

    Same cyclic federation pattern as :class:`FederatedXGBClassifier` but
    for regression targets.  The final artefact is an ``xgboost.Booster``
    stored in ``self.booster_``.

    Parameters
    ----------
    n_estimators : int, default=100
        Boosting rounds added per client per step.
    n_clients : int, default=5
        Number of simulated FL clients.
    n_rounds : int, default=1
        Number of complete cycles through all clients.
    iid : bool, default=True
        IID data partitioning; Dirichlet non-IID when ``False``.
    alpha : float, default=0.5
        Dirichlet concentration parameter (used only when ``iid=False``).
    max_depth : int, default=6
        Maximum XGBoost tree depth.
    learning_rate : float, default=0.1
        Boosting learning rate (eta).
    subsample : float, default=1.0
        Subsample ratio of the training samples.
    use_flower : bool, default=False
        If ``True``, use Flower Strategy/NumPyClient types; sets
        ``self.strategy_`` after fit.
    random_state : int or None, default=None
        Seed for data partitioning and per-client XGBoost training.

    Attributes
    ----------
    booster_ : xgboost.Booster
        The final Booster after all cyclic rounds.
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
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        use_flower: bool = False,
        device: str = "cpu",
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
        self.use_flower = use_flower
        self.device = device

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> FederatedXGBRegressor:
        """Fit via cyclic federated XGBoost regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FederatedXGBRegressor
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

        if self.use_flower:
            self._run_flower_fit(partitions, rng)
        else:
            self._run_local_fit(partitions, rng)

        return self

    def _run_local_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        rng: np.random.Generator,
    ) -> None:
        """Train via cyclic boosting using xgb.train() directly."""
        booster: xgb.Booster | None = None

        for _ in range(self.n_rounds):
            for client_idx, (X_i, y_i) in enumerate(partitions):
                client_seed = (
                    None
                    if self.random_state is None
                    else int(rng.integers(0, 2**31))
                )
                params = _xgb_params(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    seed=client_seed,
                    objective="reg:squarederror",
                    device=self.device,
                )
                dtrain = xgb.DMatrix(X_i, label=y_i.astype(np.float32))
                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.n_estimators,
                    xgb_model=booster,
                    verbose_eval=False,
                )
                logger.debug(
                    "Client %d (regressor): %d rounds added", client_idx, self.n_estimators
                )

        self.booster_: xgb.Booster = booster  # type: ignore[assignment]

    def _run_flower_fit(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        rng: np.random.Generator,
    ) -> None:
        """Drive the cyclic FL loop via Flower types (in-process, no Ray)."""
        from flwr.common import FitIns, ndarrays_to_parameters

        from flwr_trees.aggregation.cyclic import (
            FedForestCyclic,
            XGBCyclicClient,
            _LocalCyclicProxy,
        )

        proxies: list[_LocalCyclicProxy] = []
        for client_idx, (X_i, y_i) in enumerate(partitions):
            client_seed = (
                None
                if self.random_state is None
                else int(rng.integers(0, 2**31))
            )
            xgb_params = _xgb_params(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                seed=client_seed,
                objective="reg:squarederror",
                device=self.device,
            )
            xgb_params["_task"] = "regress"
            xgb_params["_num_boost_round"] = self.n_estimators
            client = XGBCyclicClient(
                X_i=X_i, y_i=y_i.astype(np.float32), xgb_params=xgb_params
            )
            proxies.append(_LocalCyclicProxy(cid=str(client_idx), client=client))

        strategy = FedForestCyclic()
        current_params = ndarrays_to_parameters([])

        for round_idx in range(1, self.n_rounds + 1):
            for client_idx, proxy in enumerate(proxies):
                step = (round_idx - 1) * len(proxies) + client_idx + 1
                fit_ins = FitIns(
                    parameters=current_params,
                    config={"round": step},
                )
                fit_res = proxy.fit(fit_ins, timeout=None, group_id=None)
                current_params, _ = strategy.aggregate_fit(
                    server_round=step,
                    results=[(proxy, fit_res)],
                    failures=[],
                )

        self.booster_ = proxies[-1]._client.booster_  # type: ignore[assignment]
        self.strategy_ = strategy

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict regression targets using the final Booster.

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
        dtest = xgb.DMatrix(X_np)
        return self.booster_.predict(dtest).astype(np.float64)
