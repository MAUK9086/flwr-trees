from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted, validate_data

from flwr_trees.compat.array_api import to_numpy

logger = logging.getLogger(__name__)

# Sentinel distinguishing "caller didn't pass y" (predict path) from
# "caller passed y=None" (fit path → let sklearn raise the proper error).
_UNSET: object = object()


class BaseFederatedTreeEstimator(BaseEstimator, ABC):
    """Abstract base class for all federated tree estimators.

    Provides shared constructor parameters, input validation, and a unified
    ``score()`` implementation that dispatches to accuracy (classifiers) or
    R² (regressors).  Concrete subclasses must implement ``fit()`` and
    ``predict()``.

    Parameters
    ----------
    n_clients : int, default=5
        Number of simulated or real FL clients.
    n_rounds : int, default=1
        Number of federated training rounds.
    random_state : int or None, default=None
        Seed for reproducibility of data partitioning and tree training.
    """

    def __init__(
        self,
        n_clients: int = 5,
        n_rounds: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.random_state = random_state

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseFederatedTreeEstimator:
        """Fit the federated estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values or labels.

        Returns
        -------
        self : BaseFederatedTreeEstimator
            Fitted estimator.
        """

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray:
        """Predict using the fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values or labels.
        """

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: NDArray | None = None,
    ) -> float:
        """Score the estimator on the given data.

        Uses accuracy for classifiers and R² for regressors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            True labels or values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights passed to the metric function.

        Returns
        -------
        score : float
            Accuracy score for classifiers, R² score for regressors.
        """
        check_is_fitted(self)
        y_pred = self.predict(X)
        y_np = to_numpy(np.asarray(y))
        if isinstance(self, ClassifierMixin):
            return float(accuracy_score(y_np, y_pred, sample_weight=sample_weight))
        return float(r2_score(y_np, y_pred, sample_weight=sample_weight))

    def _validate_data_federated(
        self,
        X: ArrayLike,
        y: ArrayLike | object = _UNSET,
        reset: bool = True,
    ) -> tuple[NDArray, NDArray] | NDArray:
        """Validate input and convert to numpy arrays.

        Wraps sklearn's ``validate_data`` (sklearn >= 1.6 module-level API)
        and converts outputs to numpy via
        :func:`~flwr_trees.compat.array_api.to_numpy`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,) or ``_UNSET``
            Target values.  Omit (or pass the sentinel ``_UNSET``) on the
            predict path so only ``X_np`` is returned.  Pass explicitly —
            even as ``None`` — on the fit path so sklearn can raise the
            appropriate ``ValueError`` when ``y`` is missing.
        reset : bool, default=True
            Pass ``True`` during ``fit()`` to record ``n_features_in_``, and
            ``False`` during ``predict()`` / ``score()`` to verify consistency.

        Returns
        -------
        X_np : ndarray
            Validated feature array (predict path, ``y`` omitted).
        (X_np, y_np) : tuple of ndarrays
            Both arrays (fit path, ``y`` provided).
        """
        if y is _UNSET:
            X_out = validate_data(self, X, reset=reset)
            return to_numpy(X_out)
        X_out, y_out = validate_data(self, X, y, reset=reset)
        return to_numpy(X_out), to_numpy(y_out)
