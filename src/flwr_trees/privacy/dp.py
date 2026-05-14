from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class NoisyHistogram:
    """Adds Laplace noise to histogram counts for local differential privacy.

    Provides an ``epsilon``-DP guarantee on per-feature bin counts.  The
    sensitivity is 1 because a single training sample changes at most one
    bin by exactly 1 count.

    Parameters
    ----------
    epsilon : float, default=1.0
        Privacy budget.  Smaller values → more noise → stronger privacy.
    random_state : int or None, default=None
        Seed for the Laplace noise generator.

    Examples
    --------
    >>> import numpy as np
    >>> counts = np.array([10.0, 20.0, 15.0])
    >>> noisy = NoisyHistogram(epsilon=1.0, random_state=0).apply(counts)
    >>> noisy.shape
    (3,)
    >>> (noisy >= 0).all()
    True
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.random_state = random_state

    def apply(self, counts: NDArray) -> NDArray:
        """Return counts with additive Laplace noise, clipped to non-negative.

        Parameters
        ----------
        counts : ndarray of shape (n_bins,)
            Raw histogram bin counts (non-negative integers or floats).

        Returns
        -------
        noisy_counts : ndarray of shape (n_bins,)
            Counts plus Laplace(0, 1/epsilon) noise, clipped to ≥ 0.
        """
        rng = np.random.default_rng(self.random_state)
        noise = rng.laplace(loc=0.0, scale=1.0 / self.epsilon, size=counts.shape)
        return np.clip(counts.astype(np.float64) + noise, 0.0, None)


_DELTA = 1e-5


class DPTreeWrapper:
    """Wraps a fitted decision tree to add Gaussian DP noise to leaf predictions.

    Adds Gaussian noise calibrated to achieve (``epsilon``, ``delta``)-DP,
    where ``delta`` = 1e-5.  The standard deviation is
    ``sigma = sqrt(2 * log(1.25 / delta)) / epsilon`` (sensitivity = 1).

    After noise addition, probabilities are clipped to ``[0, inf)`` and
    renormalised so each row sums to 1.

    Parameters
    ----------
    tree : DecisionTreeClassifier or DecisionTreeRegressor
        A fitted sklearn decision tree.
    epsilon : float, default=1.0
        Privacy budget.
    random_state : int or None, default=None
        Seed for the Gaussian noise generator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=0)
    >>> tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    >>> wrapper = DPTreeWrapper(tree, epsilon=1.0, random_state=0)
    >>> proba = wrapper.predict_proba(X[:5])
    >>> proba.shape
    (5, 2)
    >>> np.allclose(proba.sum(axis=1), 1.0)
    True
    """

    def __init__(
        self,
        tree: DecisionTreeClassifier | DecisionTreeRegressor,
        epsilon: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.tree = tree
        self.epsilon = epsilon
        self.random_state = random_state
        self._sigma = (
            np.sqrt(2.0 * np.log(1.25 / _DELTA)) / epsilon
        )

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities with Gaussian noise on leaf values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Noisy probability estimates, clipped to ≥ 0 and row-normalised
            to sum to 1.
        """
        rng = np.random.default_rng(self.random_state)
        proba = self.tree.predict_proba(X).astype(np.float64)
        proba += rng.normal(0.0, self._sigma, proba.shape)
        proba = np.clip(proba, 0.0, None)
        row_sums = proba.sum(axis=1, keepdims=True)
        zero_rows = row_sums == 0.0
        # Rows where all values clipped to 0: fall back to uniform distribution.
        n_classes = proba.shape[1]
        uniform = np.full_like(proba, 1.0 / n_classes)
        safe_sums = np.where(zero_rows, 1.0, row_sums)
        normalized = proba / safe_sums
        return np.where(zero_rows, uniform, normalized)

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels as argmax of noisy probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class indices.
        """
        return np.argmax(self.predict_proba(X), axis=1)
