from __future__ import annotations

from flwr_trees.estimators import (
    FederatedHistogramRFClassifier,
    FederatedHistogramRFRegressor,
    FederatedRandomForestClassifier,
    FederatedRandomForestRegressor,
    FederatedXGBClassifier,
    FederatedXGBRegressor,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FederatedRandomForestClassifier",
    "FederatedRandomForestRegressor",
    "FederatedXGBClassifier",
    "FederatedXGBRegressor",
    "FederatedHistogramRFClassifier",
    "FederatedHistogramRFRegressor",
]
