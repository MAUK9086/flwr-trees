from __future__ import annotations

from flwr_trees.aggregation.bagging import FedForestBagging, FedForestBaggingClient
from flwr_trees.aggregation.cyclic import FedForestCyclic, XGBCyclicClient
from flwr_trees.aggregation.histogram import FedHistogramAggregation, HistogramClient

__all__ = [
    "FedForestBagging",
    "FedForestBaggingClient",
    "FedForestCyclic",
    "XGBCyclicClient",
    "FedHistogramAggregation",
    "HistogramClient",
]
