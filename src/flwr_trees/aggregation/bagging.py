from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from flwr.client import NumPyClient
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import NDArrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)


class FedForestBaggingClient(NumPyClient):
    """Flower NumPyClient that trains a local RandomForestClassifier.

    On each call to ``fit()``, trains a fresh ``RandomForestClassifier`` on
    the local data partition and returns the serialized decision trees as a
    list of 1-D uint8 NDArrays (one per tree, pickle-encoded).

    Parameters
    ----------
    X_i : NDArray of shape (n_samples, n_features)
        Local feature partition.
    y_i : NDArray of shape (n_samples,)
        Local label partition (encoded as contiguous integers).
    rf_params : dict[str, Any]
        Keyword arguments forwarded verbatim to ``RandomForestClassifier``.
    """

    def __init__(
        self,
        X_i: NDArray,
        y_i: NDArray,
        rf_params: dict[str, Any],
    ) -> None:
        self.X_i = X_i
        self.y_i = y_i
        self.rf_params = rf_params

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train RF on local data and return serialized trees.

        Parameters
        ----------
        parameters : NDArrays
            Ignored; bagging does not use a global model.
        config : dict[str, Scalar]
            Configuration passed by the server (e.g. ``{"round": 1}``).

        Returns
        -------
        arrays : NDArrays
            One 1-D uint8 array per decision tree (pickle-serialized).
        n_examples : int
            Number of local training samples.
        metrics : dict[str, Scalar]
            Empty dict; no per-client metrics for bagging.
        """
        rf = RandomForestClassifier(**self.rf_params)
        rf.fit(self.X_i, self.y_i)
        arrays: NDArrays = [
            np.frombuffer(pickle.dumps(tree), dtype=np.uint8)
            for tree in rf.estimators_
        ]
        logger.debug(
            "FedForestBaggingClient.fit: %d trees from %d samples",
            len(arrays),
            len(self.y_i),
        )
        return arrays, int(len(self.y_i)), {}

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return empty parameter list; bagging has no global model."""
        return []

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Not used in bagging; returns zeros."""
        return 0.0, 0, {}


class _LocalClientProxy(ClientProxy):
    """Minimal ClientProxy stub for in-process FL simulation.

    Delegates ``fit()`` calls to a ``FedForestBaggingClient`` instance.
    All other abstract methods raise ``NotImplementedError`` because they
    are never called by the local runner.

    Parameters
    ----------
    cid : str
        Client identifier string.
    client : FedForestBaggingClient
        The local client to delegate ``fit()`` to.
    """

    def __init__(self, cid: str, client: FedForestBaggingClient) -> None:
        super().__init__(cid=cid)
        self._client = client

    def fit(
        self,
        ins: FitIns,
        timeout: float | None,
        group_id: int | None,
    ) -> FitRes:
        """Delegate to the local client and wrap the result in FitRes."""
        arrays, n_examples, metrics = self._client.fit(
            parameters_to_ndarrays(ins.parameters),
            ins.config,
        )
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=n_examples,
            metrics=metrics,
        )

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetPropertiesRes:
        raise NotImplementedError

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetParametersRes:
        raise NotImplementedError

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: float | None,
        group_id: int | None,
    ) -> EvaluateRes:
        raise NotImplementedError

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> DisconnectRes:
        raise NotImplementedError


class FedForestBagging(Strategy):
    """Flower Strategy for federated random forest via tree bagging.

    Each round, every client trains a local ``RandomForestClassifier`` and
    returns its serialized decision trees.  The server collects all trees
    from all clients across all rounds into ``self.trees_``.

    The stub methods (``configure_fit``, ``configure_evaluate``,
    ``aggregate_evaluate``, ``evaluate``) exist to satisfy the abstract
    ``Strategy`` interface for future real Flower deployment.  They are not
    called by ``FederatedRandomForestClassifier._run_flower_fit()``.

    Attributes
    ----------
    bytes_sent_per_round : list[int]
        Total serialized bytes received from all clients in each round.
        Populated in ``aggregate_fit()``; length equals the number of
        completed rounds.
    trees_ : list
        All deserialized decision trees accumulated across every round and
        every client.
    """

    def __init__(self) -> None:
        self.bytes_sent_per_round: list[int] = []
        self.trees_: list = []

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Parameters | None:
        """Return empty initial parameters; bagging requires no global model."""
        return ndarrays_to_parameters([])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Not called by the local runner; returns empty list."""
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Deserialize trees from all clients and accumulate into ``trees_``.

        Parameters
        ----------
        server_round : int
            Current FL round (1-based).
        results : list of (ClientProxy, FitRes)
            One entry per client that completed ``fit()``.
        failures : list
            Ignored; the local runner never produces failures.

        Returns
        -------
        parameters : Parameters
            Empty Parameters (aggregated model not needed for bagging).
        metrics : dict[str, Scalar]
            Empty dict.
        """
        round_bytes = 0
        round_trees: list = []
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            trees = [pickle.loads(arr.tobytes()) for arr in arrays]
            round_trees.extend(trees)
            round_bytes += sum(len(t) for t in fit_res.parameters.tensors)
        self.trees_.extend(round_trees)
        self.bytes_sent_per_round.append(round_bytes)
        logger.info(
            "Round %d: %d trees aggregated, %d bytes received",
            server_round,
            len(round_trees),
            round_bytes,
        )
        return ndarrays_to_parameters([]), {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Not used; returns empty list."""
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Not used; returns (None, {})."""
        return None, {}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Not used; returns None."""
        return None
