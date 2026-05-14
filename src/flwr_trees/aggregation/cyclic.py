from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import xgboost as xgb
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
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class XGBCyclicClient(NumPyClient):
    """Flower NumPyClient for cyclic XGBoost federation.

    Receives the current global Booster (or nothing on the very first step),
    continues training from it on the local partition, and returns the
    updated Booster as a single pickle-encoded uint8 NDArray.

    Parameters
    ----------
    X_i : NDArray of shape (n_samples, n_features)
        Local feature partition.
    y_i : NDArray of shape (n_samples,)
        Local label partition (encoded integers for classifiers; raw floats
        for regressors).
    xgb_params : dict[str, Any]
        Native XGBoost params dict (``eta``, ``max_depth``, ``objective``,
        etc.) plus two special keys consumed here and not forwarded:

        * ``"_task"`` — ``"classify"`` or ``"regress"`` (unused; kept for
          future logging).
        * ``"_num_boost_round"`` — number of boosting rounds per step.
    """

    def __init__(
        self,
        X_i: NDArray,
        y_i: NDArray,
        xgb_params: dict[str, Any],
    ) -> None:
        self.X_i = X_i
        self.y_i = y_i
        self._task: str = str(xgb_params.get("_task", "classify"))
        self._num_boost_round: int = int(xgb_params.get("_num_boost_round", 100))
        self.xgb_params: dict[str, Any] = {
            k: v for k, v in xgb_params.items() if not k.startswith("_")
        }
        self.booster_: xgb.Booster | None = None

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Continue-train on local data and return the updated Booster.

        Parameters
        ----------
        parameters : NDArrays
            Either empty (first step, no prior Booster) or a single 1-D
            uint8 array containing a pickle-serialized ``xgboost.Booster``.
        config : dict[str, Scalar]
            Server configuration dict (e.g. ``{"round": 1}``).

        Returns
        -------
        arrays : NDArrays
            Single-element list containing the updated Booster as a 1-D
            uint8 array (pickle-serialized).
        n_examples : int
            Number of local training samples.
        metrics : dict[str, Scalar]
            Empty dict.
        """
        prior: xgb.Booster | None = (
            pickle.loads(parameters[0].tobytes()) if parameters else None
        )

        dtrain = xgb.DMatrix(self.X_i, label=self.y_i)
        booster = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self._num_boost_round,
            xgb_model=prior,
        )
        self.booster_ = booster

        arrays: NDArrays = [
            np.frombuffer(pickle.dumps(booster), dtype=np.uint8)
        ]
        logger.debug(
            "XGBCyclicClient.fit: 1 booster returned, %d samples, step=%s",
            len(self.y_i),
            config.get("round"),
        )
        return arrays, int(len(self.y_i)), {}

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return empty list; cyclic clients have no separate parameter vector."""
        return []

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Not used in cyclic boosting; returns zeros."""
        return 0.0, 0, {}


class _LocalCyclicProxy(ClientProxy):
    """Minimal ClientProxy stub for in-process cyclic FL simulation.

    Delegates ``fit()`` to a ``XGBCyclicClient`` instance. All other abstract
    methods raise ``NotImplementedError`` because they are never called by
    the local runner.

    Parameters
    ----------
    cid : str
        Client identifier string.
    client : XGBCyclicClient
        The local client to delegate ``fit()`` to.
    """

    def __init__(self, cid: str, client: XGBCyclicClient) -> None:
        super().__init__(cid=cid)
        self._client = client

    def fit(
        self,
        ins: FitIns,
        timeout: float | None,
        group_id: int | None,
    ) -> FitRes:
        """Delegate to the local client and wrap result in FitRes."""
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


class FedForestCyclic(Strategy):
    """Flower Strategy for federated XGBoost via cyclic boosting.

    In each Flower round, exactly ONE client continues training from the
    current global Booster and returns the updated Booster.  The server
    passes the updated Booster to the next client in the next round.

    With ``n_clients`` clients and ``n_rounds`` complete cycles, the total
    number of Flower rounds (and therefore the length of
    ``bytes_sent_per_round`` after fitting) is ``n_clients * n_rounds``.

    The stub methods (``configure_fit``, ``configure_evaluate``,
    ``aggregate_evaluate``, ``evaluate``) exist to satisfy the abstract
    ``Strategy`` interface for future real Flower deployment.  They are not
    called by ``FederatedXGBClassifier._run_flower_fit()``.

    Attributes
    ----------
    bytes_sent_per_round : list[int]
        Serialized bytes of the Booster received from each client, one entry
        per Flower step.
    """

    def __init__(self) -> None:
        self.bytes_sent_per_round: list[int] = []

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Parameters | None:
        """Return empty initial parameters; first client starts from scratch."""
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
        """Accept one client's updated Booster and pass it to the next client.

        Parameters
        ----------
        server_round : int
            Current Flower step (1-based; equals the global step counter in
            the local runner, not a "round" over all clients).
        results : list of (ClientProxy, FitRes)
            Exactly one element — the single client that trained this step.
        failures : list
            Ignored; the local runner never produces failures.

        Returns
        -------
        parameters : Parameters
            The updated Booster parameters to forward to the next client.
        metrics : dict[str, Scalar]
            Empty dict.
        """
        _, fit_res = results[0]
        round_bytes = sum(len(t) for t in fit_res.parameters.tensors)
        self.bytes_sent_per_round.append(round_bytes)
        logger.info(
            "Cyclic step %d: %d bytes received from client",
            server_round,
            round_bytes,
        )
        return fit_res.parameters, {}

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
