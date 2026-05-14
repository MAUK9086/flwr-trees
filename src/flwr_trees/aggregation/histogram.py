from __future__ import annotations

import logging
import pickle

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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

from flwr_trees.aggregation._tree_store import DiskTreeStore

logger = logging.getLogger(__name__)


class HistogramClient(NumPyClient):
    """Flower NumPyClient for two-phase federated histogram aggregation.

    Implements a two-phase protocol:

    1. **Histogram phase** — computes per-feature histograms on the local
       partition using pre-agreed global bin edges and returns counts + edges.
    2. **Train phase** — receives global split thresholds from the server,
       discretises local features with :func:`numpy.digitize`, trains a local
       :class:`~sklearn.ensemble.RandomForestClassifier` or
       :class:`~sklearn.ensemble.RandomForestRegressor` on the binned
       features, and returns the fitted trees.

    Parameters
    ----------
    X_i : NDArray of shape (n_samples, n_features)
        Local feature partition.
    y_i : NDArray of shape (n_samples,)
        Local label partition (encoded integers for classifiers; raw floats
        for regressors).
    n_bins : int
        Number of histogram bins (must match the global edges length − 1).
    n_estimators : int
        Trees per local Random Forest.
    max_depth : int or None
        Maximum tree depth passed to RandomForest.
    random_state : int or None
        Seed for the local RandomForest.
    global_edges : list of NDArray
        Pre-computed global bin edges per feature, each of shape
        ``(n_bins + 1,)``.  Passed by the estimator before FL begins so all
        clients histogram on the same grid.
    task : str
        ``"classify"`` or ``"regress"`` — selects RandomForestClassifier or
        RandomForestRegressor for the training phase.
    metadata_epsilon : float
        DP privacy budget for the metadata (range) phase.  Laplace noise with
        scale ``1/metadata_epsilon`` is added to local per-feature min/max
        before sending.
    """

    def __init__(
        self,
        X_i: NDArray,
        y_i: NDArray,
        n_bins: int,
        n_estimators: int,
        max_depth: int | None,
        random_state: int | None,
        global_edges: list[NDArray] | None,
        task: str = "classify",
        metadata_epsilon: float = 5.0,
    ) -> None:
        self.X_i = X_i
        self.y_i = y_i
        self.n_bins = n_bins
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.global_edges = global_edges
        self.task = task
        self.metadata_epsilon = metadata_epsilon
        self.trees_: list[DecisionTreeClassifier | DecisionTreeRegressor] | None = None

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Two-phase fit: histogram computation or constrained tree training.

        Parameters
        ----------
        parameters : NDArrays
            Empty list for the histogram phase.  One float64 array per
            feature (global split thresholds) for the training phase.
        config : dict[str, Scalar]
            Must contain ``"phase": "histogram"`` or ``"phase": "train"``.

        Returns
        -------
        arrays : NDArrays
            Histogram phase: interleaved ``[f0_counts, f0_edges, f1_counts,
            f1_edges, ...]`` as float64 arrays.
            Training phase: one uint8 array per fitted decision tree
            (pickle-serialised).
        n_examples : int
            Number of local training samples.
        metrics : dict[str, Scalar]
            Empty dict.
        """
        phase = str(config.get("phase", "histogram"))
        if phase == "metadata":
            return self._compute_metadata()
        if phase == "histogram":
            return self._compute_histograms()
        return self._train_forest(parameters)

    def _compute_metadata(self) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Send DP-noised local per-feature min/max for Round 0 range exchange."""
        rng = np.random.default_rng(self.random_state)
        noise_scale = 1.0 / self.metadata_epsilon
        arrays: NDArrays = []
        for f in range(self.X_i.shape[1]):
            local_min = float(self.X_i[:, f].min()) + rng.laplace(0.0, noise_scale)
            local_max = float(self.X_i[:, f].max()) + rng.laplace(0.0, noise_scale)
            arrays.append(np.array([local_min], dtype=np.float64))
            arrays.append(np.array([local_max], dtype=np.float64))
        return arrays, int(len(self.y_i)), {}

    def _compute_histograms(self) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Compute per-feature histograms on the pre-agreed global grid."""
        arrays: NDArrays = []
        for f, edges in enumerate(self.global_edges):
            counts, _ = np.histogram(self.X_i[:, f], bins=edges)
            arrays.append(counts.astype(np.float64))
            arrays.append(edges.astype(np.float64))
        logger.debug(
            "HistogramClient: histogram phase, %d features, %d samples",
            len(self.global_edges),
            len(self.y_i),
        )
        return arrays, int(len(self.y_i)), {}

    def _train_forest(
        self, parameters: NDArrays
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Bin features with global thresholds and train a local RF."""
        thresholds = parameters_to_ndarrays(
            ndarrays_to_parameters(parameters)
        ) if not isinstance(parameters, list) else parameters
        # parameters is already a list of NDArrays; one array per feature
        n_features = self.X_i.shape[1]
        X_binned = np.column_stack(
            [
                np.digitize(self.X_i[:, f], thresholds[f]).clip(
                    1, len(thresholds[f]) - 1
                )
                for f in range(n_features)
            ]
        ).astype(np.float64)

        if self.task == "classify":
            rf: RandomForestClassifier | RandomForestRegressor = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

        rf.fit(X_binned, self.y_i)
        self.trees_ = rf.estimators_

        arrays: NDArrays = [
            np.frombuffer(pickle.dumps(tree), dtype=np.uint8)
            for tree in rf.estimators_
        ]
        logger.debug(
            "HistogramClient: train phase, %d trees fitted on %d samples",
            len(arrays),
            len(self.y_i),
        )
        return arrays, int(len(self.y_i)), {}

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return empty list; parameters flow server → client in this protocol."""
        return []

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Not used in histogram FL; returns zeros."""
        return 0.0, 0, {}


class _LocalHistogramProxy(ClientProxy):
    """Minimal ClientProxy stub for in-process histogram FL simulation.

    Delegates ``fit()`` to a :class:`HistogramClient` instance.  All other
    abstract methods raise ``NotImplementedError`` because they are never
    called by the local runner.

    Parameters
    ----------
    cid : str
        Client identifier string.
    client : HistogramClient
        The local client to delegate ``fit()`` to.
    """

    def __init__(self, cid: str, client: HistogramClient) -> None:
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


class FedHistogramAggregation(Strategy):
    """Flower Strategy for communication-efficient federated RF via histograms.

    Implements a two-round protocol:

    **Round 1 (server_round == 1) — histogram aggregation:**
    All clients send per-feature histogram counts and bin edges.  The strategy
    sums counts across clients (valid because all clients use the same
    pre-agreed global bin edges) and returns the bin edges as global split
    thresholds to every client.

    **Round 2+ (server_round >= 2) — tree collection:**
    Clients receive global split thresholds, discretise their features, train
    local Random Forests, and return the fitted trees.  The strategy
    deserialises each tree and accumulates them in ``self.trees_``.

    Communication savings are tracked in ``bytes_saved_vs_bagging``: after the
    first tree round, the strategy estimates how many bytes would have been
    spent if clients had sent trees directly in round 1 instead of histograms.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (used only for logging).

    Attributes
    ----------
    bytes_sent_per_round : list[int]
        Total bytes received from all clients in each Flower round.
    bytes_saved_vs_bagging : list[int]
        Retrospective savings: ``tree_bytes − histogram_bytes`` after each
        tree round.  Positive when histograms are smaller than trees.
    trees_ : DiskTreeStore
        All trees accumulated across all tree rounds and all clients.
        Stored on disk to bound memory usage with large ensembles.
    thresholds_ : list of NDArray
        Global split thresholds (bin edges) per feature, set after round 1.
    """

    def __init__(self, n_bins: int = 32) -> None:
        self.n_bins = n_bins
        self.bytes_sent_per_round: list[int] = []
        self.bytes_saved_vs_bagging: list[int] = []
        self.trees_: DiskTreeStore = DiskTreeStore()
        self.thresholds_: list[NDArray] = []
        self.global_edges_: list[NDArray] = []
        self._histogram_bytes: int = 0

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Parameters | None:
        """Return empty initial parameters; clients hold their own bin edges."""
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
        """Aggregate histograms (round 1) or trees (round 2+).

        Parameters
        ----------
        server_round : int
            1 triggers histogram aggregation; any higher value triggers tree
            collection.
        results : list of (ClientProxy, FitRes)
            One entry per participating client.
        failures : list
            Ignored in the local runner.

        Returns
        -------
        parameters : Parameters
            Round 1: global split thresholds (one float64 array per feature).
            Round 2+: empty Parameters (trees are stored in ``self.trees_``).
        metrics : dict[str, Scalar]
            Empty dict.
        """
        round_bytes = sum(
            sum(len(t) for t in fit_res.parameters.tensors)
            for _, fit_res in results
        )
        self.bytes_sent_per_round.append(round_bytes)

        if server_round == 0:
            return self._aggregate_metadata(results, round_bytes)
        if server_round == 1:
            return self._aggregate_histograms(results, round_bytes)
        return self._aggregate_trees(results, round_bytes)

    def _aggregate_metadata(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        round_bytes: int,
    ) -> tuple[Parameters, dict[str, Scalar]]:
        """Aggregate DP-noised per-feature min/max from clients; compute global edges."""
        all_arrays = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        # Each client sends 2 * n_features arrays: [min_f0, max_f0, min_f1, max_f1, ...]
        n_features = len(all_arrays[0]) // 2

        global_min = np.array([
            min(client_arrays[f * 2][0] for client_arrays in all_arrays)
            for f in range(n_features)
        ])
        global_max = np.array([
            max(client_arrays[f * 2 + 1][0] for client_arrays in all_arrays)
            for f in range(n_features)
        ])
        global_max = np.maximum(global_max, global_min + 1e-6)

        self.global_edges_ = [
            np.linspace(global_min[f], global_max[f], self.n_bins + 1)
            for f in range(n_features)
        ]
        logger.info(
            "Metadata round: %d bytes from %d clients, %d features",
            round_bytes,
            len(results),
            n_features,
        )
        return ndarrays_to_parameters(self.global_edges_), {}

    def _aggregate_histograms(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        round_bytes: int,
    ) -> tuple[Parameters, dict[str, Scalar]]:
        """Sum per-feature histogram counts; extract shared bin edges."""
        self._histogram_bytes = round_bytes
        all_arrays = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        # Each client sends 2 * n_features arrays: [counts_f0, edges_f0, counts_f1, edges_f1, ...]
        n_pairs = len(all_arrays[0])
        n_features = n_pairs // 2

        global_counts: list[NDArray] = []
        global_edges: list[NDArray] = []

        for f in range(n_features):
            counts_idx = f * 2
            edges_idx = f * 2 + 1
            summed = np.sum(
                [client_arrays[counts_idx] for client_arrays in all_arrays], axis=0
            )
            global_counts.append(summed)
            # All clients use the same pre-computed global edges — just take the first
            global_edges.append(all_arrays[0][edges_idx])

        self.thresholds_ = global_edges
        logger.info(
            "Histogram round: %d bytes from %d clients, %d features aggregated",
            round_bytes,
            len(results),
            n_features,
        )
        return ndarrays_to_parameters(self.thresholds_), {}

    def _aggregate_trees(
        self,
        results: list[tuple[ClientProxy, FitRes]],
        round_bytes: int,
    ) -> tuple[Parameters, dict[str, Scalar]]:
        """Deserialise and collect all trees; compute communication savings."""
        for _, fit_res in results:
            for arr in parameters_to_ndarrays(fit_res.parameters):
                tree = pickle.loads(arr.tobytes())
                self.trees_.append(tree)

        # Savings = what bagging would have cost in round 1 (= tree bytes now)
        # minus what histogram actually cost. Positive ↔ histograms are cheaper.
        savings = round_bytes - self._histogram_bytes
        self.bytes_saved_vs_bagging = [savings]

        logger.info(
            "Tree round: %d bytes from %d clients, %d trees total, "
            "savings vs bagging = %+d bytes",
            round_bytes,
            len(results),
            len(self.trees_),
            savings,
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
