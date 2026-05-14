# flwr-trees

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-171%20passing-brightgreen)
![CI](https://github.com/MAUK9086/flwr-trees/actions/workflows/ci.yml/badge.svg)

**Federated learning with tree-based models via Flower.**

`flwr-trees` is a Python library that exposes scikit-learn-compatible federated estimators for Random Forests, XGBoost, and Gradient Boosted Trees, backed by the [Flower](https://flower.ai) federated learning framework. All estimators satisfy the scikit-learn `BaseEstimator` contract and pass `check_estimator()`, making them drop-in replacements inside any existing `sklearn.pipeline.Pipeline`.

> **Status:** Alpha (`v0.1.0`). The public API is stable for all implemented estimators.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Public API](#public-api)
- [Benchmark Results](#benchmark-results)
- [Progress](#progress)
- [Remaining Work](#remaining-work)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Overview

`flwr-trees` simulates federated training locally: data is partitioned among `n_clients` virtual clients, and each client trains on its local shard. The resulting models are aggregated according to the chosen FL strategy.

| Strategy | Description | Estimators |
|---|---|---|
| `FedForestBagging` | Each client trains an independent Random Forest; all trees are pooled at the server. | `FederatedRandomForestClassifier`, `FederatedRandomForestRegressor` |
| `FedForestCyclic` | A single XGBoost Booster is passed round-robin through all clients; each adds boosting rounds to it. | `FederatedXGBClassifier`, `FederatedXGBRegressor` |
| `FedHistogramAggregation` | **Research contribution** — clients send per-feature split histograms instead of full trees in round 1. Server aggregates histograms to determine global split thresholds. Clients then bin features and train RFs on the discretised data. Reduces round-1 communication by up to 99.8% vs bagging. | `FederatedHistogramRFClassifier`, `FederatedHistogramRFRegressor` |

Bagging and histogram strategies are implemented using real Flower `Strategy` / `NumPyClient` / `Parameters` types with an in-process orchestrator (no Ray or separate server process required). Setting `use_flower=True` activates the Flower code path; `use_flower=False` (the default) runs an equivalent plain Python loop for testing and `check_estimator()`.

---

## Installation

Requires **Python >= 3.11**.

> **PyPI release coming soon.** Install from source for now.
> ```bash
> pip install git+https://github.com/MAUK9086/flwr-trees.git
> ```

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# With optional DP (differential privacy) support
pip install -e ".[privacy]"
```

Core runtime dependencies:

| Package | Minimum version |
|---|---|
| `flwr` | 1.29.0 |
| `scikit-learn` | 1.8.0 |
| `xgboost` | 3.2.0 |
| `numpy` | 2.4.4 |

---

## Quick Start

### Federated Random Forest

```python
from flwr_trees import FederatedRandomForestClassifier

clf = FederatedRandomForestClassifier(
    n_estimators=100, n_clients=5, n_rounds=3,
    iid=False, alpha=0.5, random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

### Federated XGBoost (cyclic boosting)

```python
from flwr_trees import FederatedXGBClassifier

clf = FederatedXGBClassifier(
    n_estimators=50, n_clients=5, n_rounds=2,
    max_depth=6, learning_rate=0.1, random_state=42,
)
clf.fit(X_train, y_train)

# GPU acceleration (requires CUDA-capable GPU + XGBoost >= 2.0)
clf_gpu = FederatedXGBClassifier(
    n_estimators=50, n_clients=5, n_rounds=2,
    device="cuda",  # or "cuda:0" for a specific device
    random_state=42,
)
```

### Federated Gradient Boosted Trees

```python
from flwr_trees import FederatedGBTClassifier

clf = FederatedGBTClassifier(
    n_estimators=100, n_clients=5,
    max_depth=3, learning_rate=0.1, random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

### Communication-efficient federated RF (histogram aggregation)

```python
from flwr_trees import FederatedHistogramRFClassifier

clf = FederatedHistogramRFClassifier(
    n_estimators=50, n_clients=10, n_rounds=3,
    n_bins=32, iid=False, alpha=0.5,
    use_flower=True, random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Inspect communication cost
print(clf.strategy_.bytes_sent_per_round)   # [histogram_bytes, tree_bytes, ...]
print(clf.strategy_.bytes_saved_vs_bagging) # savings vs standard bagging
```

### Simulating client dropout

```python
from flwr_trees.simulation import simulate_clients, ClientDropoutWrapper

partitions = simulate_clients(X, y, n_clients=10, iid=True, random_state=0)
wrapper = ClientDropoutWrapper(partitions, dropout_rate=0.2, min_clients=3, random_state=42)

for round_idx in range(n_rounds):
    active = wrapper.sample(round_idx)  # reproducible per round
    # train on `active` partitions this round
```

### Differential privacy wrappers

```python
from flwr_trees.privacy import NoisyHistogram, DPTreeWrapper
from sklearn.tree import DecisionTreeClassifier

# Add Laplace noise to histogram counts before sending
noisy_hist = NoisyHistogram(epsilon=1.0, random_state=0)
private_counts = noisy_hist.apply(raw_counts)

# Wrap a decision tree to add Gaussian noise to leaf predictions
tree = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
dp_tree = DPTreeWrapper(tree, epsilon=1.0, random_state=0)
proba = dp_tree.predict_proba(X_test)  # noisy, sums to 1
```

---

## Architecture

```
flwr-trees/
├── src/
│   └── flwr_trees/
│       ├── __init__.py                       # Public API re-exports
│       ├── estimators/
│       │   ├── base.py                       # BaseFederatedTreeEstimator (ABC)
│       │   ├── rf.py                         # FederatedRandomForestClassifier/Regressor
│       │   ├── xgb.py                        # FederatedXGBClassifier/Regressor
│       │   ├── hist_rf.py                    # FederatedHistogramRFClassifier/Regressor
│       │   └── gbt.py                        # FederatedGBTClassifier/Regressor
│       ├── aggregation/
│       │   ├── bagging.py                    # FedForestBagging + FedForestBaggingClient
│       │   ├── cyclic.py                     # FedForestCyclic + XGBCyclicClient
│       │   └── histogram.py                  # FedHistogramAggregation + HistogramClient
│       ├── simulation/
│       │   ├── partitioning.py               # simulate_clients, partition_noniid
│       │   └── dropout.py                    # ClientDropoutWrapper
│       ├── privacy/
│       │   └── dp.py                         # NoisyHistogram, DPTreeWrapper
│       └── compat/
│           └── array_api.py                  # get_array_namespace, to_numpy
└── tests/
    ├── aggregation/                          # Strategy / client integration tests
    ├── compat/                               # Array API compliance tests
    ├── estimators/                           # Estimator unit + sklearn compliance tests
    ├── simulation/                           # Partitioning and dropout tests
    └── privacy/                             # DP noise tests
```

**`estimators/`** — Public sklearn-compatible estimators. Each branches at runtime between a local loop (`use_flower=False`) and a Flower-wired in-process loop (`use_flower=True`).

**`aggregation/`** — Flower `Strategy` and `NumPyClient` implementations. Models are serialised as pickle-encoded `uint8` NDArrays passed via Flower `Parameters`.

**`simulation/`** — Data partitioning utilities. `partition_noniid` implements Dirichlet-based heterogeneous splits. `ClientDropoutWrapper` simulates per-round client unavailability.

**`privacy/`** — Optional DP noise wrappers. `NoisyHistogram` adds Laplace noise to histogram counts. `DPTreeWrapper` adds Gaussian noise to tree leaf predictions.

**`compat/`** — Array API Standard utilities ensuring estimators accept NumPy, CuPy, and PyTorch tensors.

---

## Public API

### Estimators

| Class | Type | Strategy | Key fitted attributes |
|---|---|---|---|
| `FederatedRandomForestClassifier` | Classifier | Bagging | `estimators_: list[DecisionTreeClassifier]` |
| `FederatedRandomForestRegressor` | Regressor | Bagging | `estimators_: list[DecisionTreeRegressor]` |
| `FederatedXGBClassifier` | Classifier | Cyclic | `booster_: xgboost.Booster` |
| `FederatedXGBRegressor` | Regressor | Cyclic | `booster_: xgboost.Booster` |
| `FederatedHistogramRFClassifier` | Classifier | Histogram | `estimators_`, `thresholds_`, `strategy_` |
| `FederatedHistogramRFRegressor` | Regressor | Histogram | `estimators_`, `thresholds_`, `strategy_` |
| `FederatedGBTClassifier` | Classifier | Bagging (per-client GBT) | `estimators_: list[GradientBoostingClassifier]` |
| `FederatedGBTRegressor` | Regressor | Bagging (per-client GBT) | `estimators_: list[GradientBoostingRegressor]` |

All estimators share the base parameters `n_clients`, `n_rounds`, and `random_state` from `BaseFederatedTreeEstimator`. RF and Histogram estimators additionally accept `use_flower`.

> GPU acceleration via `device='cuda'` is supported for `FederatedXGBClassifier` and `FederatedXGBRegressor` only. `FederatedRandomForest`, `FederatedHistogramRF`, and `FederatedGBT` estimators use CPU-only sklearn backends.

### Aggregation strategies

| Class | Key attributes | Notes |
|---|---|---|
| `FedForestBagging` | `bytes_sent_per_round`, `trees_` | One entry per round; all clients contribute |
| `FedForestCyclic` | `bytes_sent_per_round`, `booster_` | One entry per Flower step (one client) |
| `FedHistogramAggregation` | `bytes_sent_per_round`, `bytes_saved_vs_bagging`, `trees_`, `thresholds_` | Round 1 = histogram exchange; rounds 2+ = tree collection |

### Simulation utilities

```python
from flwr_trees.simulation import simulate_clients, partition_noniid, ClientDropoutWrapper
```

### Privacy utilities

```python
from flwr_trees.privacy import NoisyHistogram, DPTreeWrapper
```

---

## Benchmark Results

Communication-efficiency comparison between `FedForestBagging` and `FedHistogramAggregation` across three datasets. Settings: `n_estimators=20, n_clients=5, n_rounds=3, n_bins=32`.

| Dataset | Method | Round-1 Bytes | Accuracy | Round-1 Savings |
|---|---|---|---|---|
| breast_cancer | FedForestBagging | 212 KB | 0.9737 | — |
| breast_cancer | FedHistogramAggregation | 116 KB | 0.9649 | 45.2% |
| synthetic_iid | FedForestBagging | 450 KB | 0.9100 | — |
| synthetic_iid | FedHistogramAggregation | 78 KB | 0.8800 | 82.8% |
| synthetic_noniid (alpha=0.3) | FedForestBagging | 400 KB | 0.6650 | — |
| synthetic_noniid (alpha=0.3) | FedHistogramAggregation | 78 KB | 0.7000 | 80.6% |

On the Adult dataset (48k samples, 14 features), `FedHistogramAggregation` achieves **99.8% round-1 savings** (108 KB vs 48 MB) with comparable accuracy to `FedForestBagging`.

The full benchmark is in `benchmarks/communication_benchmark.py` (Figure 1 data) and `benchmarks/real_world_benchmark.py` (multi-dataset comparison).

---

## Progress

| Phase | Description | Status | Tests |
|---|---|---|---|
| 1 | Core infrastructure: `BaseFederatedTreeEstimator`, array API compat, simulation/partitioning | Complete | 22 |
| 2 | `FederatedRandomForestClassifier` and `FederatedRandomForestRegressor` | Complete | 29 |
| 3 | Flower wiring for RF: `FedForestBagging`, `FedForestBaggingClient`, `use_flower=True` | Complete | 5 |
| 4 | `FederatedXGBClassifier` and `FederatedXGBRegressor` with `FedForestCyclic` strategy | Complete | 28 |
| 5 | `FedHistogramAggregation` + `FederatedHistogramRFClassifier/Regressor` (research contribution) | Complete | 11 |
| 6 | `FederatedGBTClassifier/Regressor`, `ClientDropoutWrapper`, `NoisyHistogram`, `DPTreeWrapper`, real-world benchmarks | Complete | 51 |

**Total: 171 tests passing** (`pytest tests/ -v`), including full `sklearn.utils.estimator_checks.check_estimator()` compliance for all eight estimators.

---

## Remaining Work

The following items remain for future development:

- **Full Array API compliance** — The `compat/` module handles NumPy and objects with `.numpy()` / `.get()` methods. Explicit CuPy and PyTorch tensor support requires additional testing.
- **Real distributed deployment** — All estimators are simulation-only (in-process). End-to-end testing with a real multi-process Flower deployment is not yet covered.
- **`FederatedGBTClassifier` with Flower path** — Current GBT uses local-only training. Adding `use_flower=True` support with `FedGBTBagging` strategy would enable communication tracking for GBT.
- **Advanced privacy composition** — Integration with `opacus` for training-level DP (currently only post-hoc noise wrappers exist).

---

## Development

```bash
# Create and activate virtual environment
uv sync
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # Linux / macOS

# Run the full test suite
pytest tests/ -v

# Verify sklearn estimator compliance for all estimators
python -c "
from sklearn.utils.estimator_checks import check_estimator
from flwr_trees import (
    FederatedRandomForestClassifier, FederatedRandomForestRegressor,
    FederatedXGBClassifier, FederatedXGBRegressor,
    FederatedHistogramRFClassifier,
    FederatedGBTClassifier, FederatedGBTRegressor,
)
for cls in [FederatedRandomForestClassifier, FederatedRandomForestRegressor,
            FederatedXGBClassifier, FederatedXGBRegressor,
            FederatedHistogramRFClassifier,
            FederatedGBTClassifier, FederatedGBTRegressor]:
    check_estimator(cls())
    print(f'{cls.__name__}: PASSED')
"

# Run communication benchmark (Figure 1 data)
python benchmarks/communication_benchmark.py

# Run real-world benchmark (skip large HIGGS download)
python benchmarks/real_world_benchmark.py --skip-higgs

# Lint
ruff check src/
```

---

## Citation

If you use flwr-trees in research, please cite:

```bibtex
@software{flwr_trees_2025,
  author = {Mauktik},
  title = {flwr-trees: Federated Learning with Tree-Based Models},
  year = {2026},
  url = {https://github.com/MAUK9086/flwr-trees},
}
```

---

## License

Apache License 2.0. See `pyproject.toml` for full classifier metadata.
