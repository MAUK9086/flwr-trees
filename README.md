# flwr-trees

**Federated learning with tree-based models via Flower.**

`flwr-trees` is a Python library that exposes scikit-learn-compatible federated estimators for Random Forests and XGBoost, backed by the [Flower](https://flower.ai) federated learning framework. All estimators satisfy the scikit-learn `BaseEstimator` contract and pass `check_estimator()`, making them drop-in replacements inside any existing `sklearn.pipeline.Pipeline`.

> **Status:** Alpha (`v0.1.0`). The public API is stable for the implemented estimators; modules marked *planned* below are under active development.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Public API](#public-api)
- [Progress](#progress)
- [Remaining Work](#remaining-work)
- [Development](#development)
- [License](#license)

---

## Overview

`flwr-trees` simulates federated training locally: data is partitioned among `n_clients` virtual clients, and each client trains on its local shard. The resulting models are aggregated according to the chosen FL strategy. Two aggregation strategies are currently implemented:

| Strategy | Description | Estimators |
|---|---|---|
| `FedForestBagging` | Each client trains an independent Random Forest; all trees are collected and pooled at the server. | `FederatedRandomForestClassifier`, `FederatedRandomForestRegressor` |
| `FedForestCyclic` | A single XGBoost Booster is passed round-robin through all clients; each client adds boosting rounds to it. | `FederatedXGBClassifier`, `FederatedXGBRegressor` |

Both strategies are implemented using real Flower `Strategy` / `NumPyClient` / `Parameters` types, with an in-process orchestrator that requires no Ray or separate server process. Setting `use_flower=True` on any estimator activates the Flower code path; `use_flower=False` (the default) runs an equivalent plain Python loop suited for unit testing and `check_estimator()`.

---

## Installation

Requires **Python ≥ 3.13**.

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Core runtime dependencies (pinned in `pyproject.toml`):

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
    n_estimators=100,
    n_clients=5,
    n_rounds=3,
    iid=False,          # Dirichlet non-IID partitioning
    alpha=0.5,
    random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

### Federated XGBoost (cyclic boosting)

```python
from flwr_trees import FederatedXGBClassifier

clf = FederatedXGBClassifier(
    n_estimators=50,    # boosting rounds added per client per step
    n_clients=5,
    n_rounds=2,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

### Activating the Flower code path

```python
from flwr_trees import FederatedXGBClassifier

clf = FederatedXGBClassifier(
    n_estimators=50,
    n_clients=5,
    n_rounds=2,
    use_flower=True,    # drives training via Flower Strategy / NumPyClient
    random_state=42,
)
clf.fit(X_train, y_train)

# Introspect communication cost (one entry per Flower step)
print(clf.strategy_.bytes_sent_per_round)
```

### Non-IID data partitioning

```python
from flwr_trees.simulation import simulate_clients, partition_noniid

# IID uniform split
partitions = simulate_clients(X, y, n_clients=5, iid=True, random_state=0)

# Dirichlet non-IID split (lower alpha = more skewed)
partitions = simulate_clients(X, y, n_clients=5, iid=False, alpha=0.3, random_state=0)
```

---

## Architecture

```
flwr-trees/
├── src/
│   └── flwr_trees/
│       ├── __init__.py                   # Public API re-exports
│       ├── estimators/
│       │   ├── base.py                   # BaseFederatedTreeEstimator (ABC)
│       │   ├── rf.py                     # FederatedRandomForestClassifier/Regressor
│       │   └── xgb.py                   # FederatedXGBClassifier/Regressor
│       ├── aggregation/
│       │   ├── bagging.py                # FedForestBagging + FedForestBaggingClient
│       │   ├── cyclic.py                 # FedForestCyclic + XGBCyclicClient
│       │   └── client_app.py             # Re-exports for Flower client app wiring
│       ├── simulation/
│       │   └── partitioning.py           # simulate_clients, partition_noniid
│       ├── compat/
│       │   └── array_api.py              # Array API utilities (get_array_namespace, to_numpy)
│       └── privacy/                      # Planned — DP wrappers
└── tests/
    ├── aggregation/                      # Strategy / client integration tests
    ├── compat/                           # Array API compliance tests
    ├── estimators/                       # Estimator unit + sklearn compliance tests
    └── simulation/                       # Partitioning tests
```

### Module responsibilities

**`estimators/`** — Public-facing sklearn-compatible estimators. Thin wrappers that delegate FL protocol logic to `aggregation/`. Each estimator branches at runtime between a plain local loop (`use_flower=False`) and a Flower-wired in-process loop (`use_flower=True`).

**`aggregation/`** — Flower `Strategy` and `NumPyClient` implementations. Serialises models as pickle-encoded `uint8` NDArrays passed through Flower `Parameters`. `FedForestBagging` accumulates all client trees; `FedForestCyclic` passes a single Booster through clients sequentially.

**`simulation/`** — Data partitioning utilities. `partition_noniid` implements Dirichlet-based heterogeneous partitioning; `simulate_clients` wraps both IID and non-IID splits behind a single API.

**`compat/`** — Array API Standard utilities ensuring estimators accept NumPy, CuPy, and PyTorch tensors without hardcoding `np.*` calls.

---

## Public API

### Estimators

| Class | Type | Strategy | Key fitted attribute |
|---|---|---|---|
| `FederatedRandomForestClassifier` | Classifier | Bagging | `estimators_: list[DecisionTreeClassifier]` |
| `FederatedRandomForestRegressor` | Regressor | Bagging | `estimators_: list[DecisionTreeRegressor]` |
| `FederatedXGBClassifier` | Classifier | Cyclic | `booster_: xgboost.Booster` |
| `FederatedXGBRegressor` | Regressor | Cyclic | `booster_: xgboost.Booster` |

All four estimators share the following constructor parameters via `BaseFederatedTreeEstimator`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clients` | `int` | `5` | Number of simulated FL clients |
| `n_rounds` | `int` | `1` | Number of complete cycles through all clients |
| `random_state` | `int \| None` | `None` | Seed for partitioning and model training |
| `use_flower` | `bool` | `False` | Activate the Flower Strategy / NumPyClient code path |

XGBoost estimators additionally accept `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `iid`, and `alpha`.

### Aggregation strategies

| Class | `bytes_sent_per_round` | Notes |
|---|---|---|
| `FedForestBagging` | One entry per round (all clients contribute) | Each entry = total serialised tree bytes from all clients that round |
| `FedForestCyclic` | One entry per Flower step (one client per step) | Length = `n_clients × n_rounds` |

### Simulation utilities

```python
from flwr_trees.simulation import simulate_clients, partition_noniid
```

---

## Progress

| Phase | Description | Status | Tests |
|---|---|---|---|
| 1 | Core infrastructure: `BaseFederatedTreeEstimator`, array API compat, simulation/partitioning | Complete | 22 |
| 2 | `FederatedRandomForestClassifier` and `FederatedRandomForestRegressor` (local path) | Complete | 29 |
| 3 | Flower wiring for Random Forest: `FedForestBagging`, `FedForestBaggingClient`, `use_flower=True` | Complete | 5 |
| 4 | `FederatedXGBClassifier` and `FederatedXGBRegressor` with `FedForestCyclic` strategy | Complete | 28 |

**Total: 85 tests passing** (`pytest tests/ -v`), including full `sklearn.utils.estimator_checks.check_estimator()` compliance for all four estimators.

---

## Remaining Work

The following items are planned for future phases:

### Estimators

- **`FederatedGradientBoostingClassifier` / `FederatedGradientBoostingRegressor`** — Federated wrappers around `sklearn.ensemble.GradientBoostingClassifier/Regressor`, using the same cyclic boosting pattern as the XGBoost estimators.

### Aggregation strategies

- **`FedHistogramAggregation`** *(research contribution)* — Instead of serialising and transmitting full trees, clients compute and share split histograms. The server aggregates histograms to determine global split points. This is expected to reduce per-round communication by a factor proportional to tree size. It is the primary novel contribution for the associated research paper and will require a dedicated benchmarking suite measuring communication bytes per round versus accuracy versus non-IID degree (alpha).

### Simulation utilities

- **`ClientDropoutWrapper`** — Wraps a client list to simulate random client dropout during training.

### Privacy

- **`DPTreeWrapper`** — Adds calibrated Gaussian noise to tree outputs for local differential privacy.
- **`NoisyHistogram`** — Applies DP noise to split histograms before transmission, designed for use with `FedHistogramAggregation`.

### Infrastructure

- **Full Array API compliance** — The `compat/` module currently handles NumPy and objects exposing `.numpy()` / `.get()` methods. Explicit CuPy and PyTorch tensor support requires additional testing.
- **Real distributed deployment** — All estimators are currently simulation-only (in-process). End-to-end testing with a real multi-process Flower deployment is not yet covered.
- **Benchmarking suite** — Communication cost (bytes per round) versus accuracy versus `alpha` (non-IID degree) for all strategies, required for the research paper.
- **Optional dependencies** — `opacus` (for DP wrappers) and `cupy` (for GPU array support) are not yet declared in `pyproject.toml`.

---

## Development

```bash
# Create and activate virtual environment
uv sync
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # Linux / macOS

# Run the full test suite
pytest tests/ -v

# Run a specific test module
pytest tests/estimators/test_xgb.py -v

# Verify sklearn estimator compliance
python -c "
from sklearn.utils.estimator_checks import check_estimator
from flwr_trees import (
    FederatedRandomForestClassifier,
    FederatedRandomForestRegressor,
    FederatedXGBClassifier,
    FederatedXGBRegressor,
)
for cls in [FederatedRandomForestClassifier, FederatedRandomForestRegressor,
            FederatedXGBClassifier, FederatedXGBRegressor]:
    check_estimator(cls())
    print(f'{cls.__name__}: PASSED')
"

# Lint
ruff check src/
```

---

## License

Apache License 2.0. See `pyproject.toml` for full classifier metadata.
