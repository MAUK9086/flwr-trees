# CLAUDE.md — flwr-trees

## What This Project Is
`flwr-trees` is a Python library for **federated learning with tree-based models** (Random Forests, Gradient Boosted Trees, XGBoost) using Flower (flwr) as the FL backend. It exposes a **scikit-learn-compatible API** so federated estimators drop into any existing sklearn Pipeline.

This is simultaneously:
- A production-usable open-source library (target: `pip install flwr-trees`)
- The implementation backing a research paper on communication-efficient federated tree aggregation

---

## Non-Negotiable Rules

### 1. sklearn API Compliance — ALWAYS
Every estimator MUST:
- Inherit from the correct sklearn mixin(s): `BaseEstimator`, `ClassifierMixin` or `RegressorMixin`
- Implement `fit(X, y)`, `predict(X)`, `score(X, y)`
- Implement `get_params(deep=True)` and `set_params(**params)` via `BaseEstimator`
- Pass `sklearn.utils.estimator_checks.check_estimator()` — run this before marking any estimator done
- Use `check_is_fitted(self)` at the start of `predict()` and `score()`
- Store all constructor arguments as attributes with the SAME name (sklearn convention): `self.n_estimators = n_estimators`
- Never mutate constructor arguments inside `fit()`
- Set `self.classes_` in classifiers after fitting

### 2. Array API Standard — REQUIRED
- All estimators must support the Python Array API Standard (https://data-apis.org/array-api/)
- Use `sklearn.utils._array_api.get_namespace` to detect array namespace
- Never hardcode `np.` calls in estimator logic — use the detected namespace
- Input arrays from NumPy, CuPy, and PyTorch tensors must all work

### 3. Type Hints — EVERYWHERE
- All public functions and methods must have full type hints
- Use `from __future__ import annotations` at top of every file
- Return types must be explicit — never omit them
- Use `numpy.typing.ArrayLike` and `numpy.typing.NDArray` for array inputs/outputs

### 4. Docstrings — NumPy Format
Every public class and function must have a NumPy-style docstring:
```python
def fit(self, X: ArrayLike, y: ArrayLike) -> "FederatedRandomForestClassifier":
    """Fit the federated random forest.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data for the local client.
    y : array-like of shape (n_samples,)
        Target labels.

    Returns
    -------
    self : FederatedRandomForestClassifier
        Fitted estimator.
    """
```

### 5. No Hardcoded Paths or Magic Numbers
- All configurable values go in `__init__` as parameters with defaults
- No hardcoded dataset paths, port numbers, or client counts outside simulation utils

### 6. Tests — Mandatory
- Every new class or function must have a corresponding test in `tests/`
- Test files mirror source structure: `src/flwr_trees/estimators/rf.py` → `tests/estimators/test_rf.py`
- Use `pytest` only — no `unittest`
- Each test must be runnable in under 30 seconds (use small n_estimators and few clients in tests)

### 7. No Jupyter Notebooks in src/
- Development notebooks go in `notebooks/` (not committed to main branch)
- Source code lives only in `src/flwr_trees/`

---

## Project Structure
```
flwr-trees/
├── src/
│   └── flwr_trees/
│       ├── __init__.py          # public API exports
│       ├── estimators/          # FederatedRF, FederatedXGB, FederatedGBT
│       ├── aggregation/         # FedForestBagging, FedHistogram (novel)
│       ├── simulation/          # simulate_clients, partition_noniid
│       ├── privacy/             # DP wrappers
│       └── compat/              # Array API compliance utilities
├── tests/
├── notebooks/                   # scratch only, not committed
├── pyproject.toml
├── CLAUDE.md                    # this file
└── README.md
```

---

## Module Responsibilities

### `estimators/`
Public-facing sklearn-compatible estimators. These are thin wrappers — they should not contain FL logic directly. They call into `aggregation/` for the FL protocol.

Classes to build:
- `FederatedRandomForestClassifier`
- `FederatedRandomForestRegressor`
- `FederatedXGBClassifier`
- `FederatedXGBRegressor`
- `FederatedGradientBoostingClassifier`
- `FederatedGradientBoostingRegressor`

### `aggregation/`
Flower strategies. This is where FL protocol logic lives.

- `FedForestBagging` — collect all trees from all clients each round
- `FedForestCyclic` — round-robin, one client trains per round
- `FedHistogramAggregation` — **novel contribution**: clients send split histograms not full trees

### `simulation/`
Utilities for local simulation of FL with multiple clients.

- `simulate_clients(X, y, n_clients, iid=True)` → list of (X_i, y_i)
- `partition_noniid(X, y, n_clients, alpha)` — Dirichlet-based non-IID partitioning
- `ClientDropoutWrapper` — wrap a client list to simulate random dropout

### `privacy/`
Optional differential privacy wrappers.

- `DPTreeWrapper` — adds calibrated noise to tree outputs
- `NoisyHistogram` — adds DP noise to histograms before sending

### `compat/`
Array API utilities shared across modules.

- `get_array_namespace(X)` — detect and return the array namespace
- `to_numpy(X)` — safe conversion from any array type to numpy

---

## Key Design Decisions

**Python, not C++**: Tree training is delegated to XGBoost and sklearn (which are already in C++/Cython). Our code is Python orchestration only. Do not write Cython or C extensions.

**Flower as transport**: All client-server communication goes through Flower. Do not implement custom networking.

**Simulation-first**: The library supports both real distributed deployment (via Flower) and local simulation (via `simulation/`). All estimators must work in simulation mode without requiring multiple processes.

**Non-IID is the default assumption**: Do not assume IID data. All experiments use Dirichlet partitioning (alpha=0.5 unless specified).

---

## Common Mistakes to Avoid

- ❌ Calling `np.array()` directly — use array namespace utilities
- ❌ Storing mutable default arguments in `__init__` (e.g. `def __init__(self, clients=[])`)
- ❌ Fitting inside `__init__` — fitting happens ONLY in `fit()`
- ❌ Using `print()` for logging — use Python's `logging` module
- ❌ Returning `None` from `fit()` — always return `self`
- ❌ Modifying `X` or `y` in-place inside any estimator method
- ❌ Breaking the Flower client/server interface (always test with `flwr.simulation.run_superlink`)
- ❌ Writing tests that require network access or GPU

---

## Running the Project

```bash
# Activate venv
.venv\Scripts\Activate.ps1       # Windows
source .venv/bin/activate         # Linux/Mac

# Install in editable mode
uv sync

# Run tests
pytest tests/ -v

# Run a specific test
pytest tests/estimators/test_rf.py -v

# Check sklearn compatibility
python -c "from sklearn.utils.estimator_checks import check_estimator; from flwr_trees.estimators import FederatedRandomForestClassifier; check_estimator(FederatedRandomForestClassifier())"

# Lint
ruff check src/
```

---

## Dependencies

Core: `flwr>=1.29.0`, `scikit-learn>=1.5.0`, `xgboost>=2.0.0`, `numpy>=1.26.0`
Optional: `opacus` (for DP), `cupy` (for GPU array support)
Dev: `pytest`, `ruff`, `mypy`, `jupyter`

---

## Paper Contribution Summary

The research contribution (for the paper) is `FedHistogramAggregation` in `aggregation/`:
- Instead of serializing and sending full trees across clients, clients compute and share **split histograms**
- Server aggregates histograms and determines global split points
- This reduces per-round communication proportional to tree size
- Measured as: communication bytes per round vs. accuracy vs. non-IID degree (alpha)

This is the one thing that must be benchmarked carefully and documented thoroughly.