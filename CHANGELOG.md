# Changelog

All notable changes to flwr-trees are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.1.0] - 2026-05-14

### Added

**Phase 1 - Core infrastructure**
- `BaseFederatedTreeEstimator` ABC with shared `n_clients`, `n_rounds`, `random_state` params
- `score()` dispatches to accuracy (classifiers) or R2 (regressors)
- Array API compat module (`compat/array_api.py`) -- `get_array_namespace`, `to_numpy`
- `simulate_clients` and `partition_noniid` (Dirichlet-based non-IID splits)

**Phase 2 - Random Forest estimators**
- `FederatedRandomForestClassifier` and `FederatedRandomForestRegressor`
- `iid` and `alpha` parameters for data partitioning control
- Full `check_estimator()` compliance

**Phase 3 - Flower wiring for RF**
- `FedForestBagging` Flower Strategy with `FedForestBaggingClient`
- `use_flower=True` activates real Flower in-process orchestration
- `bytes_sent_per_round` communication tracking

**Phase 4 - XGBoost estimators**
- `FederatedXGBClassifier` and `FederatedXGBRegressor`
- `FedForestCyclic` strategy -- single booster passed round-robin through clients
- GPU acceleration via `device="cuda"` parameter (requires XGBoost >= 2.0)

**Phase 5 - Histogram aggregation (research contribution)**
- `FedHistogramAggregation` strategy -- clients send split histograms instead of full trees
- `FederatedHistogramRFClassifier` and `FederatedHistogramRFRegressor`
- `feature_ranges` parameter for pre-agreed bin edges (zero extra data sharing)
- `metadata_epsilon` for differentially-private Round 0 range exchange
- `bytes_saved_vs_bagging` communication savings tracking
- Up to 99.8% round-1 communication reduction vs standard bagging (Adult dataset)

**Phase 6 - Additional estimators, simulation utilities, and privacy**
- `FederatedGBTClassifier` and `FederatedGBTRegressor` (sklearn GradientBoosting backend)
- `ClientDropoutWrapper` -- simulate per-round client unavailability
- `NoisyHistogram` -- Laplace noise applied to histogram counts before sending
- `DPTreeWrapper` -- Gaussian noise applied to tree leaf predictions
- Real-world benchmarks (`benchmarks/communication_benchmark.py`, `benchmarks/real_world_benchmark.py`)

**Phase 7 - Robustness and storage improvements**
- Pickle support in all estimators -- `__getstate__`/`__setstate__` drop transient `strategy_`
- `DiskTreeStore` -- lazy, disk-backed tree storage replacing in-memory lists (joblib serialization)
- Privacy-correct histogram bin edges -- DP Round 0 metadata exchange (Option A) and pre-agreed ranges (Option C)
- XGBoost GPU support via `device` parameter on both XGB estimators

---

[0.1.0]: https://github.com/MAUK9086/flwr-trees/releases/tag/v0.1.0
