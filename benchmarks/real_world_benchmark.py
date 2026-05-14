"""Real-world benchmark: federated vs centralized classifiers across multiple datasets.

Compares five methods on up to four datasets:
  - FederatedRandomForestClassifier  (bagging, use_flower=True)
  - FederatedHistogramRFClassifier   (histogram, use_flower=True)
  - FederatedXGBClassifier           (cyclic, use_flower=True)
  - Centralized RandomForestClassifier (upper-bound baseline)
  - Centralized XGBClassifier          (upper-bound baseline)

For each dataset both IID and non-IID (alpha=0.5) partitioning are benchmarked.

Usage
-----
    python benchmarks/real_world_benchmark.py
    python benchmarks/real_world_benchmark.py --skip-higgs

Output
------
Prints a formatted table and saves results to ``benchmarks/real_world_results.json``.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from flwr_trees.estimators.hist_rf import FederatedHistogramRFClassifier
from flwr_trees.estimators.rf import FederatedRandomForestClassifier
from flwr_trees.estimators.xgb import FederatedXGBClassifier

_N_ESTIMATORS = 50
_N_CLIENTS = 10
_N_ROUNDS = 3
_N_BINS = 32
_RANDOM_STATE = 42
_TEST_SIZE = 0.2
_DATA_HOME = Path("~/.cache/sklearn_datasets").expanduser()
_DATA_HOME.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_breast_cancer() -> tuple[np.ndarray, np.ndarray, str]:
    data = load_breast_cancer()
    return data.data, data.target, "breast_cancer"


def _load_openml(name: str, version: int, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    ds = fetch_openml(name, version=version, as_frame=False, parser="liac-arff", data_home=str(_DATA_HOME))
    X = ds.data
    y = ds.target

    if max_rows is not None:
        X, y = X[:max_rows], y[:max_rows]

    # Convert string labels to integers
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

    # Preprocess: impute missing values, ordinally encode strings
    # Detect whether features contain strings (object dtype)
    if hasattr(X, "dtype") and X.dtype == object:
        pipe = Pipeline([
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("impute", SimpleImputer(strategy="median")),
        ])
    else:
        # Float array — just impute
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
    X = pipe.fit_transform(X).astype(np.float64)
    return X, y.astype(np.int64), name.replace("-", "_")


def _datasets(skip_higgs: bool) -> list[tuple[str, np.ndarray, np.ndarray]]:
    datasets = []

    try:
        X, y, name = _load_breast_cancer()
        datasets.append((name, X, y))
    except Exception as exc:
        print(f"[WARN] breast_cancer failed: {exc}")

    for ds_name, version, max_rows in [
        ("adult", 2, None),
        ("credit-g", 1, None),
        ("higgs", 1, 50_000),
    ]:
        if ds_name == "higgs" and skip_higgs:
            print("[INFO] Skipping HIGGS dataset (--skip-higgs flag set)")
            continue
        try:
            X, y, name = _load_openml(ds_name, version, max_rows)
            datasets.append((name, X, y))
        except Exception as exc:
            print(f"[WARN] {ds_name} failed to load: {exc}")

    return datasets


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def _run_federated_rf(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
    iid: bool, alpha: float,
) -> dict:
    clf = FederatedRandomForestClassifier(
        n_estimators=_N_ESTIMATORS,
        n_clients=_N_CLIENTS,
        n_rounds=_N_ROUNDS,
        iid=iid,
        alpha=alpha,
        use_flower=True,
        random_state=_RANDOM_STATE,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    ypred = clf.predict(Xte)
    return {
        "method": "FedRF",
        "round1_bytes": clf.strategy_.bytes_sent_per_round[0],
        "total_bytes": sum(clf.strategy_.bytes_sent_per_round),
        "accuracy": round(float(np.mean(ypred == yte)), 4),
        "f1_weighted": round(float(f1_score(yte, ypred, average="weighted", zero_division=0)), 4),
        "time_s": round(elapsed, 3),
    }


def _run_federated_hist_rf(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
    iid: bool, alpha: float,
) -> dict:
    clf = FederatedHistogramRFClassifier(
        n_estimators=_N_ESTIMATORS,
        n_clients=_N_CLIENTS,
        n_rounds=_N_ROUNDS,
        iid=iid,
        alpha=alpha,
        n_bins=_N_BINS,
        use_flower=True,
        random_state=_RANDOM_STATE,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    ypred = clf.predict(Xte)
    return {
        "method": "FedHistRF",
        "round1_bytes": clf.strategy_.bytes_sent_per_round[0],
        "total_bytes": sum(clf.strategy_.bytes_sent_per_round),
        "accuracy": round(float(np.mean(ypred == yte)), 4),
        "f1_weighted": round(float(f1_score(yte, ypred, average="weighted", zero_division=0)), 4),
        "time_s": round(elapsed, 3),
    }


def _run_federated_xgb(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
    iid: bool, alpha: float,
) -> dict:
    clf = FederatedXGBClassifier(
        n_estimators=_N_ESTIMATORS,
        n_clients=_N_CLIENTS,
        n_rounds=_N_ROUNDS,
        iid=iid,
        alpha=alpha,
        use_flower=True,
        random_state=_RANDOM_STATE,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    ypred = clf.predict(Xte)
    return {
        "method": "FedXGB",
        "round1_bytes": clf.strategy_.bytes_sent_per_round[0],
        "total_bytes": sum(clf.strategy_.bytes_sent_per_round),
        "accuracy": round(float(np.mean(ypred == yte)), 4),
        "f1_weighted": round(float(f1_score(yte, ypred, average="weighted", zero_division=0)), 4),
        "time_s": round(elapsed, 3),
    }


def _run_central_rf(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
) -> dict:
    clf = RandomForestClassifier(
        n_estimators=_N_ESTIMATORS * _N_CLIENTS,
        random_state=_RANDOM_STATE,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    ypred = clf.predict(Xte)
    return {
        "method": "CentralRF",
        "round1_bytes": None,
        "total_bytes": None,
        "accuracy": round(float(np.mean(ypred == yte)), 4),
        "f1_weighted": round(float(f1_score(yte, ypred, average="weighted", zero_division=0)), 4),
        "time_s": round(elapsed, 3),
    }


def _run_central_xgb(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
) -> dict:
    n_classes = len(np.unique(ytr))
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    clf = XGBClassifier(
        n_estimators=_N_ESTIMATORS,
        random_state=_RANDOM_STATE,
        objective=objective,
        eval_metric="logloss",
        verbosity=0,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    ypred = clf.predict(Xte)
    return {
        "method": "CentralXGB",
        "round1_bytes": None,
        "total_bytes": None,
        "accuracy": round(float(np.mean(ypred == yte)), 4),
        "f1_weighted": round(float(f1_score(yte, ypred, average="weighted", zero_division=0)), 4),
        "time_s": round(elapsed, 3),
    }


def _fmt_bytes(b: int | None) -> str:
    if b is None:
        return "N/A"
    if b >= 1_000_000:
        return f"{b / 1_000_000:.2f} MB"
    if b >= 1_000:
        return f"{b / 1_000:.1f} KB"
    return f"{b} B"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-world federated learning benchmark")
    parser.add_argument("--skip-higgs", action="store_true", help="Skip HIGGS dataset download")
    args = parser.parse_args()

    print(
        f"Settings: n_estimators={_N_ESTIMATORS}, n_clients={_N_CLIENTS}, "
        f"n_rounds={_N_ROUNDS}, n_bins={_N_BINS}, random_state={_RANDOM_STATE}"
    )

    col = dict(ds=20, partition=8, method=12, r1=14, total=12, acc=10, f1=10, t=10)
    hdr = (
        f"{'Dataset':<{col['ds']}} {'Partition':<{col['partition']}} "
        f"{'Method':<{col['method']}} {'Round-1 Bytes':>{col['r1']}} "
        f"{'Total Bytes':>{col['total']}} {'Accuracy':>{col['acc']}} "
        f"{'F1-weighted':>{col['f1']}} {'Time (s)':>{col['t']}}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    all_results: list[dict] = []

    for ds_name, X, y in _datasets(args.skip_higgs):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE
        )
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for iid_flag, partition_label, alpha in [(True, "IID", 0.5), (False, "nonIID", 0.5)]:
            row_results: list[dict] = []

            for runner, needs_iid in [
                (lambda: _run_federated_rf(Xtr, Xte, ytr, yte, iid_flag, alpha), True),
                (lambda: _run_federated_hist_rf(Xtr, Xte, ytr, yte, iid_flag, alpha), True),
                (lambda: _run_federated_xgb(Xtr, Xte, ytr, yte, iid_flag, alpha), True),
                (lambda: _run_central_rf(Xtr, Xte, ytr, yte), False),
                (lambda: _run_central_xgb(Xtr, Xte, ytr, yte), False),
            ]:
                try:
                    result = runner()
                    row_results.append(result)
                    r1 = _fmt_bytes(result["round1_bytes"])
                    total = _fmt_bytes(result["total_bytes"])
                    print(
                        f"{ds_name:<{col['ds']}} {partition_label:<{col['partition']}} "
                        f"{result['method']:<{col['method']}} {r1:>{col['r1']}} "
                        f"{total:>{col['total']}} {result['accuracy']:>{col['acc']}.4f} "
                        f"{result['f1_weighted']:>{col['f1']}.4f} "
                        f"{result['time_s']:>{col['t']}.3f}"
                    )
                except Exception as exc:
                    print(f"[WARN] {ds_name}/{partition_label}/{runner.__name__ if hasattr(runner, '__name__') else '?'} failed: {exc}")

            all_results.append({
                "dataset": ds_name,
                "n_samples": n_samples,
                "n_features": n_features,
                "partition": partition_label,
                "iid": iid_flag,
                "alpha": alpha,
                "results": row_results,
            })

        print(sep)

    out_path = Path(__file__).parent / "real_world_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
