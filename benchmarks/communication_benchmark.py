"""Communication efficiency benchmark: FedForestBagging vs FedHistogramAggregation.

Generates the data for Figure 1 of the flwr-trees research paper.

For a fair comparison, both strategies produce the same total number of trees
(n_clients * n_estimators * n_rounds).  The histogram strategy uses one extra
Flower round for the histogram exchange before tree training begins; this is the
round where the communication savings occur.

Metrics recorded per dataset:
  - Round-1 bytes: what each method sends in its first Flower round
    (histograms vs full trees -- this is the primary research metric)
  - Total bytes: across all Flower rounds
  - Test accuracy
  - Wall-clock training time

Usage
-----
    python benchmarks/communication_benchmark.py

Output
------
Prints a comparison table to stdout and saves full results to
``benchmarks/results.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split

from flwr_trees.estimators.hist_rf import FederatedHistogramRFClassifier
from flwr_trees.estimators.rf import FederatedRandomForestClassifier

_N_ESTIMATORS = 20
_N_CLIENTS = 5
_N_ROUNDS = 3          # tree-training rounds
_N_BINS = 32
_RANDOM_STATE = 42


def _datasets() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (name, X_train, X_test, y_train, y_test) tuples."""
    datasets = []

    # 1. Synthetic IID
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=_RANDOM_STATE
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)
    datasets.append(("synthetic_iid", Xtr, Xte, ytr, yte, True, 0.5))

    # 2. Breast cancer (real-world, binary)
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.2, random_state=_RANDOM_STATE
    )
    datasets.append(("breast_cancer", Xtr, Xte, ytr, yte, True, 0.5))

    # 3. Synthetic non-IID (Dirichlet alpha=0.3)
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=_RANDOM_STATE + 1
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)
    datasets.append(("synthetic_noniid", Xtr, Xte, ytr, yte, False, 0.3))

    return datasets


def _fmt_bytes(b: int) -> str:
    if b >= 1_000_000:
        return f"{b / 1_000_000:.2f} MB"
    if b >= 1_000:
        return f"{b / 1_000:.1f} KB"
    return f"{b} B"


def _run_bagging(Xtr, Xte, ytr, yte, iid, alpha) -> dict:
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
    return {
        "method": "FedForestBagging",
        "round1_bytes": clf.strategy_.bytes_sent_per_round[0],
        "total_bytes": sum(clf.strategy_.bytes_sent_per_round),
        "bytes_per_round": clf.strategy_.bytes_sent_per_round,
        "n_flower_rounds": len(clf.strategy_.bytes_sent_per_round),
        "accuracy": round(float(clf.score(Xte, yte)), 4),
        "time_s": round(elapsed, 3),
    }


def _run_histogram(Xtr, Xte, ytr, yte, iid, alpha) -> dict:
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
    return {
        "method": "FedHistogramAggregation",
        "round1_bytes": clf.strategy_.bytes_sent_per_round[0],
        "total_bytes": sum(clf.strategy_.bytes_sent_per_round),
        "bytes_per_round": clf.strategy_.bytes_sent_per_round,
        "n_flower_rounds": len(clf.strategy_.bytes_sent_per_round),
        "bytes_saved_vs_bagging": clf.strategy_.bytes_saved_vs_bagging,
        "accuracy": round(float(clf.score(Xte, yte)), 4),
        "time_s": round(elapsed, 3),
    }


def main() -> None:
    col = dict(ds=30, method=25, r1=16, total=14, acc=10, t=10)
    hdr = (
        f"{'Dataset':<{col['ds']}} {'Method':<{col['method']}} "
        f"{'Round-1 bytes':>{col['r1']}} {'Total bytes':>{col['total']}} "
        f"{'Accuracy':>{col['acc']}} {'Time (s)':>{col['t']}}"
    )
    sep = "-" * len(hdr)

    note = (
        f"Settings: n_estimators={_N_ESTIMATORS}, n_clients={_N_CLIENTS}, "
        f"n_rounds={_N_ROUNDS}, n_bins={_N_BINS}\n"
        "Note: histogram strategy has 1 extra Flower round for histogram exchange.\n"
        "      Round-1 savings = bagging_round1_bytes - histogram_round1_bytes.\n"
        "      This is the primary communication metric for the paper."
    )
    print(note)
    print()
    print(sep)
    print(hdr)
    print(sep)

    all_results: list[dict] = []

    for name, Xtr, Xte, ytr, yte, iid, alpha in _datasets():
        bag = _run_bagging(Xtr, Xte, ytr, yte, iid, alpha)
        hist = _run_histogram(Xtr, Xte, ytr, yte, iid, alpha)

        r1_savings = bag["round1_bytes"] - hist["round1_bytes"]
        r1_savings_pct = r1_savings / bag["round1_bytes"] * 100 if bag["round1_bytes"] else 0
        acc_delta = hist["accuracy"] - bag["accuracy"]

        def row(r: dict) -> str:
            return (
                f"{'':<{col['ds']}} {r['method']:<{col['method']}} "
                f"{_fmt_bytes(r['round1_bytes']):>{col['r1']}} "
                f"{_fmt_bytes(r['total_bytes']):>{col['total']}} "
                f"{r['accuracy']:>{col['acc']}.4f} "
                f"{r['time_s']:>{col['t']}.3f}"
            )

        print(f"{name:<{col['ds']}} {'(see rows below)'}")
        print(row(bag))
        print(row(hist))
        print(
            f"{'':>{col['ds'] + col['method'] + 1}}"
            f"  Round-1 savings: {_fmt_bytes(r1_savings)} ({r1_savings_pct:.1f}%)"
            f"  Accuracy delta: {acc_delta:+.4f}"
        )
        print()

        all_results.append(
            {
                "dataset": name,
                "iid": iid,
                "alpha": alpha,
                "bagging": bag,
                "histogram": hist,
                "round1_savings_bytes": r1_savings,
                "round1_savings_pct": round(r1_savings_pct, 2),
                "accuracy_delta": round(acc_delta, 4),
            }
        )

    print(sep)

    out_path = Path(__file__).parent / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
