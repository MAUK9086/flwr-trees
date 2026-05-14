# Contributing to flwr-trees

Thanks for your interest in contributing. This document covers setup, workflow, and scope for v0.1.

---

## Setup

```bash
# Clone and install in editable mode
git clone https://github.com/MAUK9086/flwr-trees.git
cd flwr-trees
uv sync

# Activate venv
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # Linux / macOS
```

---

## Before opening a PR

Run these locally and make sure they all pass:

```bash
# Full test suite (171 tests)
pytest tests/ -v

# Lint
ruff check src/

# sklearn estimator compliance (for any estimator you modified)
python -c "
from sklearn.utils.estimator_checks import check_estimator
from flwr_trees import FederatedRandomForestClassifier
check_estimator(FederatedRandomForestClassifier())
print('PASSED')
"
```

---

## Project conventions

- All estimators must inherit from `BaseEstimator` + the appropriate sklearn mixin, pass `check_estimator()`, and use NumPy-style docstrings. See `CLAUDE.md` for the full non-negotiable rules.
- Tests mirror source structure: `src/flwr_trees/estimators/rf.py` -> `tests/estimators/test_rf.py`
- No `print()` -- use `logging`
- Full type hints with `from __future__ import annotations` at top of every file

---

## What is in scope for v0.1

- Bug fixes in existing estimators
- Improvements to communication tracking / DP noise
- Documentation improvements
- New datasets or benchmarks

## What is out of scope for v0.1

- Real distributed deployment (all estimators are simulation-only)
- New FL strategies beyond Bagging, Cyclic, and Histogram
- `use_flower=True` support for `FederatedGBTClassifier`

---

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include a minimal reproducible example.
