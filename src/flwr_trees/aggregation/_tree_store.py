from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Iterator


class DiskTreeStore:
    """Write-once, read-many disk buffer for fitted sklearn decision trees.

    Stores each tree as a joblib file in a temporary directory.  Iteration
    loads trees lazily (one at a time), keeping only one tree in memory at a
    time regardless of ensemble size.

    File naming: ``tree_{idx:06d}.joblib`` inside a ``tempfile.mkdtemp()``
    directory.  The directory is deleted on ``close()`` or garbage collection.

    Parameters
    ----------
    base_dir : str or None, default=None
        Directory under which the temp directory is created.  ``None`` uses
        the system default (``tempfile.gettempdir()``).

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=50, random_state=0)
    >>> tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    >>> store = DiskTreeStore()
    >>> store.append(tree)
    >>> len(store)
    1
    >>> loaded = store[0]
    >>> loaded.n_leaves == tree.n_leaves
    True
    >>> store.close()
    """

    def __init__(self, base_dir: str | None = None) -> None:
        self._dir: str = tempfile.mkdtemp(dir=base_dir)
        self._count: int = 0

    def _path(self, idx: int) -> str:
        return os.path.join(self._dir, f"tree_{idx:06d}.joblib")

    def append(self, tree: Any) -> None:
        """Serialize and store a single tree.

        Parameters
        ----------
        tree : fitted sklearn decision tree
            The tree to persist.
        """
        import joblib
        joblib.dump(tree, self._path(self._count))
        self._count += 1

    def extend(self, trees: list[Any]) -> None:
        """Serialize and store multiple trees.

        Parameters
        ----------
        trees : list of fitted sklearn decision trees
        """
        for tree in trees:
            self.append(tree)

    def __iter__(self) -> Iterator[Any]:
        """Yield trees one at a time (lazy loading)."""
        import joblib
        for i in range(self._count):
            yield joblib.load(self._path(i))

    def __len__(self) -> int:
        """Return the number of stored trees."""
        return self._count

    def __getitem__(self, idx: int) -> Any:
        """Load and return the tree at position *idx*.

        Parameters
        ----------
        idx : int
            Zero-based index.
        """
        import joblib
        if idx < 0 or idx >= self._count:
            raise IndexError(f"index {idx} out of range for store with {self._count} trees")
        return joblib.load(self._path(idx))

    def close(self) -> None:
        """Delete the temporary directory and all stored trees."""
        if os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize all trees to bytes so the store is picklable."""
        trees_bytes = []
        for i in range(self._count):
            with open(self._path(i), "rb") as f:
                trees_bytes.append(f.read())
        return {"trees_bytes": trees_bytes}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled store into a fresh temp directory."""
        self._dir = tempfile.mkdtemp()
        self._count = len(state["trees_bytes"])
        for i, data in enumerate(state["trees_bytes"]):
            with open(self._path(i), "wb") as f:
                f.write(data)
