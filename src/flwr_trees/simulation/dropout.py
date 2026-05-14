from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ClientDropoutWrapper:
    """Wraps a partition list to simulate per-round client dropout.

    At each round, each client independently drops out with probability
    ``dropout_rate``.  At least ``min_clients`` partitions are always
    returned regardless of dropout outcomes.

    Reproducibility is achieved by seeding the RNG with
    ``random_state + round_idx``, so different rounds produce different
    (but deterministic) dropout patterns when ``random_state`` is set.

    Parameters
    ----------
    partitions : list of (NDArray, NDArray)
        Per-client (X_i, y_i) partitions to sample from.
    dropout_rate : float, default=0.2
        Probability that any individual client drops out in a given round.
        Must be in ``[0.0, 1.0)``.
    min_clients : int, default=2
        Minimum number of partitions always returned.  Must be ≤
        ``len(partitions)``.
    random_state : int or None, default=None
        Base seed for reproducibility.  ``None`` disables reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> parts = [(np.ones((10, 3)), np.ones(10)) for _ in range(5)]
    >>> wrapper = ClientDropoutWrapper(parts, dropout_rate=0.4, random_state=42)
    >>> surviving = wrapper.sample(round_idx=0)
    >>> 2 <= len(surviving) <= 5
    True
    """

    def __init__(
        self,
        partitions: list[tuple[NDArray, NDArray]],
        dropout_rate: float = 0.2,
        min_clients: int = 2,
        random_state: int | None = None,
    ) -> None:
        self.partitions = partitions
        self.dropout_rate = dropout_rate
        self.min_clients = min_clients
        self.random_state = random_state

    def sample(self, round_idx: int) -> list[tuple[NDArray, NDArray]]:
        """Return surviving client partitions for this round.

        Each partition survives independently with probability
        ``1 - dropout_rate``.  If fewer than ``min_clients`` survive,
        additional partitions are filled in from the dropped set (in their
        original order) until the minimum is met.

        Parameters
        ----------
        round_idx : int
            Zero-based round index.  Used with ``random_state`` to derive a
            per-round seed so each round has a distinct, reproducible dropout
            pattern.

        Returns
        -------
        surviving : list of (NDArray, NDArray)
            Partitions that survived dropout this round.  Length is always
            ≥ ``min_clients``.
        """
        seed = (
            None
            if self.random_state is None
            else self.random_state + round_idx
        )
        rng = np.random.default_rng(seed)

        keep: list[tuple[NDArray, NDArray]] = []
        dropped: list[tuple[NDArray, NDArray]] = []
        for partition in self.partitions:
            if rng.random() >= self.dropout_rate:
                keep.append(partition)
            else:
                dropped.append(partition)

        if len(keep) < self.min_clients:
            needed = self.min_clients - len(keep)
            keep.extend(dropped[:needed])
            logger.debug(
                "Round %d: dropout reduced survivors to %d; "
                "refilled %d from dropped to meet min_clients=%d",
                round_idx,
                len(keep) - needed,
                needed,
                self.min_clients,
            )

        logger.debug(
            "Round %d: %d/%d clients surviving (dropout_rate=%.2f)",
            round_idx,
            len(keep),
            len(self.partitions),
            self.dropout_rate,
        )
        return keep
