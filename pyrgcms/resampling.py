"""Stratified partitioning, the equivalent of ``caret::createDataPartition``."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np

from .config import RANDOM_SEED, TRAIN_FRACTION

__all__ = ["create_data_partition", "train_test_indices"]


def create_data_partition(
    y: Sequence,
    p: float = TRAIN_FRACTION,
    times: int = 1,
    random_state: int | np.random.Generator | None = RANDOM_SEED,
) -> list[np.ndarray]:
    """Stratified random sampling within each class.

    Each class contributes ``ceil(n_class * p)`` randomly chosen members, and
    the selected positions are returned sorted -- exactly what
    ``caret::createDataPartition`` does for a factor outcome.

    .. note::
       R and NumPy use different random number generators, so a given seed will
       not reproduce R's partition.  Only the *procedure* is identical.

    Returns
    -------
    list of numpy.ndarray
        ``times`` sorted arrays of zero-based positions.
    """
    labels = np.asarray(list(y))
    if labels.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {labels.shape}")
    if labels.size < 2:
        raise ValueError("y must have at least 2 data points")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie strictly between 0 and 1, got {p}")
    if times < 1:
        raise ValueError(f"times must be >= 1, got {times}")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    classes, counts = np.unique(labels, return_counts=True)
    singletons = classes[counts == 1]
    if singletons.size:
        warnings.warn(
            f"class(es) {list(singletons)} have a single record and will always "
            "be selected for the sample",
            RuntimeWarning,
            stacklevel=2,
        )

    partitions = []
    for _ in range(times):
        selected: list[np.ndarray] = []
        for label in classes:
            positions = np.flatnonzero(labels == label)
            if positions.size == 1:
                selected.append(positions)
                continue
            n_take = math.ceil(positions.size * p)
            selected.append(rng.choice(positions, size=n_take, replace=False))
        partitions.append(np.sort(np.concatenate(selected)))
    return partitions


def train_test_indices(
    y: Sequence,
    p: float = TRAIN_FRACTION,
    random_state: int | np.random.Generator | None = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning ``(train_positions, test_positions)``."""
    train = create_data_partition(y, p=p, times=1, random_state=random_state)[0]
    test = np.setdiff1d(np.arange(len(list(y))), train)
    return train, test
