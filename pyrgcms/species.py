"""Sample origin labels for the 141 raw pyr-GC-MS files.

The original R scripts hard-code the label vector and drop every ``"C"``
sample before modelling, leaving 134 samples (75 abiotic, 59 biotic).  The
labels are positional: entry *i* describes the *i*-th file returned by
``list.files()``, so the ordering convention in :mod:`pyrgcms.io` matters.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "ABIOTIC",
    "BIOTIC",
    "EXCLUDED",
    "SPECIES",
    "excluded_indices",
    "kept_indices",
    "labels",
]

#: Label of the abiotic class.
ABIOTIC = "A"
#: Label of the biotic class (the positive class for AUC).
BIOTIC = "B"
#: Label of the samples excluded from the analysis.
EXCLUDED = "C"


def _rep(value: str, times: int) -> list[str]:
    return [value] * times


#: Positional origin label of every raw file, transcribed from the R scripts.
SPECIES: tuple[str, ...] = tuple(
    _rep("A", 5)
    + ["B", "A", "A", "B", "B", "A", "C"]
    + _rep("B", 5)
    + _rep("C", 4)
    + _rep("B", 4)
    + ["A"]
    + _rep("B", 11)
    + ["A", "A", "B", "A", "A", "B", "B", "A"]
    + _rep("B", 3)
    + ["A", "B"]
    + _rep("A", 6)
    + _rep("B", 3)
    + ["A", "A", "B", "B"]
    + _rep("A", 4)
    + ["B"]
    + _rep("A", 5)
    + ["B"]
    + _rep("A", 10)
    + _rep("B", 4)
    + _rep("A", 5)
    + _rep("B", 3)
    + _rep("A", 3)
    + _rep("B", 4)
    + _rep("A", 8)
    + ["B", "A", "A", "B", "C", "C", "A"]
    + _rep("B", 3)
    + ["A", "B", "B", "A", "B"]
    + _rep("A", 3)
    + ["B", "A"]
    + _rep("A", 7)
    + ["B", "A", "B"]
)


def excluded_indices(species: Sequence[str] = SPECIES) -> list[int]:
    """Zero-based positions of the samples dropped before modelling."""
    return [i for i, label in enumerate(species) if label == EXCLUDED]


def kept_indices(species: Sequence[str] = SPECIES) -> list[int]:
    """Zero-based positions of the samples kept for modelling."""
    return [i for i, label in enumerate(species) if label != EXCLUDED]


def labels(species: Sequence[str] = SPECIES) -> list[str]:
    """Class labels of the kept samples, in file order."""
    return [label for label in species if label != EXCLUDED]
