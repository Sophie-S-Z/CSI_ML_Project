"""Peak alignment by hierarchical clustering of scan numbers.

Port of the ``for (i in 1:MZ)`` alignment block in both R scripts, which
implements the binning scheme of Tibshirani et al. (2004), *Sample
classification from protein mass spectrometry by peak probability contrasts*,
Bioinformatics 20(17):3034-44.

For each m/z channel the algorithm

1. collects every scan number at which *any* sample has a peak, repeated once
   per sample that peaks there,
2. runs complete-linkage hierarchical clustering on those scan numbers and cuts
   the dendrogram at a height of ``cluster_height`` scans,
3. turns each resulting cluster into one feature whose value is the maximum
   peak intensity a sample shows anywhere in that cluster.

The feature is named ``X<mean scan>.<m/z>``, matching R's ``make.names`` output
for ``"<mean scan>;<m/z>"``.

.. note::
   The R scripts deliberately treat training and test samples differently.
   Training features take the maximum over exactly the scan numbers that
   appeared in the cluster, while test features take the maximum over the whole
   contiguous span ``min(scans)..max(scans)``.  :meth:`PeakAligner.fit_transform`
   reproduces the former and :meth:`PeakAligner.transform` the latter, so the
   two code paths match the published pipeline.  Pass ``span="exact"`` to
   :meth:`~PeakAligner.transform` if you want the training rule everywhere.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .config import CLUSTER_HEIGHT, PreprocessConfig

__all__ = [
    "NONZERO_RATIO_COLUMN",
    "PeakAligner",
    "ScanCluster",
    "make_names",
    "nonzero_ratio",
]

#: Name of the extra feature holding the per-sample ratio of non-zero features.
NONZERO_RATIO_COLUMN = "Perc_Nnonzero"

_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9._]")
_RESERVED_WORDS = frozenset(
    {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "in",
    }
)


def make_names(name: str) -> str:
    """Port of R's ``make.names`` for a single name (``unique = FALSE``).

    Characters outside ``[A-Za-z0-9._]`` become ``"."``; names that are empty,
    reserved, start with a digit, or start with ``"."`` followed by a digit get
    an ``"X"`` prefix.
    """
    out = _INVALID_NAME_CHARS.sub(".", str(name))
    needs_prefix = (
        out == ""
        or out in _RESERVED_WORDS
        or out[0].isdigit()
        or (out[0] == "." and len(out) > 1 and out[1].isdigit())
    )
    return f"X{out}" if needs_prefix else out


@dataclass(frozen=True)
class ScanCluster:
    """One aligned peak bin: a set of scan numbers within a single m/z channel."""

    mz: int
    #: 1-based scan numbers that carried a peak in the fitted data, ascending.
    scans: tuple[int, ...]
    #: Frequency-weighted mean scan number, rounded half-to-even (R's ``round``).
    mean_scan: int

    @property
    def name(self) -> str:
        """Feature name, e.g. ``"X1234.57"``."""
        return make_names(f"{self.mean_scan};{self.mz}")

    @property
    def span(self) -> tuple[int, ...]:
        """Contiguous scan range ``min..max`` used for test-set features."""
        return tuple(range(min(self.scans), max(self.scans) + 1))


def _cluster_scan_numbers(scans: np.ndarray, height: float) -> list[np.ndarray]:
    """Group scan numbers into complete-linkage clusters cut at ``height``.

    ``scans`` are the distinct scan numbers that carried a peak.  The R code
    clusters the *expanded* vector (each scan repeated once per sample that
    peaked there); because every copy of a scan sits at
    distance 0 from its twins, complete linkage merges them first and then
    treats the group exactly like a single point, so clustering the distinct
    values yields the identical partition at any cut height above 0 -- and is
    dramatically cheaper.

    Returns clusters in order of first appearance, i.e. ascending scan number,
    matching R's ``unique(clusters)`` over an ascending input vector.
    """
    if scans.size == 0:
        return []
    if scans.size == 1:
        return [scans]

    distances = pdist(scans.reshape(-1, 1))
    tree = linkage(distances, method="complete")
    membership = fcluster(tree, t=height, criterion="distance")

    # Dicts keep insertion order and `scans` is ascending, so the groups come
    # out in first-appearance order, as R's `unique(clusters)` produces.
    groups: dict[int, list[int]] = {}
    for scan, label in zip(scans, membership, strict=True):
        groups.setdefault(int(label), []).append(int(scan))
    return [np.array(group, dtype=int) for group in groups.values()]


class PeakAligner:
    """Learn peak bins from one set of samples and apply them to another.

    Parameters
    ----------
    mz_values:
        m/z channels, in the column order of the peak matrices.
    cluster_height:
        Dendrogram cut height, in scan numbers.
    """

    def __init__(
        self,
        mz_values: Sequence[int] | None = None,
        cluster_height: float = CLUSTER_HEIGHT,
    ) -> None:
        if cluster_height <= 0:
            raise ValueError(f"cluster_height must be > 0, got {cluster_height}")
        self.mz_values = list(
            mz_values if mz_values is not None else PreprocessConfig().mz_values
        )
        self.cluster_height = float(cluster_height)
        self.clusters_: list[ScanCluster] | None = None

    # -- fitting --------------------------------------------------------------

    def _check_input(self, peaks: np.ndarray) -> np.ndarray:
        arr = np.asarray(peaks, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"expected a (n_samples, n_scans, n_mz) array, got {arr.ndim} dimensions"
            )
        if arr.shape[2] != len(self.mz_values):
            raise ValueError(
                f"array has {arr.shape[2]} m/z channels but {len(self.mz_values)} "
                "m/z values were configured"
            )
        return arr

    def fit(self, peaks: np.ndarray) -> PeakAligner:
        """Derive the peak bins from a ``(n_samples, n_scans, n_mz)`` array."""
        arr = self._check_input(peaks)
        n_scans = arr.shape[1]
        scan_numbers = np.arange(1, n_scans + 1)

        clusters: list[ScanCluster] = []
        empty_channels: list[int] = []
        for mz_index, mz in enumerate(self.mz_values):
            channel = arr[:, :, mz_index]
            counts = (channel > 0).sum(axis=0)
            present = counts > 0
            if not present.any():
                empty_channels.append(mz)
                continue

            groups = _cluster_scan_numbers(scan_numbers[present], self.cluster_height)
            for group in groups:
                weights = counts[group - 1]  # scan numbers are 1-based
                weighted_mean = float(np.dot(group, weights) / weights.sum())
                clusters.append(
                    ScanCluster(
                        mz=int(mz),
                        scans=tuple(int(s) for s in group),
                        mean_scan=int(np.round(weighted_mean)),
                    )
                )

        if empty_channels:
            warnings.warn(
                f"{len(empty_channels)} m/z channel(s) had no peaks in any sample "
                f"and produced no features (first few: {empty_channels[:5]})",
                RuntimeWarning,
                stacklevel=2,
            )
        if not clusters:
            raise ValueError("no peaks were detected in any m/z channel")

        # Two clusters in the same channel occupy disjoint scan ranges, so their
        # rounded means cannot collide -- but a collision would silently produce
        # duplicate column labels, so make it loud rather than trusting that.
        name_counts = Counter(cluster.name for cluster in clusters)
        duplicates = {name for name, count in name_counts.items() if count > 1}
        if duplicates:
            raise ValueError(
                f"aligned features have duplicate names: {sorted(duplicates)[:5]}; "
                "lower cluster_height so the bins stay distinct"
            )

        self.clusters_ = clusters
        return self

    # -- applying -------------------------------------------------------------

    @property
    def feature_names_(self) -> list[str]:
        """Names of the aligned features, in column order."""
        return [cluster.name for cluster in self._require_fitted()]

    def _require_fitted(self) -> list[ScanCluster]:
        """Return the fitted clusters, or explain that there are none yet."""
        if self.clusters_ is None:
            raise RuntimeError("PeakAligner has not been fitted yet; call fit() first")
        return self.clusters_

    def _apply(self, peaks: np.ndarray, span: str) -> pd.DataFrame:
        if span not in ("exact", "contiguous"):
            raise ValueError(f"span must be 'exact' or 'contiguous', got {span!r}")
        clusters = self._require_fitted()
        arr = self._check_input(peaks)
        n_scans = arr.shape[1]
        mz_index = {mz: i for i, mz in enumerate(self.mz_values)}

        columns = np.empty((arr.shape[0], len(clusters)), dtype=float)
        for position, cluster in enumerate(clusters):
            scans = cluster.scans if span == "exact" else cluster.span
            if max(scans) > n_scans:
                raise ValueError(
                    f"cluster {cluster.name} needs scan {max(scans)} but the data "
                    f"only has {n_scans} scans"
                )
            indices = np.asarray(scans, dtype=int) - 1  # scan numbers are 1-based
            columns[:, position] = arr[:, indices, mz_index[cluster.mz]].max(axis=1)

        return pd.DataFrame(columns, columns=self.feature_names_)

    def fit_transform(self, peaks: np.ndarray) -> pd.DataFrame:
        """Fit the bins and apply them with the *training* rule (exact scans)."""
        return self.fit(peaks)._apply(peaks, span="exact")

    def transform(self, peaks: np.ndarray, span: str = "contiguous") -> pd.DataFrame:
        """Apply fitted bins to new samples.

        ``span="contiguous"`` (the default, and what the R test path does) takes
        the maximum over ``min(scans)..max(scans)``; ``span="exact"`` restricts
        it to the scans seen during fitting.
        """
        return self._apply(peaks, span=span)


def nonzero_ratio(features: pd.DataFrame) -> pd.Series:
    """Per-sample ratio of strictly positive feature values.

    Equivalent to the R helper ``lg``; the denominator is the feature count
    *before* this column is appended.
    """
    if features.shape[1] == 0:
        raise ValueError("cannot compute a non-zero ratio over zero features")
    values = features.to_numpy(dtype=float)
    return pd.Series(
        (values > 0).sum(axis=1) / values.shape[1],
        index=features.index,
        name=NONZERO_RATIO_COLUMN,
    )
