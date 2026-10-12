"""Steps shared by both published pipelines.

The two R scripts duplicate their preprocessing almost verbatim; the shared
parts live here so the pipeline modules only describe what actually differs
(a train/test split versus nested resampling over all samples).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import NONZERO_RATIO_COLUMN, PeakAligner, nonzero_ratio
from .config import (
    CORRELATION_CUTOFF,
    NZV_FREQ_CUT,
    NZV_UNIQUE_CUT,
    PreprocessConfig,
)
from .feature_selection import drop_columns, find_correlation, near_zero_variance
from .io import list_sample_files, load_samples
from .preprocessing import preprocess_sample
from .species import SPECIES, kept_indices
from .species import labels as species_labels

logger = logging.getLogger(__name__)

__all__ = [
    "align_and_filter",
    "append_nonzero_ratio",
    "correlation_matrix",
    "filter_features",
    "load_peak_arrays",
]


def load_peak_arrays(
    data_dir: str | Path,
    config: PreprocessConfig | None = None,
    species: Sequence[str] = SPECIES,
) -> tuple[np.ndarray, list[str], list[Path]]:
    """Read every kept sample and run the chromatogram chain over it.

    Returns
    -------
    (peaks, labels, paths)
        ``peaks`` is ``(n_samples, n_scans, n_mz)``; ``labels`` are the class
        labels in the same order; ``paths`` are the source files.
    """
    cfg = config or PreprocessConfig()
    # Labels are positional, so a count mismatch would silently shift them.
    n_files = len(list_sample_files(data_dir))
    if n_files != len(species):
        raise ValueError(
            f"found {n_files} sample file(s) in {data_dir} but {len(species)} "
            "label(s); every file needs exactly one label"
        )

    samples, paths = load_samples(
        data_dir,
        keep=kept_indices(species),
        n_scans=cfg.n_scans,
        mz_min=cfg.mz_min,
        mz_max=cfg.mz_max,
    )

    logger.info("Preprocessing %d samples from %s", len(samples), data_dir)
    peaks = np.empty((len(samples), cfg.n_scans, cfg.n_mz_used), dtype=float)
    for position, sample in enumerate(samples):
        peaks[position] = preprocess_sample(sample, cfg)
        logger.debug(
            "Preprocessed %s (%d/%d)", paths[position].name, position + 1, len(samples)
        )

    return peaks, species_labels(species), paths


def append_nonzero_ratio(features: pd.DataFrame) -> pd.DataFrame:
    """Append the ``Perc_Nnonzero`` column, as the R scripts do via ``mutate``."""
    ratio = nonzero_ratio(features)
    out = features.copy()
    out[NONZERO_RATIO_COLUMN] = ratio.to_numpy()
    return out


#: Refuse to build a correlation matrix larger than this, in bytes (~8 GB).
MAX_CORRELATION_BYTES = 8_000_000_000


def correlation_matrix(
    features: pd.DataFrame, max_bytes: int = MAX_CORRELATION_BYTES
) -> np.ndarray:
    """Pearson correlation matrix, with constant columns handled explicitly.

    ``cor()`` in R yields ``NA`` for a zero-variance column, which would make
    ``findCorrelation`` fail.  Such columns are given a correlation of ``0``
    with everything else (and ``1`` with themselves), which leaves them
    untouched by the cutoff.

    The matrix is quadratic in the feature count, so an unexpectedly permissive
    near-zero-variance filter can ask for tens of gigabytes.  That is reported
    as a clear error rather than an opaque :class:`MemoryError`.
    """
    n_features = features.shape[1]
    required = n_features * n_features * 8
    if required > max_bytes:
        raise ValueError(
            f"a correlation matrix over {n_features} features needs "
            f"{required / 1e9:.1f} GB, above the {max_bytes / 1e9:.1f} GB limit; "
            "tighten the near-zero-variance filter, use fewer m/z channels, or "
            "raise max_bytes"
        )

    values = features.to_numpy(dtype=float)
    # Compare range, not std: the std of e.g. a repeated 0.1 is ~1e-17, and
    # corrcoef would then correlate two such columns at 1.0 through rounding.
    constant = np.ptp(values, axis=0) == 0

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.atleast_2d(np.corrcoef(values, rowvar=False))

    corr[constant, :] = 0.0
    corr[:, constant] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def filter_features(
    features: pd.DataFrame,
    freq_cut: float = NZV_FREQ_CUT,
    unique_cut: float = NZV_UNIQUE_CUT,
    correlation_cutoff: float = CORRELATION_CUTOFF,
) -> tuple[pd.DataFrame, dict]:
    """Drop near-zero-variance features, then highly correlated ones.

    Returns the filtered frame plus a report of how many features each stage
    removed.
    """
    n_start = features.shape[1]

    nzv_positions = near_zero_variance(features, freq_cut=freq_cut, unique_cut=unique_cut)
    reduced = drop_columns(features, nzv_positions)
    logger.info("Near-zero variance filter: %d -> %d features", n_start, reduced.shape[1])

    corr_positions = find_correlation(
        correlation_matrix(reduced), cutoff=correlation_cutoff
    )
    filtered = drop_columns(reduced, corr_positions)
    logger.info(
        "Correlation filter (cutoff %.2f): %d -> %d features",
        correlation_cutoff,
        reduced.shape[1],
        filtered.shape[1],
    )

    report = {
        "n_input": n_start,
        "n_after_near_zero_variance": int(reduced.shape[1]),
        "n_after_correlation": int(filtered.shape[1]),
        "removed_near_zero_variance": len(nzv_positions),
        "removed_correlated": len(corr_positions),
    }
    return filtered, report


def align_and_filter(
    train_peaks: np.ndarray,
    test_peaks: np.ndarray,
    mz_values: Sequence[int],
    cluster_height: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, PeakAligner]:
    """Fit peak bins and feature filters on training samples, apply them to test.

    Test features use the contiguous-span rule, as in the R test path.

    Returns
    -------
    (train_features, test_features, filter_report, aligner)
    """
    aligner = PeakAligner(mz_values=mz_values, cluster_height=cluster_height)
    train = append_nonzero_ratio(aligner.fit_transform(train_peaks))
    test = append_nonzero_ratio(aligner.transform(test_peaks, span="contiguous"))
    train, report = filter_features(train)
    return train, test[train.columns], report, aligner
