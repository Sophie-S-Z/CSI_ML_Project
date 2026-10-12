"""Tunable constants for the pyr-GC-MS preprocessing and modelling pipelines.

Every value here mirrors a literal that appears in the original R scripts
(``reference/r/*.R``).  They are collected in one place so the pipelines stay
free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Raw data geometry -------------------------------------------------------

#: Lowest mass-to-charge ratio present in a raw sample file.
MZ_MIN: int = 50
#: Highest mass-to-charge ratio present in a raw sample file (``NN`` in R).
MZ_MAX: int = 700
#: Number of chromatogram scans retained per sample (``MM`` in R).
N_SCANS: int = 3240
#: Number of leading m/z channels actually used for modelling (``MZ`` in R).
N_MZ_USED: int = 151

# --- MALDIquant chromatogram processing --------------------------------------

#: Half window size of the moving-average smoother.
SMOOTH_HALF_WINDOW: int = 5
#: Iteration count of the SNIP baseline estimator (MALDIquant default).
SNIP_ITERATIONS: int = 100
#: Whether SNIP uses a decreasing clipping window (MALDIquant default).
SNIP_DECREASING: bool = True
#: Half window size used when searching for local maxima.
PEAK_HALF_WINDOW: int = 20
#: Signal-to-noise ratio a local maximum must exceed to count as a peak.
PEAK_SNR: float = 4.0

# --- Peak alignment (Tibshirani et al., 2004) --------------------------------

#: Cut height for the complete-linkage dendrogram over scan numbers.
CLUSTER_HEIGHT: float = 20.0

# --- Feature filtering -------------------------------------------------------

#: ``caret::nearZeroVar`` frequency-ratio cutoff.
NZV_FREQ_CUT: float = 95.0 / 5.0
#: ``caret::nearZeroVar`` percent-unique cutoff.
NZV_UNIQUE_CUT: float = 10.0
#: ``caret::findCorrelation`` absolute-correlation cutoff.
CORRELATION_CUTOFF: float = 0.85

# --- Resampling / modelling --------------------------------------------------

#: Fraction of samples assigned to the training partition.
TRAIN_FRACTION: float = 0.7
#: Global RNG seed (``set.seed(99)`` in R).
RANDOM_SEED: int = 99
#: Folds of the (inner) tuning cross-validation.
INNER_FOLDS: int = 10
#: Folds of the outer cross-validation used for nested resampling.
OUTER_FOLDS: int = 5
#: Number of random-search evaluations per tuning run.
TUNING_EVALS: int = 20
#: Inclusive search range for ``mtry`` / ``max_features``.
MTRY_RANGE: tuple[int, int] = (1, 2000)
#: Inclusive search range for ``num.trees`` / ``n_estimators``.
NUM_TREES_RANGE: tuple[int, int] = (1, 8000)
#: Class label treated as the positive class for AUC.
POSITIVE_CLASS: str = "B"

#: Hyper-parameters reported in the paper, reused for the final refit.
PAPER_MTRY: int = 388
PAPER_NUM_TREES: int = 5096


@dataclass(frozen=True)
class PreprocessConfig:
    """Bundle of every knob used by :mod:`pyrgcms.preprocessing`."""

    n_scans: int = N_SCANS
    n_mz_used: int = N_MZ_USED
    mz_min: int = MZ_MIN
    mz_max: int = MZ_MAX
    smooth_half_window: int = SMOOTH_HALF_WINDOW
    snip_iterations: int = SNIP_ITERATIONS
    snip_decreasing: bool = SNIP_DECREASING
    peak_half_window: int = PEAK_HALF_WINDOW
    peak_snr: float = PEAK_SNR

    def __post_init__(self) -> None:
        available = self.mz_max - self.mz_min + 1
        if self.n_mz_used > available:
            raise ValueError(
                f"n_mz_used ({self.n_mz_used}) exceeds the {available} m/z "
                f"channel(s) between {self.mz_min} and {self.mz_max}"
            )
        if self.n_mz_used < 1:
            raise ValueError(f"n_mz_used must be >= 1, got {self.n_mz_used}")
        if self.n_scans < 1:
            raise ValueError(f"n_scans must be >= 1, got {self.n_scans}")

    @property
    def mz_values(self) -> list[int]:
        """The m/z channels retained for modelling."""
        return list(range(self.mz_min, self.mz_min + self.n_mz_used))


@dataclass(frozen=True)
class ModelConfig:
    """Bundle of every knob used by :mod:`pyrgcms.model`."""

    seed: int = RANDOM_SEED
    inner_folds: int = INNER_FOLDS
    outer_folds: int = OUTER_FOLDS
    tuning_evals: int = TUNING_EVALS
    mtry_range: tuple[int, int] = MTRY_RANGE
    num_trees_range: tuple[int, int] = NUM_TREES_RANGE
    positive_class: str = POSITIVE_CLASS
    n_jobs: int = -1
