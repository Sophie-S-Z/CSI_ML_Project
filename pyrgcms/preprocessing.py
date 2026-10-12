"""Per-sample chromatogram preprocessing.

This is the Python equivalent of the ``for (i in 1:N)`` block that both R
scripts run over every sample:

1. keep the first ``n_mz_used`` m/z channels,
2. square-root transform to stabilise the variance,
3. moving-average smooth,
4. remove the SNIP baseline,
5. min-max normalise across the whole sample,
6. detect peaks as local maxima above ``snr`` times the MAD noise.

The result per sample is a dense ``(n_scans, n_mz_used)`` peak matrix holding
the peak intensity where a peak was found and ``0`` everywhere else.
"""

from __future__ import annotations

import numpy as np

from .config import PreprocessConfig
from .maldiquant import (
    detect_peaks,
    moving_average,
    remove_baseline,
    transform_intensity_sqrt,
)

__all__ = ["normalise_sample", "preprocess_sample"]


def _sample_to_array(sample, n_scans: int, n_mz_used: int) -> np.ndarray:
    """Coerce one sample to a ``(n_scans, n_mz_used)`` float array."""
    values = np.asarray(sample, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D sample, got {values.ndim} dimensions")
    if values.shape[0] < n_scans:
        raise ValueError(f"sample has {values.shape[0]} scans, need at least {n_scans}")
    if values.shape[1] < n_mz_used:
        raise ValueError(
            f"sample has {values.shape[1]} m/z channels, need at least {n_mz_used}"
        )
    return values[:n_scans, :n_mz_used]


def normalise_sample(processed: np.ndarray) -> np.ndarray:
    """Min-max normalise across the entire sample.

    The R code takes the minimum and maximum over *all* chromatograms of the
    sample at once (``Ma <- max(...)``, ``Mi <- min(...)``), not per m/z
    channel.  A completely flat sample would divide by zero, so that case
    returns all zeros instead of ``NaN``.
    """
    arr = np.asarray(processed, dtype=float)
    lo = float(arr.min())
    hi = float(arr.max())
    span = hi - lo
    if span == 0.0:
        return np.zeros_like(arr)
    return (arr - lo) / span


def preprocess_sample(sample, config: PreprocessConfig | None = None) -> np.ndarray:
    """Run the full chromatogram chain for one sample.

    Returns
    -------
    numpy.ndarray
        ``(n_scans, n_mz_used)`` array of peak intensities, zero off-peak.
    """
    cfg = config or PreprocessConfig()
    values = _sample_to_array(sample, cfg.n_scans, cfg.n_mz_used)

    chrom = transform_intensity_sqrt(values)
    chrom = moving_average(chrom, half_window_size=cfg.smooth_half_window)
    chrom = remove_baseline(
        chrom, iterations=cfg.snip_iterations, decreasing=cfg.snip_decreasing
    )
    chrom = normalise_sample(chrom)
    return detect_peaks(chrom, half_window_size=cfg.peak_half_window, snr=cfg.peak_snr)
