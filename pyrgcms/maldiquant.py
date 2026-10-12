"""NumPy ports of the MALDIquant routines used by the original R scripts.

The R pipeline calls five MALDIquant primitives on every chromatogram:

===========================  ========================================
R call                       Python equivalent
===========================  ========================================
``transformIntensity(sqrt)`` :func:`transform_intensity_sqrt`
``smoothIntensity(...)``     :func:`moving_average`
``removeBaseline("SNIP")``   :func:`remove_baseline`
``detectPeaks("MAD", SNR)``  :func:`detect_peaks`
(internal ``.localMaxima``)  :func:`local_maxima`
===========================  ========================================

Each function is a faithful translation of MALDIquant (GPL-3), including its
edge handling and tie-breaking rules, but is vectorised so that a whole
``(n_scans, n_mz)`` block of chromatograms is processed in one call.  Every
function accepts a 1-D vector or a 2-D array; for 2-D input each *column* is
treated as an independent chromatogram, matching the ``(scan, m/z)`` layout of
the raw data.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "MAD_CONSTANT",
    "detect_peaks",
    "local_maxima",
    "mad",
    "moving_average",
    "remove_baseline",
    "snip_baseline",
    "transform_intensity_sqrt",
]

#: Scale factor R's ``stats::mad`` applies so the estimator is consistent for
#: normally distributed data.
MAD_CONSTANT: float = 1.4826


def _as_2d(x) -> tuple[np.ndarray, bool]:
    """Return ``x`` as a 2-D float array plus a flag saying it was 1-D."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"expected a 1-D or 2-D array, got {arr.ndim} dimensions")


def _restore(arr: np.ndarray, was_1d: bool) -> np.ndarray:
    return arr.reshape(-1) if was_1d else arr


def _check_half_window_size(half_window_size: int, n: int) -> int:
    """Port of MALDIquant's ``.stopIfNotIsValidHalfWindowSize``."""
    hws = int(half_window_size)
    if hws < 1:
        raise ValueError(f"half_window_size must be >= 1, got {half_window_size}")
    if 2 * hws + 1 > n:
        raise ValueError(
            f"half_window_size ({hws}) is too large for a signal of length {n}"
        )
    return hws


def transform_intensity_sqrt(intensity, on_negative: str = "clip") -> np.ndarray:
    """Variance-stabilising square-root transform.

    ``MALDIquant::transformIntensity(method="sqrt")`` drops any point whose
    transformed intensity is ``NaN`` *together with its mass*.  That would
    desynchronise the fixed scan grid the rest of this pipeline relies on, so
    negative inputs are handled explicitly instead.

    Parameters
    ----------
    intensity:
        Raw intensities, 1-D or ``(n_scans, n_mz)``.
    on_negative:
        ``"clip"`` (default) replaces negative inputs with ``0`` and warns;
        ``"raise"`` raises :class:`ValueError`.
    """
    if on_negative not in ("clip", "raise"):
        raise ValueError(f"on_negative must be 'clip' or 'raise', got {on_negative!r}")
    arr = np.asarray(intensity, dtype=float)
    negative = arr < 0
    if negative.any():
        if on_negative == "raise":
            raise ValueError(
                f"{int(negative.sum())} negative intensity value(s); "
                "sqrt is undefined for them"
            )
        warnings.warn(
            f"{int(negative.sum())} negative intensity value(s) clipped to 0 "
            "before the sqrt transform",
            RuntimeWarning,
            stacklevel=2,
        )
        arr = np.where(negative, 0.0, arr)
    return np.sqrt(arr)


def moving_average(intensity, half_window_size: int = 2) -> np.ndarray:
    """Unweighted two-sided moving average, MALDIquant edge handling included.

    MALDIquant filters the interior with a centred window of width
    ``2 * half_window_size + 1`` and then overwrites the first and last
    ``half_window_size`` points with the mean of, respectively, the first and
    the last full window.  Negative results are clipped to ``0`` because
    ``smoothIntensity`` routes through ``.transformIntensity``.
    """
    arr, was_1d = _as_2d(intensity)
    n = arr.shape[0]
    hws = _check_half_window_size(half_window_size, n)
    width = 2 * hws + 1

    # Interior: centred mean over `width` points, computed from a cumulative
    # sum so the cost does not depend on the window size.
    csum = np.zeros((n + 1, arr.shape[1]), dtype=float)
    np.cumsum(arr, axis=0, out=csum[1:])
    out = np.empty_like(arr)
    out[hws : n - hws] = (csum[width:] - csum[: n - width + 1]) / width

    # Edges: MALDIquant replaces them with the mean of the nearest full window.
    out[:hws] = arr[:width].mean(axis=0)
    out[n - hws :] = arr[n - width :].mean(axis=0)

    np.clip(out, 0.0, None, out=out)
    return _restore(out, was_1d)


def snip_baseline(
    intensity, iterations: int = 100, decreasing: bool = True
) -> np.ndarray:
    """Estimate a baseline with the SNIP algorithm (``C_snip`` in MALDIquant).

    Parameters
    ----------
    intensity:
        Signal, 1-D or ``(n_scans, n_mz)``.
    iterations:
        Number of clipping passes.
    decreasing:
        ``True`` walks the clipping window from ``iterations`` down to ``1``
        (MALDIquant's default), ``False`` walks it upwards.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    arr, was_1d = _as_2d(intensity)
    y = arr.astype(float, copy=True)
    n = y.shape[0]

    steps = range(iterations, 0, -1) if decreasing else range(1, iterations + 1)
    for i in steps:
        if 2 * i >= n:
            # The C loop `for (j = i; j < n - i; ++j)` is empty here.
            continue
        midpoint = (y[: n - 2 * i] + y[2 * i :]) / 2.0
        y[i : n - i] = np.minimum(y[i : n - i], midpoint)
    return _restore(y, was_1d)


def remove_baseline(
    intensity, iterations: int = 100, decreasing: bool = True
) -> np.ndarray:
    """Subtract the SNIP baseline.

    Mirrors ``MALDIquant::removeBaseline``: the result is *not* clipped at zero,
    because ``removeBaseline`` does not route through ``.transformIntensity``.
    """
    return np.asarray(intensity, dtype=float) - snip_baseline(
        intensity, iterations=iterations, decreasing=decreasing
    )


def mad(intensity):
    """Median absolute deviation, scaled like R's ``stats::mad``.

    For 2-D input the statistic is computed per column, matching MALDIquant's
    per-spectrum noise estimate.
    """
    arr, was_1d = _as_2d(intensity)
    centre = np.median(arr, axis=0)
    value = MAD_CONSTANT * np.median(np.abs(arr - centre), axis=0)
    return float(value[0]) if was_1d else value


def local_maxima(intensity, half_window_size: int = 1) -> np.ndarray:
    """Boolean mask of local maxima, matching MALDIquant's ``.localMaxima``.

    MALDIquant zero-pads the signal by ``half_window_size`` on both sides and
    then flags a point when it is the *left-most* maximum of the window centred
    on it.  That tie-breaking rule is reproduced here as

    ``x[j] > max(window left of j)`` **and** ``x[j] >= max(window right of j)``.
    """
    arr, was_1d = _as_2d(intensity)
    n = arr.shape[0]
    hws = int(half_window_size)
    if hws < 0:
        raise ValueError(f"half_window_size must be >= 0, got {half_window_size}")
    if hws == 0:
        # A zero-width window makes every point the maximum of its own window.
        return _restore(np.ones(arr.shape, dtype=bool), was_1d)

    pad = np.zeros((hws, arr.shape[1]), dtype=float)
    padded = np.concatenate([pad, arr, pad], axis=0)

    # For centre c (padded coordinates) the left window is padded[c - hws : c]
    # and the right window is padded[c + 1 : c + hws + 1].
    # win_max[k] == max(padded[k : k + hws])
    win_max = sliding_window_view(padded, hws, axis=0).max(axis=-1)
    centres = np.arange(hws, hws + n)
    left_max = win_max[centres - hws]
    right_max = win_max[centres + 1]
    centre_values = padded[centres]

    mask = (centre_values > left_max) & (centre_values >= right_max)
    return _restore(mask, was_1d)


def detect_peaks(intensity, half_window_size: int = 20, snr: float = 2.0) -> np.ndarray:
    """Peak intensities on the original grid, zero everywhere else.

    Port of ``MALDIquant::detectPeaks(method="MAD")``: a point is a peak when it
    is a local maximum *and* its intensity exceeds ``snr`` times the MAD noise
    estimate of its own chromatogram.

    The R script writes the detected peaks back onto the full scan grid
    (``v <- numeric(MM); v[scan.number] <- intensity(peaks)``); this function
    returns that dense representation directly.
    """
    arr, was_1d = _as_2d(intensity)
    noise = np.atleast_1d(mad(arr))
    is_above_noise = arr > (snr * noise)
    is_local_max = local_maxima(arr, half_window_size=half_window_size)
    peaks = np.where(is_above_noise & is_local_max, arr, 0.0)
    return _restore(peaks, was_1d)
