"""Slow, literal transliterations of the original R and C algorithms.

These are deliberately naive line-by-line translations of MALDIquant's C
sources and caret's R sources.  They exist only so the fast vectorised
implementations in :mod:`pyrgcms` can be checked against something that is
obviously faithful to the original.  Never use them in production code.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "find_correlation_fast_r",
    "local_maxima_c",
    "moving_average_r",
    "near_zero_var_r",
    "snip_c",
]


def snip_c(y, iterations: int = 100, decreasing: bool = True) -> np.ndarray:
    """Transliteration of ``C_snip`` from MALDIquant's ``src/snip.c``."""
    xy = np.asarray(y, dtype=float).copy()
    n = len(xy)
    xo = np.zeros(n, dtype=float)
    k = iterations

    steps = range(k, 0, -1) if decreasing else range(1, k + 1)
    for i in steps:
        for j in range(i, n - i):
            a = xy[j]
            b = (xy[j - i] + xy[j + i]) / 2.0
            if b < a:
                a = b
            xo[j] = a
        for j in range(i, n - i):
            xy[j] = xo[j]

    return xy.copy()


def _window_max_idx(x, start: int, end: int) -> int:
    """Transliteration of ``windowMaxIdx``; ``end`` is inclusive."""
    m = start
    for i in range(start + 1, end + 1):
        if x[m] < x[i]:
            m = i
    return m


def local_maxima_c(y, half_window_size: int = 1) -> np.ndarray:
    """Transliteration of ``.localMaxima`` plus ``C_localMaxima``."""
    hws = half_window_size
    original = np.asarray(y, dtype=float)
    padded = np.concatenate([np.zeros(hws), original, np.zeros(hws)])

    n = len(padded)
    out = np.zeros(n, dtype=bool)
    q = hws
    window_size = q * 2

    m = _window_max_idx(padded, 0, window_size)
    out[m] = m == q

    i = window_size + 1
    left = i - window_size
    mid = (left + i) // 2
    while i < n:
        if m < left:
            m = _window_max_idx(padded, left, i)
        elif padded[i] > padded[m]:
            m = i
        if m == mid:
            out[m] = True
        i += 1
        mid += 1
        left += 1

    # R slices back to the unpadded region: `[(hws + 1):(length(y) - hws)]`.
    return out[hws : n - hws]


def moving_average_r(y, half_window_size: int = 2) -> np.ndarray:
    """Transliteration of MALDIquant's ``.movingAverage`` via ``.filter``."""
    x = np.asarray(y, dtype=float)
    n = len(x)
    hws = half_window_size
    width = 2 * hws + 1
    coefficients = np.full((width, width), 1.0 / width)

    out = np.empty(n, dtype=float)
    # stats::filter(sides = 2) -- centred, NA at both extrema.
    for i in range(hws, n - hws):
        out[i] = float(np.dot(coefficients[hws], x[i - hws : i + hws + 1][::-1]))

    # `y[seq_len(hws)] <- head(coef, hws) %*% head(x, w)`
    out[:hws] = coefficients[:hws] @ x[:width]
    # `y[(n - hws + 1):n] <- tail(coef, hws) %*% tail(x, w)`
    out[n - hws :] = coefficients[-hws:] @ x[-width:]

    # smoothIntensity routes through .transformIntensity, which zeroes negatives.
    return np.where(out < 0, 0.0, out)


def near_zero_var_r(x, freq_cut: float = 95 / 5, unique_cut: float = 10) -> list[int]:
    """Transliteration of ``caret:::nzv``."""
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    n_rows = values.shape[0]

    flagged = []
    for column in range(values.shape[1]):
        data = values[:, column]
        data = data[~np.isnan(data)]
        uniques, counts = np.unique(data, return_counts=True)

        if len(counts) <= 1:
            freq_ratio = 0.0
        else:
            sorted_counts = np.sort(counts)
            freq_ratio = float(sorted_counts[-1]) / float(sorted_counts[-2])

        n_unique = len(uniques)
        percent_unique = 100.0 * n_unique / n_rows
        zero_var = (n_unique == 1) or (len(data) == 0)

        if ((freq_ratio > freq_cut) and (percent_unique <= unique_cut)) or zero_var:
            flagged.append(column)
    return flagged


def find_correlation_fast_r(corr, cutoff: float = 0.9) -> list[int]:
    """Transliteration of ``caret:::findCorrelation_fast``."""
    x = np.asarray(corr, dtype=float)
    n = x.shape[0]
    average_corr = np.abs(x).mean(axis=0)

    # `which()` on the upper triangle, walking the matrix column-major.
    pairs = []
    for col in range(n):
        for row in range(n):
            if row < col and abs(x[row, col]) > cutoff:
                pairs.append((row, col))

    delete: list[int] = []
    for row, col in pairs:
        if average_corr[col] > average_corr[row]:
            delete.append(col)
    for row, col in pairs:
        if not average_corr[col] > average_corr[row]:
            delete.append(row)

    seen: list[int] = []
    for index in delete:
        if index not in seen:
            seen.append(index)
    return seen
