"""Ports of the ``caret`` feature filters used by the R scripts.

Both filters follow *Applied Predictive Modeling* (Kuhn & Johnson, 2013) and
are transcribed from ``caret``'s own implementations so the selected feature
sets match the published pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CORRELATION_CUTOFF, NZV_FREQ_CUT, NZV_UNIQUE_CUT

__all__ = [
    "drop_columns",
    "find_correlation",
    "near_zero_variance",
    "near_zero_variance_metrics",
]


def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D input, got {arr.ndim} dimensions")
    return pd.DataFrame(arr)


def near_zero_variance_metrics(
    x, freq_cut: float = NZV_FREQ_CUT, unique_cut: float = NZV_UNIQUE_CUT
) -> pd.DataFrame:
    """Per-column diagnostics, the equivalent of ``nearZeroVar(saveMetrics=TRUE)``.

    Returns a frame with ``freq_ratio``, ``percent_unique``, ``zero_var`` and
    ``nzv`` columns, indexed by the input column names.
    """
    frame = _as_frame(x)
    n_rows = frame.shape[0]
    if n_rows == 0:
        raise ValueError("cannot evaluate near-zero variance on zero rows")

    freq_ratio = []
    n_unique = []
    all_missing = []
    for column in frame.columns:
        values = frame[column].dropna()
        counts = values.value_counts()
        n_unique.append(len(counts))
        all_missing.append(len(values) == 0)
        if len(counts) <= 1:
            # caret returns 0 here, which never exceeds freq_cut.
            freq_ratio.append(0.0)
        else:
            # `value_counts()` may hand back a read-only view, so sort a copy.
            ordered = np.sort(counts.to_numpy())
            freq_ratio.append(float(ordered[-1]) / float(ordered[-2]))

    freq_ratio_arr = np.asarray(freq_ratio, dtype=float)
    n_unique_arr = np.asarray(n_unique, dtype=float)
    percent_unique = 100.0 * n_unique_arr / n_rows
    zero_var = (n_unique_arr == 1) | np.asarray(all_missing, dtype=bool)
    nzv = ((freq_ratio_arr > freq_cut) & (percent_unique <= unique_cut)) | zero_var

    return pd.DataFrame(
        {
            "freq_ratio": freq_ratio_arr,
            "percent_unique": percent_unique,
            "zero_var": zero_var,
            "nzv": nzv,
        },
        index=frame.columns,
    )


def near_zero_variance(
    x, freq_cut: float = NZV_FREQ_CUT, unique_cut: float = NZV_UNIQUE_CUT
) -> np.ndarray:
    """Positions of near-zero-variance columns, like ``caret::nearZeroVar``.

    A column is flagged when it has a single distinct value, or when its
    most-common-to-second-most-common frequency ratio exceeds ``freq_cut``
    *and* its share of distinct values is at most ``unique_cut`` percent.
    """
    metrics = near_zero_variance_metrics(x, freq_cut=freq_cut, unique_cut=unique_cut)
    return np.flatnonzero(metrics["nzv"].to_numpy())


def _find_correlation_fast(corr: np.ndarray, cutoff: float) -> np.ndarray:
    """``caret:::findCorrelation_fast`` -- average correlations computed once."""
    average_corr = np.abs(corr).mean(axis=0)

    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows, cols = np.nonzero(upper & (np.abs(corr) > cutoff))
    # R's `which()` walks a matrix column-major; keep that pair ordering so the
    # returned index list matches caret's element for element.
    column_major = np.lexsort((rows, cols))
    rows, cols = rows[column_major], cols[column_major]

    discard_col = average_corr[cols] > average_corr[rows]
    flagged = np.concatenate([cols[discard_col], rows[~discard_col]])
    # `unique` after concatenation, preserving caret's column-then-row ordering.
    _, first_seen = np.unique(flagged, return_index=True)
    return flagged[np.sort(first_seen)]


def _find_correlation_exact(corr: np.ndarray, cutoff: float) -> np.ndarray:
    """``caret:::findCorrelation_exact`` -- averages recomputed at every step."""
    n_vars = corr.shape[0]
    if n_vars == 1:
        raise ValueError("only one variable given")
    if not np.allclose(corr, corr.T):
        raise ValueError("correlation matrix is not symmetric")

    abs_corr = np.abs(corr)
    with_nan_diag = abs_corr.copy()
    np.fill_diagonal(with_nan_diag, np.nan)

    # Order columns by decreasing mean absolute correlation; `np.argsort` is
    # stable with kind="stable", matching R's `order` tie-breaking.
    column_means = np.nanmean(with_nan_diag, axis=0)
    order = np.argsort(-column_means, kind="stable")
    reordered = abs_corr[np.ix_(order, order)]

    working = reordered.copy()
    np.fill_diagonal(working, np.nan)
    delete = np.zeros(n_vars, dtype=bool)

    for i in range(n_vars - 1):
        if not np.any(working[~np.isnan(working)] > cutoff):
            break
        if delete[i]:
            continue
        for j in range(i + 1, n_vars):
            if delete[i] or delete[j]:
                continue
            if reordered[i, j] > cutoff:
                mean_i = np.nanmean(working[i, :])
                mean_j = np.nanmean(np.delete(working, j, axis=0))
                if mean_i > mean_j:
                    delete[i] = True
                    working[i, :] = np.nan
                    working[:, i] = np.nan
                else:
                    delete[j] = True
                    working[j, :] = np.nan
                    working[:, j] = np.nan

    return order[np.flatnonzero(delete)]


def find_correlation(
    corr, cutoff: float = CORRELATION_CUTOFF, exact: bool | None = None
) -> np.ndarray:
    """Positions of columns to drop, like ``caret::findCorrelation``.

    When two columns correlate above ``cutoff``, the one with the larger mean
    absolute correlation is dropped.

    Parameters
    ----------
    corr:
        A square correlation matrix.
    cutoff:
        Absolute pairwise correlation cutoff.
    exact:
        ``True`` recomputes the mean absolute correlations after each removal.
        ``None`` reproduces caret's default, ``ncol(x) < 100``.
    """
    matrix = np.asarray(
        corr.to_numpy() if isinstance(corr, pd.DataFrame) else corr, dtype=float
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square correlation matrix, got {matrix.shape}")
    if np.isnan(matrix).any():
        raise ValueError("the correlation matrix has some missing values")

    use_exact = (matrix.shape[1] < 100) if exact is None else bool(exact)
    if use_exact:
        return _find_correlation_exact(matrix, cutoff)
    return _find_correlation_fast(matrix, cutoff)


def drop_columns(frame: pd.DataFrame, positions) -> pd.DataFrame:
    """Return ``frame`` without the columns at the given integer positions."""
    positions = np.asarray(list(positions), dtype=int)
    if positions.size == 0:
        return frame
    keep = np.setdiff1d(np.arange(frame.shape[1]), positions)
    if keep.size == 0:
        raise ValueError("the filter would remove every column")
    return frame.iloc[:, keep]
