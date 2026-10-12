"""The caret filter ports must match the literal R transliterations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrgcms.feature_selection import (
    drop_columns,
    find_correlation,
    near_zero_variance,
    near_zero_variance_metrics,
)

from .reference_impl import find_correlation_fast_r, near_zero_var_r


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(4242)


# --- near zero variance -------------------------------------------------------


def test_constant_column_is_flagged():
    frame = pd.DataFrame({"constant": [1.0] * 20, "varied": np.arange(20.0)})
    assert near_zero_variance(frame).tolist() == [0]


def test_lopsided_column_is_flagged():
    # 99 zeros and one 1: frequency ratio 99 > 19, unique share 2% <= 10%.
    values = np.zeros(100)
    values[0] = 1.0
    frame = pd.DataFrame({"lopsided": values, "spread": np.arange(100.0)})
    assert near_zero_variance(frame).tolist() == [0]


def test_balanced_binary_column_survives():
    frame = pd.DataFrame({"balanced": np.tile([0.0, 1.0], 50)})
    assert near_zero_variance(frame).size == 0


def test_metrics_report_the_expected_diagnostics():
    values = np.zeros(100)
    values[0] = 1.0
    metrics = near_zero_variance_metrics(pd.DataFrame({"lopsided": values}))

    assert metrics.loc["lopsided", "freq_ratio"] == pytest.approx(99.0)
    assert metrics.loc["lopsided", "percent_unique"] == pytest.approx(2.0)
    assert not metrics.loc["lopsided", "zero_var"]
    assert metrics.loc["lopsided", "nzv"]


def test_constant_column_reports_zero_freq_ratio():
    metrics = near_zero_variance_metrics(pd.DataFrame({"c": [3.0] * 10}))
    assert metrics.loc["c", "freq_ratio"] == 0.0
    assert metrics.loc["c", "zero_var"]


def test_near_zero_variance_matches_reference(rng):
    block = np.column_stack(
        [
            rng.normal(size=60),
            np.zeros(60),
            np.repeat([0.0, 1.0], 30),
            np.where(np.arange(60) == 0, 1.0, 0.0),
            rng.integers(0, 3, size=60).astype(float),
        ]
    )
    np.testing.assert_array_equal(near_zero_variance(block), near_zero_var_r(block))


def test_near_zero_variance_matches_reference_on_sparse_data(rng):
    # Sparse peak-like data, the regime the real pipeline hits.
    block = (rng.random((40, 30)) < 0.15) * rng.random((40, 30))
    np.testing.assert_array_equal(near_zero_variance(block), near_zero_var_r(block))


def test_near_zero_variance_rejects_empty_input():
    with pytest.raises(ValueError, match="zero rows"):
        near_zero_variance(pd.DataFrame({"a": []}))


def test_near_zero_variance_accepts_a_single_vector():
    assert near_zero_variance(np.zeros(10)).tolist() == [0]


# --- find correlation ---------------------------------------------------------


def _caret_doc_matrix() -> np.ndarray:
    """The example matrix from ``?findCorrelation``."""
    return np.array(
        [
            [1.00, 0.86, 0.56, 0.32, 0.85],
            [0.86, 1.00, 0.01, 0.74, 0.32],
            [0.56, 0.01, 1.00, 0.65, 0.91],
            [0.32, 0.74, 0.65, 1.00, 0.36],
            [0.85, 0.32, 0.91, 0.36, 1.00],
        ]
    )


def test_fast_variant_matches_the_documented_caret_example():
    # Hand-tracing caret:::findCorrelation_fast over this matrix flags columns
    # x1, x3, x4 and x5 (1-based), i.e. zero-based {0, 2, 3, 4}.
    result = find_correlation(_caret_doc_matrix(), cutoff=0.6, exact=False)
    assert sorted(result.tolist()) == [0, 2, 3, 4]
    assert result.tolist() == find_correlation_fast_r(_caret_doc_matrix(), 0.6)


def test_exact_variant_keeps_more_columns_than_the_fast_one():
    # The exact path re-evaluates the averages after each removal, which is why
    # caret documents it as removing fewer predictors: here it flags x1, x4, x5.
    exact = find_correlation(_caret_doc_matrix(), cutoff=0.6, exact=True)
    fast = find_correlation(_caret_doc_matrix(), cutoff=0.6, exact=False)
    assert sorted(exact.tolist()) == [0, 3, 4]
    assert set(exact.tolist()) < set(fast.tolist())


def test_fast_variant_matches_reference(rng):
    data = rng.normal(size=(80, 12))
    data[:, 3] = data[:, 2] + rng.normal(scale=0.01, size=80)
    data[:, 7] = data[:, 6] * 2 + rng.normal(scale=0.01, size=80)
    corr = np.corrcoef(data, rowvar=False)

    np.testing.assert_array_equal(
        find_correlation(corr, cutoff=0.85, exact=False),
        find_correlation_fast_r(corr, cutoff=0.85),
    )


def test_uncorrelated_matrix_drops_nothing():
    assert find_correlation(np.eye(6), cutoff=0.9).size == 0


def test_the_more_correlated_partner_is_dropped():
    # Column 0 correlates with both 1 and 2; column 1 only with 0.
    corr = np.array(
        [
            [1.0, 0.95, 0.95],
            [0.95, 1.0, 0.10],
            [0.95, 0.10, 1.0],
        ]
    )
    assert find_correlation(corr, cutoff=0.9, exact=False).tolist() == [0]


def test_default_exactness_follows_caret(rng):
    small = np.corrcoef(rng.normal(size=(50, 10)), rowvar=False)
    # ncol < 100 -> exact path; both paths agree when nothing is correlated.
    assert find_correlation(small, cutoff=0.9).size == 0


def test_find_correlation_rejects_non_square_input():
    with pytest.raises(ValueError, match="square"):
        find_correlation(np.zeros((3, 4)))


def test_find_correlation_rejects_missing_values():
    corr = np.eye(3)
    corr[0, 1] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        find_correlation(corr)


def test_find_correlation_accepts_a_dataframe():
    frame = pd.DataFrame(np.eye(4))
    assert find_correlation(frame, cutoff=0.9).size == 0


# --- drop_columns -------------------------------------------------------------


def test_drop_columns_removes_the_given_positions():
    frame = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert drop_columns(frame, [1]).columns.tolist() == ["a", "c"]


def test_drop_columns_is_a_no_op_for_an_empty_selection():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    assert drop_columns(frame, []).equals(frame)


def test_drop_columns_refuses_to_empty_the_frame():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="every column"):
        drop_columns(frame, [0, 1])
