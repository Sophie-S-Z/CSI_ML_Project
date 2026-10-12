"""Peak alignment: clustering, naming and the train/test span asymmetry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from pyrgcms.alignment import (
    NONZERO_RATIO_COLUMN,
    PeakAligner,
    ScanCluster,
    make_names,
    nonzero_ratio,
)

# --- make.names ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1234;57", "X1234.57"),
        ("12;50", "X12.50"),
        ("Perc_Nnonzero", "Perc_Nnonzero"),
        ("a b", "a.b"),
        ("", "X"),
        (".5", "X.5"),
        (".name", ".name"),
        ("if", "Xif"),
        ("TRUE", "XTRUE"),
        ("x1", "x1"),
    ],
)
def test_make_names_matches_r(raw, expected):
    assert make_names(raw) == expected


# --- ScanCluster --------------------------------------------------------------


def test_cluster_name_and_span():
    cluster = ScanCluster(mz=57, scans=(100, 104, 110), mean_scan=105)
    assert cluster.name == "X105.57"
    assert cluster.span == tuple(range(100, 111))


# --- clustering ---------------------------------------------------------------


def _peaks(n_samples: int, n_scans: int, n_mz: int) -> np.ndarray:
    return np.zeros((n_samples, n_scans, n_mz))


def test_nearby_scans_collapse_into_one_feature():
    peaks = _peaks(3, 300, 1)
    peaks[0, 100, 0] = 1.0
    peaks[1, 105, 0] = 2.0
    peaks[2, 110, 0] = 3.0  # all within the height-20 window

    aligner = PeakAligner(mz_values=[50], cluster_height=20)
    features = aligner.fit_transform(peaks)

    assert features.shape == (3, 1)
    np.testing.assert_allclose(features.to_numpy().ravel(), [1.0, 2.0, 3.0])


def test_distant_scans_become_separate_features():
    peaks = _peaks(2, 300, 1)
    peaks[0, 50, 0] = 1.0
    peaks[1, 200, 0] = 2.0

    features = PeakAligner(mz_values=[50], cluster_height=20).fit_transform(peaks)

    assert features.shape == (2, 2)
    np.testing.assert_allclose(features.to_numpy(), [[1.0, 0.0], [0.0, 2.0]])


def test_features_are_ordered_by_ascending_scan_then_mz():
    peaks = _peaks(1, 300, 2)
    peaks[0, 200, 0] = 1.0
    peaks[0, 50, 0] = 1.0
    peaks[0, 120, 1] = 1.0

    aligner = PeakAligner(mz_values=[50, 51], cluster_height=20).fit(peaks)
    assert aligner.feature_names_ == ["X51.50", "X201.50", "X121.51"]


def test_mean_scan_is_frequency_weighted():
    # Three samples peak at scan 100, one at scan 112 -> weighted mean 103.
    peaks = _peaks(4, 300, 1)
    peaks[0, 99, 0] = peaks[1, 99, 0] = peaks[2, 99, 0] = 1.0
    peaks[3, 111, 0] = 1.0

    aligner = PeakAligner(mz_values=[50], cluster_height=20).fit(peaks)
    assert aligner.clusters_[0].mean_scan == 103


def test_mean_scan_rounds_half_to_even_like_r():
    # Scans 100 and 105 with equal weight -> 102.5 -> 102 under R's `round`.
    peaks = _peaks(2, 300, 1)
    peaks[0, 99, 0] = 1.0
    peaks[1, 104, 0] = 1.0

    aligner = PeakAligner(mz_values=[50], cluster_height=20).fit(peaks)
    assert aligner.clusters_[0].mean_scan == 102


def test_deduplicated_clustering_matches_the_expanded_vector():
    """Clustering distinct scans must equal clustering the R-style repeated vector."""
    rng = np.random.default_rng(7)
    n_samples, n_scans = 12, 400
    peaks = np.zeros((n_samples, n_scans, 1))
    for sample in range(n_samples):
        for scan in rng.choice(n_scans, size=25, replace=False):
            peaks[sample, scan, 0] = rng.random() + 0.1

    aligner = PeakAligner(mz_values=[50], cluster_height=20).fit(peaks)

    # Rebuild R's `vec`: each scan repeated once per sample that peaks there.
    counts = (peaks[:, :, 0] > 0).sum(axis=0)
    expanded = np.repeat(np.arange(1, n_scans + 1), counts)
    membership = fcluster(
        linkage(pdist(expanded.reshape(-1, 1)), method="complete"),
        t=20,
        criterion="distance",
    )
    expected_groups: list[list[int]] = []
    seen: dict[int, int] = {}
    for scan, label in zip(expanded, membership, strict=True):
        if label not in seen:
            seen[label] = len(expected_groups)
            expected_groups.append([])
        if int(scan) not in expected_groups[seen[label]]:
            expected_groups[seen[label]].append(int(scan))

    assert [list(c.scans) for c in aligner.clusters_] == expected_groups


def test_channel_without_peaks_is_skipped_with_a_warning():
    peaks = _peaks(2, 300, 2)
    peaks[0, 100, 0] = 1.0  # only the first m/z has any peak

    with pytest.warns(RuntimeWarning, match="no peaks"):
        aligner = PeakAligner(mz_values=[50, 51], cluster_height=20).fit(peaks)
    assert aligner.feature_names_ == ["X101.50"]


def test_completely_empty_data_is_rejected():
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(ValueError, match="no peaks were detected"),
    ):
        PeakAligner(mz_values=[50], cluster_height=20).fit(_peaks(2, 100, 1))


# --- train vs test span -------------------------------------------------------


def test_transform_uses_the_contiguous_span_like_the_r_test_path():
    train = _peaks(2, 300, 1)
    train[0, 99, 0] = 1.0  # scan 100
    train[1, 109, 0] = 1.0  # scan 110 -> cluster scans {100, 110}

    aligner = PeakAligner(mz_values=[50], cluster_height=20).fit(train)
    assert aligner.clusters_[0].scans == (100, 110)

    test = _peaks(1, 300, 1)
    test[0, 104, 0] = 7.0  # scan 105: inside the span but not a fitted scan

    contiguous = aligner.transform(test, span="contiguous")
    exact = aligner.transform(test, span="exact")

    assert contiguous.iloc[0, 0] == pytest.approx(7.0)
    assert exact.iloc[0, 0] == pytest.approx(0.0)


def test_fit_transform_uses_the_exact_scans():
    train = _peaks(2, 300, 1)
    train[0, 99, 0] = 1.0
    train[1, 109, 0] = 2.0
    aligner = PeakAligner(mz_values=[50], cluster_height=20)

    fitted = aligner.fit_transform(train)
    np.testing.assert_allclose(fitted.to_numpy().ravel(), [1.0, 2.0])


def test_transform_rejects_an_unknown_span():
    peaks = _peaks(1, 100, 1)
    peaks[0, 50, 0] = 1.0
    aligner = PeakAligner(mz_values=[50], cluster_height=20).fit(peaks)
    with pytest.raises(ValueError, match="span must be"):
        aligner.transform(peaks, span="sideways")


def test_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not been fitted"):
        PeakAligner(mz_values=[50]).transform(_peaks(1, 100, 1))


def test_transform_rejects_a_shorter_scan_grid():
    train = _peaks(1, 300, 1)
    train[0, 250, 0] = 1.0
    aligner = PeakAligner(mz_values=[50], cluster_height=20).fit(train)
    with pytest.raises(ValueError, match="only has"):
        aligner.transform(_peaks(1, 100, 1))


def test_mismatched_channel_count_is_rejected():
    with pytest.raises(ValueError, match="m/z channels"):
        PeakAligner(mz_values=[50, 51]).fit(_peaks(1, 100, 3))


def test_non_three_dimensional_input_is_rejected():
    with pytest.raises(ValueError, match="n_samples, n_scans, n_mz"):
        PeakAligner(mz_values=[50]).fit(np.zeros((10, 10)))


def test_invalid_cluster_height_is_rejected():
    with pytest.raises(ValueError, match="cluster_height"):
        PeakAligner(mz_values=[50], cluster_height=0)


# --- non-zero ratio -----------------------------------------------------------


def test_nonzero_ratio_counts_strictly_positive_values():
    frame = pd.DataFrame([[0.0, 1.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    np.testing.assert_allclose(nonzero_ratio(frame).to_numpy(), [0.5, 1.0])
    assert nonzero_ratio(frame).name == NONZERO_RATIO_COLUMN


def test_nonzero_ratio_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="zero features"):
        nonzero_ratio(pd.DataFrame(index=[0, 1]))


def test_duplicate_feature_names_are_rejected(monkeypatch):
    peaks = _peaks(2, 300, 1)
    peaks[0, 49, 0] = 1.0
    peaks[1, 249, 0] = 1.0  # two well-separated clusters

    # Force both clusters onto the same rounded mean scan.
    monkeypatch.setattr(np, "round", lambda value: 100.0)
    with pytest.raises(ValueError, match="duplicate names"):
        PeakAligner(mz_values=[50], cluster_height=20).fit(peaks)
