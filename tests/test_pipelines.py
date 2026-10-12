"""End-to-end runs of both published pipelines over synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from pyrgcms.pipelines import nested_resampling, train_test
from pyrgcms.workflow import correlation_matrix, filter_features, load_peak_arrays

from .conftest import TEST_SPECIES


@pytest.fixture(scope="module")
def _cluster_height() -> float:
    # The synthetic chromatograms are 240 scans long, so the published height of
    # 20 scans is proportionally far too coarse; 8 keeps distinct peaks apart.
    return 8.0


# --- shared workflow steps ----------------------------------------------------


def test_loading_drops_the_excluded_samples(
    synthetic_data_dir, preprocess_config, synthetic_species
):
    peaks, labels, paths = load_peak_arrays(
        synthetic_data_dir, preprocess_config, species=synthetic_species
    )

    n_kept = sum(1 for s in synthetic_species if s != "C")
    assert peaks.shape == (
        n_kept,
        preprocess_config.n_scans,
        preprocess_config.n_mz_used,
    )
    assert len(labels) == len(paths) == n_kept
    assert set(labels) == {"A", "B"}


@pytest.mark.parametrize("extra_labels", [5, -5])
def test_loading_requires_one_label_per_file(
    synthetic_data_dir, preprocess_config, extra_labels
):
    # Labels are positional, so any count mismatch would silently shift them.
    species = (
        TEST_SPECIES + ("A",) * extra_labels
        if extra_labels > 0
        else TEST_SPECIES[:extra_labels]
    )
    with pytest.raises(ValueError, match="label"):
        load_peak_arrays(synthetic_data_dir, preprocess_config, species=species)


def test_correlation_matrix_handles_constant_columns():
    import pandas as pd

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    corr = correlation_matrix(frame)

    assert not np.isnan(corr).any()
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[1, 1] == pytest.approx(1.0)
    assert corr[0, 1] == pytest.approx(0.0)


def test_correlation_matrix_treats_float_constants_as_constant():
    import pandas as pd

    # np.std of a repeated 0.1 is ~1e-17, not 0, so a std-based check misses
    # these and the two columns would correlate through rounding noise alone.
    frame = pd.DataFrame(
        {"a": np.arange(7.0), "b": np.full(7, 0.1), "c": np.full(7, 0.1)}
    )
    corr = correlation_matrix(frame)

    np.testing.assert_array_equal(corr, np.eye(3))


def test_filter_report_accounts_for_every_stage():
    import pandas as pd

    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(40, 8)))
    frame[8] = 0.0  # constant -> near-zero variance
    frame[9] = frame[0] * 2.0 + 1e-9  # near-perfect copy -> correlated

    filtered, report = filter_features(frame)

    assert report["n_input"] == 10
    assert report["removed_near_zero_variance"] >= 1
    assert report["removed_correlated"] >= 1
    assert filtered.shape[1] == report["n_after_correlation"]
    assert report["n_after_correlation"] < report["n_input"]


# --- train/test pipeline ------------------------------------------------------


@pytest.fixture(scope="module")
def train_test_result(
    synthetic_data_dir, preprocess_config, model_config, _cluster_height
):
    return train_test.run(
        synthetic_data_dir,
        preprocess_config=preprocess_config,
        model_config=model_config,
        cluster_height=_cluster_height,
        species=TEST_SPECIES,
        tune=True,
    )


def test_train_test_pipeline_produces_a_usable_model(train_test_result):
    result = train_test_result

    n_kept = sum(1 for s in TEST_SPECIES if s != "C")
    assert len(result.y_train) + len(result.y_test) == n_kept
    assert set(result.y_train) == {"A", "B"}
    assert set(result.y_test) == {"A", "B"}

    # ceil(17 * .7) + ceil(17 * .7) = 12 + 12
    assert len(result.y_train) == 24

    assert result.training_features.shape[1] == result.testing_features.shape[1]
    assert list(result.training_features.columns) == list(result.testing_features.columns)
    assert result.training_features.shape[0] == len(result.y_train)
    assert result.testing_features.shape[0] == len(result.y_test)


def test_train_test_pipeline_separates_the_classes(train_test_result):
    # The synthetic biotic samples carry three extra marker peaks, so a working
    # pipeline must classify the held-out samples well above chance.
    assert train_test_result.test_metrics["auc"] > 0.8


def test_train_test_partitions_do_not_overlap(train_test_result):
    train = set(train_test_result.train_positions.tolist())
    test = set(train_test_result.test_positions.tolist())
    assert train & test == set()


def test_importances_cover_the_surviving_features(train_test_result):
    importances = train_test_result.importances
    assert 0 < len(importances) <= train_test_result.training_features.shape[1]
    assert importances["importance"].is_monotonic_decreasing
    assert set(importances["feature"]) <= set(train_test_result.training_features.columns)


def test_train_test_summary_mentions_the_key_numbers(train_test_result):
    summary = train_test_result.summary()
    for fragment in ("Training samples", "Test AUC", "Confusion matrix"):
        assert fragment in summary


def test_feature_names_follow_the_r_convention(train_test_result):
    named = [
        c for c in train_test_result.training_features.columns if c != "Perc_Nnonzero"
    ]
    assert named, "expected at least one aligned feature"
    for column in named:
        assert column.startswith("X")
        scan, mz = column[1:].split(".")
        assert scan.isdigit() and mz.isdigit()


def test_skipping_tuning_still_produces_a_model(
    synthetic_data_dir, preprocess_config, model_config, _cluster_height
):
    result = train_test.run(
        synthetic_data_dir,
        preprocess_config=preprocess_config,
        model_config=model_config,
        cluster_height=_cluster_height,
        species=TEST_SPECIES,
        tune=False,
        final_mtry=5,
        final_num_trees=30,
    )
    assert result.tuning_best_params == {
        "rf__max_features": 5,
        "rf__n_estimators": 30,
    }
    assert np.isnan(result.tuning_best_score)
    assert 0.0 <= result.test_metrics["auc"] <= 1.0


# --- nested resampling pipeline ----------------------------------------------


def test_nested_resampling_reports_per_fold_scores(
    synthetic_data_dir, preprocess_config, model_config, _cluster_height
):
    result = nested_resampling.run(
        synthetic_data_dir,
        preprocess_config=preprocess_config,
        model_config=model_config,
        cluster_height=_cluster_height,
        species=TEST_SPECIES,
    )

    assert result.auc.shape == (model_config.outer_folds,)
    assert result.ce.shape == (model_config.outer_folds,)
    assert result.auc_mean == pytest.approx(result.auc.mean())
    assert len(result.inner_best_params) == model_config.outer_folds
    assert result.auc_mean > 0.8
    assert not result.leak_free
    assert "Outer AUC mean" in result.summary()


def test_leak_free_variant_refits_alignment_per_fold(
    synthetic_data_dir, preprocess_config, model_config, _cluster_height
):
    result = nested_resampling.run(
        synthetic_data_dir,
        preprocess_config=preprocess_config,
        model_config=model_config,
        cluster_height=_cluster_height,
        species=TEST_SPECIES,
        leak_free=True,
    )

    assert result.leak_free
    assert result.auc.shape == (model_config.outer_folds,)
    assert 0.0 <= result.auc_mean <= 1.0
    assert "Leak-free variant  : True" in result.summary()


def test_correlation_matrix_refuses_an_oversized_request():
    import pandas as pd

    from pyrgcms.workflow import correlation_matrix

    frame = pd.DataFrame(np.zeros((3, 40)))
    with pytest.raises(ValueError, match=r"above the \S+ GB limit"):
        correlation_matrix(frame, max_bytes=1000)
