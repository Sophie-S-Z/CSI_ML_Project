"""The scikit-learn equivalents of the mlr3 / ranger blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pyrgcms.config import ModelConfig
from pyrgcms.model import (
    ClippedRandomForestClassifier,
    RemoveConstants,
    encode_target,
    evaluate,
    feature_importances,
    make_learner,
    make_search,
    nested_resample,
    tune_learner,
)


@pytest.fixture
def separable_data() -> tuple[pd.DataFrame, list[str]]:
    """A small, cleanly separable problem with a constant decoy feature."""
    rng = np.random.default_rng(17)
    n_per_class = 25
    signal = np.concatenate(
        [rng.normal(0.0, 1.0, n_per_class), rng.normal(5.0, 1.0, n_per_class)]
    )
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=2 * n_per_class),
            "constant": np.zeros(2 * n_per_class),
        }
    )
    labels = ["A"] * n_per_class + ["B"] * n_per_class
    return frame, labels


# --- RemoveConstants ----------------------------------------------------------


def test_remove_constants_drops_only_constant_columns():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 3.0], "c": [0.0, 5.0]})
    transformer = RemoveConstants().fit(frame)

    np.testing.assert_allclose(transformer.transform(frame), [[1.0, 0.0], [2.0, 5.0]])
    assert transformer.get_feature_names_out().tolist() == ["a", "c"]


def test_remove_constants_respects_a_tolerance():
    frame = pd.DataFrame({"a": [1.0, 1.0005], "b": [0.0, 10.0]})
    transformer = RemoveConstants(tolerance=0.01).fit(frame)
    assert transformer.get_feature_names_out().tolist() == ["b"]


def test_remove_constants_refuses_an_all_constant_matrix():
    with pytest.raises(ValueError, match="every feature is constant"):
        RemoveConstants().fit(pd.DataFrame({"a": [1.0, 1.0]}))


def test_remove_constants_checks_the_feature_count():
    transformer = RemoveConstants().fit(pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0]}))
    with pytest.raises(ValueError, match="expected 2 features"):
        transformer.transform(np.zeros((2, 3)))


def test_remove_constants_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2-D"):
        RemoveConstants().fit(np.zeros(5))


def test_remove_constants_names_unnamed_columns():
    transformer = RemoveConstants().fit(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert transformer.get_feature_names_out().tolist() == ["x0"]


# --- ClippedRandomForestClassifier -------------------------------------------


def test_max_features_is_clipped_to_the_available_columns():
    X = np.random.default_rng(0).normal(size=(30, 4))
    y = np.array([0] * 15 + [1] * 15)

    forest = ClippedRandomForestClassifier(
        n_estimators=5, max_features=100, random_state=0
    )
    forest.fit(X, y)

    # The requested value survives so the estimator stays clonable.
    assert forest.max_features == 100
    assert forest.predict(X).shape == (30,)


def test_clipping_reaches_the_individual_trees():
    X = np.random.default_rng(0).normal(size=(30, 4))
    y = np.array([0] * 15 + [1] * 15)

    forest = ClippedRandomForestClassifier(
        n_estimators=5, max_features=100, random_state=0
    ).fit(X, y)

    # Every tree sees mtry = n_features rather than the out-of-range request,
    # which is what ranger would have required.
    assert {tree.max_features for tree in forest.estimators_} == {4}


def test_string_max_features_is_left_alone():
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.array([0] * 10 + [1] * 10)
    forest = ClippedRandomForestClassifier(
        n_estimators=5, max_features="sqrt", random_state=0
    ).fit(X, y)
    assert forest.max_features == "sqrt"


# --- encode_target ------------------------------------------------------------


def test_encoding_maps_the_positive_class_to_one():
    np.testing.assert_array_equal(encode_target(["A", "B", "A"], "B"), [0, 1, 0])


def test_encoding_rejects_a_missing_positive_class():
    with pytest.raises(ValueError, match="not among the labels"):
        encode_target(["A", "B"], "Z")


def test_encoding_requires_exactly_two_classes():
    with pytest.raises(ValueError, match="exactly 2 classes"):
        encode_target(["A", "B", "C"], "B")


# --- learner / search ---------------------------------------------------------


def test_learner_is_a_two_step_pipeline():
    learner = make_learner()
    assert isinstance(learner, Pipeline)
    assert list(learner.named_steps) == ["removeconstants", "rf"]


def test_search_ranges_are_inclusive():
    config = ModelConfig(mtry_range=(3, 3), num_trees_range=(7, 7), tuning_evals=1)
    search = make_search(config)
    distribution = search.param_distributions["rf__max_features"]
    assert distribution.rvs(random_state=0) == 3
    assert search.param_distributions["rf__n_estimators"].rvs(random_state=0) == 7


def test_search_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="mtry_range"):
        make_search(ModelConfig(mtry_range=(0, 10)))
    with pytest.raises(ValueError, match="num_trees_range"):
        make_search(ModelConfig(num_trees_range=(10, 5)))


# --- tuning / evaluation ------------------------------------------------------


def _cheap_config() -> ModelConfig:
    return ModelConfig(
        seed=0,
        inner_folds=3,
        outer_folds=3,
        tuning_evals=3,
        mtry_range=(1, 3),
        num_trees_range=(5, 15),
        n_jobs=1,
    )


def test_tuning_finds_a_workable_model(separable_data):
    X, y = separable_data
    search = tune_learner(X, y, _cheap_config())

    assert search.best_score_ > 0.9
    assert 1 <= search.best_params_["rf__max_features"] <= 3
    assert 5 <= search.best_params_["rf__n_estimators"] <= 15


def test_evaluation_reports_auc_error_and_confusion(separable_data):
    X, y = separable_data
    learner = make_learner(max_features=2, n_estimators=25, random_state=0)
    learner.fit(X, encode_target(y, "B"))

    metrics = evaluate(learner, X, y, positive_class="B")
    assert metrics["auc"] > 0.95
    assert metrics["ce"] < 0.1
    assert metrics["accuracy"] == pytest.approx(1.0 - metrics["ce"])
    assert metrics["confusion"].to_numpy().sum() == len(y)


def test_confusion_labels_follow_the_positive_class(separable_data):
    X, y = separable_data
    learner = make_learner(max_features=2, n_estimators=25, random_state=0)
    learner.fit(X, encode_target(y, "A"))

    confusion = evaluate(learner, X, y, positive_class="A")["confusion"]
    # Row/column 0 is the negative class, so with "A" positive it must be "B".
    assert confusion.index.tolist() == ["B (predicted)", "A (predicted)"]
    assert confusion.columns.tolist() == ["B (truth)", "A (truth)"]
    assert confusion.loc["A (predicted)", "A (truth)"] == y.count("A")


def test_importances_skip_the_constant_feature(separable_data):
    X, y = separable_data
    learner = make_learner(max_features=2, n_estimators=25, random_state=0)
    learner.fit(X, encode_target(y, "B"))

    importances = feature_importances(learner, X.columns)
    assert "constant" not in importances["feature"].tolist()
    assert importances.iloc[0]["feature"] == "signal"
    assert importances["importance"].is_monotonic_decreasing


def test_importances_check_the_name_count(separable_data):
    X, y = separable_data
    learner = make_learner(max_features=2, n_estimators=5, random_state=0)
    learner.fit(X, encode_target(y, "B"))
    with pytest.raises(ValueError, match="expected 3 feature name"):
        feature_importances(learner, ["only", "two"])


def test_nested_resampling_returns_per_fold_scores(separable_data):
    X, y = separable_data
    result = nested_resample(X, y, _cheap_config())

    assert result["auc"].shape == (3,)
    assert result["ce"].shape == (3,)
    assert result["auc_mean"] == pytest.approx(result["auc"].mean())
    assert result["ce_mean"] == pytest.approx(result["ce"].mean())
    assert len(result["inner_best_params"]) == 3
    assert result["auc_mean"] > 0.85
