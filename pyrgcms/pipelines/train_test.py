"""Train/test pipeline.

Python port of ``Preprocessing_training_and_test_data_random_forest.R``.

The script splits the 134 samples 70/30 with stratification, derives the peak
bins from the training partition only, applies them to both partitions, filters
features on the training data, tunes a random forest by random search under
10-fold cross-validation, refits on the full training set and scores the held
out samples.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..alignment import PeakAligner
from ..config import (
    CLUSTER_HEIGHT,
    PAPER_MTRY,
    PAPER_NUM_TREES,
    ModelConfig,
    PreprocessConfig,
)
from ..model import (
    encode_target,
    evaluate,
    feature_importances,
    make_learner,
    tune_learner,
)
from ..resampling import train_test_indices
from ..species import SPECIES
from ..workflow import align_and_filter, load_peak_arrays

logger = logging.getLogger(__name__)

__all__ = ["TrainTestResult", "run"]


@dataclass
class TrainTestResult:
    """Everything the train/test pipeline produces."""

    training_features: pd.DataFrame
    testing_features: pd.DataFrame
    y_train: list[str]
    y_test: list[str]
    train_positions: np.ndarray
    test_positions: np.ndarray
    filter_report: dict
    tuning_best_params: dict
    tuning_best_score: float
    test_metrics: dict
    importances: pd.DataFrame
    aligner: PeakAligner
    learner: Pipeline

    def summary(self) -> str:
        """Human-readable one-screen summary of the run."""
        confusion = self.test_metrics["confusion"]
        return "\n".join(
            [
                f"Training samples : {len(self.y_train)}",
                f"Test samples     : {len(self.y_test)}",
                f"Aligned features : {self.filter_report['n_input']}",
                f"After NZV filter : {self.filter_report['n_after_near_zero_variance']}",
                f"After corr filter: {self.filter_report['n_after_correlation']}",
                f"Best mtry        : {self.tuning_best_params.get('rf__max_features')}",
                f"Best num.trees   : {self.tuning_best_params.get('rf__n_estimators')}",
                f"Inner CV AUC     : {self.tuning_best_score:.4f}",
                f"Test AUC         : {self.test_metrics['auc']:.4f}",
                f"Test error       : {self.test_metrics['ce']:.4f}",
                "Confusion matrix (rows = prediction, columns = truth):",
                confusion.to_string(),
            ]
        )


def run(
    data_dir: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    model_config: ModelConfig | None = None,
    cluster_height: float = CLUSTER_HEIGHT,
    tune: bool = True,
    final_mtry: int = PAPER_MTRY,
    final_num_trees: int = PAPER_NUM_TREES,
    species: Sequence[str] = SPECIES,
) -> TrainTestResult:
    """Run the full train/test pipeline.

    Parameters
    ----------
    data_dir:
        Directory of raw pyr-GC-MS sample files.
    tune:
        ``True`` runs the random search and refits with its best settings;
        ``False`` skips straight to ``final_mtry`` / ``final_num_trees`` (the
        values reported in the paper), which is far cheaper.
    """
    pre_cfg = preprocess_config or PreprocessConfig()
    mdl_cfg = model_config or ModelConfig()

    peaks, labels, _paths = load_peak_arrays(data_dir, pre_cfg, species=species)
    train_positions, test_positions = train_test_indices(
        labels, random_state=mdl_cfg.seed
    )
    y_train = [labels[i] for i in train_positions]
    y_test = [labels[i] for i in test_positions]
    logger.info("Split: %d training / %d test samples", len(y_train), len(y_test))

    training_features, testing_features, filter_report, aligner = align_and_filter(
        peaks[train_positions], peaks[test_positions], pre_cfg.mz_values, cluster_height
    )

    if tune:
        search = tune_learner(training_features, y_train, mdl_cfg)
        best_params = dict(search.best_params_)
        best_score = float(search.best_score_)
        logger.info("Random search best AUC %.4f with %s", best_score, best_params)
        mtry = int(best_params["rf__max_features"])
        num_trees = int(best_params["rf__n_estimators"])
    else:
        best_params = {
            "rf__max_features": final_mtry,
            "rf__n_estimators": final_num_trees,
        }
        best_score = float("nan")
        mtry, num_trees = final_mtry, final_num_trees
        logger.info("Skipping tuning; using mtry=%d, num.trees=%d", mtry, num_trees)

    learner = make_learner(
        max_features=mtry,
        n_estimators=num_trees,
        random_state=mdl_cfg.seed,
        n_jobs=mdl_cfg.n_jobs,
    )
    learner.fit(training_features, encode_target(y_train, mdl_cfg.positive_class))

    test_metrics = evaluate(
        learner, testing_features, y_test, positive_class=mdl_cfg.positive_class
    )
    importances = feature_importances(learner, training_features.columns)

    return TrainTestResult(
        training_features=training_features,
        testing_features=testing_features,
        y_train=y_train,
        y_test=y_test,
        train_positions=train_positions,
        test_positions=test_positions,
        filter_report=filter_report,
        tuning_best_params=best_params,
        tuning_best_score=best_score,
        test_metrics=test_metrics,
        importances=importances,
        aligner=aligner,
        learner=learner,
    )
