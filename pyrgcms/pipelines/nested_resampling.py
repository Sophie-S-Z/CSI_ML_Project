"""Nested resampling pipeline.

Python port of ``Nested_resampling_random_forest.R``.

There is no held-out partition here: the peak bins and the feature filters are
derived from all 134 samples, and the generalisation estimate comes from nested
resampling -- a 10-fold tuning loop inside each fold of a 5-fold outer loop.

.. warning::
   Like the R original, the peak alignment and the feature filters run *before*
   the outer resampling loop, so the outer estimate is mildly optimistic:
   information from every fold's test samples reached the feature definitions.
   The port keeps this behaviour so its numbers stay comparable to the paper.
   :func:`run` exposes ``leak_free=True`` to refit alignment and filtering
   inside each outer fold instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..alignment import PeakAligner
from ..config import CLUSTER_HEIGHT, ModelConfig, PreprocessConfig
from ..model import encode_target, evaluate, make_search, nested_resample
from ..species import SPECIES
from ..workflow import (
    align_and_filter,
    append_nonzero_ratio,
    filter_features,
    load_peak_arrays,
)

logger = logging.getLogger(__name__)

__all__ = ["NestedResamplingResult", "run"]


@dataclass
class NestedResamplingResult:
    """Outcome of the nested resampling run."""

    features: pd.DataFrame
    labels: list[str]
    filter_report: dict
    auc: np.ndarray
    ce: np.ndarray
    auc_mean: float
    ce_mean: float
    inner_best_params: list[dict]
    inner_best_scores: list[float]
    leak_free: bool

    def summary(self) -> str:
        """Human-readable one-screen summary of the run."""
        lines = [
            f"Samples            : {len(self.labels)}",
            f"Aligned features   : {self.filter_report['n_input']}",
            f"After NZV filter   : {self.filter_report['n_after_near_zero_variance']}",
            f"After corr filter  : {self.filter_report['n_after_correlation']}",
            f"Leak-free variant  : {self.leak_free}",
            f"Outer AUC per fold : {np.round(self.auc, 4).tolist()}",
            f"Outer AUC mean     : {self.auc_mean:.4f}",
            f"Outer error mean   : {self.ce_mean:.4f}",
            "Inner tuning winners:",
        ]
        for fold, (params, score) in enumerate(
            zip(self.inner_best_params, self.inner_best_scores, strict=True), start=1
        ):
            lines.append(
                f"  fold {fold}: mtry={params.get('rf__max_features')} "
                f"num.trees={params.get('rf__n_estimators')} AUC={score:.4f}"
            )
        return "\n".join(lines)


def _aligned_features(
    peaks: np.ndarray, mz_values, cluster_height: float
) -> pd.DataFrame:
    aligner = PeakAligner(mz_values=mz_values, cluster_height=cluster_height)
    return append_nonzero_ratio(aligner.fit_transform(peaks))


def _run_leak_free(
    peaks: np.ndarray,
    labels: list[str],
    mz_values,
    cluster_height: float,
    mdl_cfg: ModelConfig,
) -> dict:
    """Refit alignment and filtering inside every outer fold."""
    target = encode_target(labels, mdl_cfg.positive_class)
    outer = StratifiedKFold(
        n_splits=mdl_cfg.outer_folds, shuffle=True, random_state=mdl_cfg.seed
    )

    auc_scores, errors, best_params, best_scores = [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(peaks, target), start=1):
        logger.info("Outer fold %d/%d", fold, mdl_cfg.outer_folds)
        train_features, test_features, _, _ = align_and_filter(
            peaks[train_idx], peaks[test_idx], mz_values, cluster_height
        )

        search = make_search(mdl_cfg)
        search.fit(train_features, target[train_idx])
        best_params.append(dict(search.best_params_))
        best_scores.append(float(search.best_score_))

        metrics = evaluate(
            search,
            test_features,
            [labels[i] for i in test_idx],
            positive_class=mdl_cfg.positive_class,
        )
        auc_scores.append(metrics["auc"])
        errors.append(metrics["ce"])

    return {
        "auc": np.asarray(auc_scores, dtype=float),
        "ce": np.asarray(errors, dtype=float),
        "auc_mean": float(np.mean(auc_scores)),
        "ce_mean": float(np.mean(errors)),
        "inner_best_params": best_params,
        "inner_best_scores": best_scores,
    }


def run(
    data_dir: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    model_config: ModelConfig | None = None,
    cluster_height: float = CLUSTER_HEIGHT,
    leak_free: bool = False,
    species: Sequence[str] = SPECIES,
) -> NestedResamplingResult:
    """Run the nested resampling pipeline.

    Parameters
    ----------
    leak_free:
        ``False`` (default) reproduces the R script, where alignment and
        filtering happen once over all samples.  ``True`` moves both inside the
        outer loop for a strictly unbiased estimate.
    """
    pre_cfg = preprocess_config or PreprocessConfig()
    mdl_cfg = model_config or ModelConfig()

    peaks, labels, _paths = load_peak_arrays(data_dir, pre_cfg, species=species)

    features_all = _aligned_features(peaks, pre_cfg.mz_values, cluster_height)
    features, filter_report = filter_features(features_all)

    if leak_free:
        outcome = _run_leak_free(
            peaks, labels, pre_cfg.mz_values, cluster_height, mdl_cfg
        )
    else:
        outcome = nested_resample(features, labels, mdl_cfg)

    return NestedResamplingResult(
        features=features,
        labels=labels,
        filter_report=filter_report,
        auc=outcome["auc"],
        ce=outcome["ce"],
        auc_mean=outcome["auc_mean"],
        ce_mean=outcome["ce_mean"],
        inner_best_params=outcome["inner_best_params"],
        inner_best_scores=outcome["inner_best_scores"],
        leak_free=leak_free,
    )
