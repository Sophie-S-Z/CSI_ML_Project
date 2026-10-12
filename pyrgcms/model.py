"""Random forest modelling, the equivalent of the ``mlr3`` / ``ranger`` blocks.

Mapping from the R pipeline to scikit-learn:

=================================  =================================================
mlr3 / ranger                      scikit-learn
=================================  =================================================
``po("removeconstants")``          :class:`RemoveConstants`
``lrn("classif.ranger")``          :class:`ClippedRandomForestClassifier`
``rf.mtry``                        ``max_features``
``rf.num.trees``                   ``n_estimators``
``importance = "impurity"``        ``feature_importances_`` (mean decrease impurity)
``rsmp("cv", folds = k)`` + strata :class:`~sklearn.model_selection.StratifiedKFold`
``tune(method = "random_search")`` :class:`~sklearn.model_selection.RandomizedSearchCV`
``msr("classif.auc")``             ``scoring="roc_auc"``
``AutoTuner`` + outer ``resample`` :func:`nested_resample`
=================================  =================================================
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .config import POSITIVE_CLASS, ModelConfig

__all__ = [
    "ClippedRandomForestClassifier",
    "RemoveConstants",
    "encode_target",
    "evaluate",
    "feature_importances",
    "make_learner",
    "make_search",
    "nested_resample",
    "tune_learner",
]


class RemoveConstants(BaseEstimator, TransformerMixin):
    """Drop features that take a single value, like ``po("removeconstants")``.

    Parameters
    ----------
    tolerance:
        Features whose range is at most ``tolerance`` are treated as constant.
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    def fit(self, X, y=None) -> RemoveConstants:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                f"expected a 2-D feature matrix, got {values.ndim} dimensions"
            )
        spread = values.max(axis=0) - values.min(axis=0)
        self.support_ = spread > self.tolerance
        if not self.support_.any():
            raise ValueError("every feature is constant; nothing would be left to fit")
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "support_")
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {values.shape[1]}"
            )
        return values[:, self.support_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "support_")
        if input_features is None:
            input_features = getattr(
                self,
                "feature_names_in_",
                np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object),
            )
        names = np.asarray(input_features, dtype=object)
        if names.shape[0] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} feature name(s), got {names.shape[0]}"
            )
        return names[self.support_]


class ClippedRandomForestClassifier(RandomForestClassifier):
    """Random forest that clips ``max_features`` to the available feature count.

    ``ranger`` errors out when ``mtry`` exceeds the number of predictors, and
    older scikit-learn versions do too.  The R scripts search ``mtry`` up to
    2000 regardless of how many features survive filtering, so the value is
    clipped at fit time: the search space stays usable, the meaning ("consider
    every feature") is explicit, and the estimator stays clonable because the
    requested value is restored afterwards.
    """

    def fit(self, X, y, sample_weight=None):
        n_features = np.asarray(X).shape[1]
        requested = self.max_features
        if isinstance(requested, (int, np.integer)) and not isinstance(requested, bool):
            self.max_features = int(min(int(requested), n_features))
        try:
            return super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.max_features = requested


def make_learner(
    max_features: int | str | None = "sqrt",
    n_estimators: int = 500,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> Pipeline:
    """Build the ``removeconstants -> random forest`` pipeline."""
    return Pipeline(
        [
            ("removeconstants", RemoveConstants()),
            (
                "rf",
                ClippedRandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    # ranger's classification defaults: bootstrap with
                    # replacement, full-size samples, unlimited depth,
                    # min.node.size = 1, gini splitting.
                    bootstrap=True,
                    min_samples_leaf=1,
                    criterion="gini",
                ),
            ),
        ]
    )


def make_search(config: ModelConfig | None = None) -> RandomizedSearchCV:
    """Build the random search over ``mtry`` and ``num.trees``."""
    cfg = config or ModelConfig()
    mtry_lo, mtry_hi = cfg.mtry_range
    trees_lo, trees_hi = cfg.num_trees_range
    if mtry_lo < 1 or mtry_hi < mtry_lo:
        raise ValueError(f"invalid mtry_range: {cfg.mtry_range}")
    if trees_lo < 1 or trees_hi < trees_lo:
        raise ValueError(f"invalid num_trees_range: {cfg.num_trees_range}")

    distributions = {
        # scipy's randint upper bound is exclusive; R's p_int is inclusive.
        "rf__max_features": randint(mtry_lo, mtry_hi + 1),
        "rf__n_estimators": randint(trees_lo, trees_hi + 1),
    }
    return RandomizedSearchCV(
        estimator=make_learner(random_state=cfg.seed, n_jobs=1),
        param_distributions=distributions,
        n_iter=cfg.tuning_evals,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.seed),
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        refit=True,
        error_score="raise",
    )


def encode_target(y: Sequence, positive_class: str) -> np.ndarray:
    """Map labels to ``{0, 1}`` with ``positive_class`` as ``1``."""
    labels = np.asarray(list(y))
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError(f"expected exactly 2 classes, found {list(classes)}")
    if positive_class not in classes:
        raise ValueError(
            f"positive class {positive_class!r} is not among the labels {list(classes)}"
        )
    return (labels == positive_class).astype(int)


def tune_learner(
    X: pd.DataFrame, y: Sequence, config: ModelConfig | None = None
) -> RandomizedSearchCV:
    """Run the inner random search and return the fitted search object."""
    cfg = config or ModelConfig()
    search = make_search(cfg)
    search.fit(X, encode_target(y, cfg.positive_class))
    return search


def nested_resample(
    X: pd.DataFrame, y: Sequence, config: ModelConfig | None = None
) -> dict:
    """Nested resampling: tuned model inside, unbiased estimate outside.

    The equivalent of ``AutoTuner`` + ``resample()`` in the R script.

    Returns
    -------
    dict
        ``auc`` and ``ce`` arrays (one entry per outer fold), their means, and
        the per-fold best hyper-parameters.
    """
    cfg = config or ModelConfig()
    target = encode_target(y, cfg.positive_class)
    outer = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.seed)

    results = cross_validate(
        estimator=make_search(cfg),
        X=X,
        y=target,
        cv=outer,
        scoring={"auc": "roc_auc", "accuracy": "accuracy"},
        return_estimator=True,
        n_jobs=1,
        error_score="raise",
    )

    auc = np.asarray(results["test_auc"], dtype=float)
    classification_error = 1.0 - np.asarray(results["test_accuracy"], dtype=float)
    return {
        "auc": auc,
        "ce": classification_error,
        "auc_mean": float(auc.mean()),
        "ce_mean": float(classification_error.mean()),
        "inner_best_params": [est.best_params_ for est in results["estimator"]],
        "inner_best_scores": [float(est.best_score_) for est in results["estimator"]],
    }


def feature_importances(fitted: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    """Impurity importances of a fitted learner, most important first.

    Names are mapped back through :class:`RemoveConstants`, so dropped constant
    features do not shift the labels.
    """
    remover = fitted.named_steps["removeconstants"]
    forest = fitted.named_steps["rf"]
    kept = remover.get_feature_names_out(np.asarray(list(feature_names), dtype=object))
    if len(kept) != len(forest.feature_importances_):
        raise ValueError(
            f"{len(kept)} surviving feature name(s) but "
            f"{len(forest.feature_importances_)} importance value(s)"
        )
    frame = pd.DataFrame({"feature": kept, "importance": forest.feature_importances_})
    return frame.sort_values("importance", ascending=False, ignore_index=True)


def evaluate(
    fitted: Pipeline | RandomizedSearchCV,
    X: pd.DataFrame,
    y: Sequence,
    positive_class: str = POSITIVE_CLASS,
) -> dict:
    """Score a fitted learner: AUC, classification error and confusion matrix."""
    truth = encode_target(y, positive_class)
    negative_class = next(label for label in set(y) if label != positive_class)
    classes = [negative_class, positive_class]  # encoded as 0, 1
    positive_column = list(fitted.classes_).index(1)
    probability = fitted.predict_proba(X)[:, positive_column]
    predicted = fitted.predict(X)

    true_positive = int(np.sum((predicted == 1) & (truth == 1)))
    false_positive = int(np.sum((predicted == 1) & (truth == 0)))
    true_negative = int(np.sum((predicted == 0) & (truth == 0)))
    false_negative = int(np.sum((predicted == 0) & (truth == 1)))

    return {
        "auc": float(roc_auc_score(truth, probability)),
        "ce": float(np.mean(predicted != truth)),
        "accuracy": float(np.mean(predicted == truth)),
        "confusion": pd.DataFrame(
            [[true_negative, false_negative], [false_positive, true_positive]],
            index=pd.Index([f"{c} (predicted)" for c in classes], name="response"),
            columns=pd.Index([f"{c} (truth)" for c in classes], name="truth"),
        ),
    }
