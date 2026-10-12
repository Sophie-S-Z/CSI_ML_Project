"""Preprocessing and machine learning for pyrolysis GC-MS biosignature data.

Python port of the R code accompanying Cleaves et al. (2023), *A robust,
agnostic molecular biosignature based on machine learning*, PNAS 120(41),
e2307149120.

The package mirrors the two published scripts:

* :mod:`pyrgcms.pipelines.train_test` -- stratified 70/30 split, tuned random
  forest, held-out evaluation and impurity importances.
* :mod:`pyrgcms.pipelines.nested_resampling` -- nested resampling for an
  unbiased estimate of prediction performance.

Typical use::

    from pyrgcms.pipelines import train_test

    result = train_test.run("path/to/Cleavesetal.pyrGCMSData")
    print(result.summary())
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "alignment",
    "config",
    "feature_selection",
    "io",
    "maldiquant",
    "model",
    "pipelines",
    "preprocessing",
    "resampling",
    "species",
    "workflow",
]

from . import (
    alignment,
    config,
    feature_selection,
    io,
    maldiquant,
    model,
    pipelines,
    preprocessing,
    resampling,
    species,
    workflow,
)
