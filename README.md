# Machine learning and preprocessing for pyr-GC-MS data

Python implementation of the analysis behind

> **A robust, agnostic molecular biosignature based on machine learning.**
> H. James Cleaves II, Grethe Hystad, Anirudh Prabhu, Michael L. Wong,
> George D. Cody, Sophia Economon, and Robert M. Hazen.
> *Proceedings of the National Academy of Sciences*, 120(41), e2307149120, 2023.
> <https://www.pnas.org/doi/abs/10.1073/pnas.2307149120>

This repository was originally written in R. It has been ported to Python; the
original scripts are kept under [`reference/r/`](reference/r/) so every step can
be traced back to its source.

## What it does

Three-dimensional (scan number / mass-to-charge ratio / intensity) data from
biotic and abiotic samples are collected by pyrolysis gas chromatography mass
spectrometry. The code preprocesses those data and trains a random forest to
predict whether a sample is biotic or abiotic, and to identify the (m/z, scan)
features that discriminate between the two groups.

Peaks are aligned across samples using the binning approach of
Tibshirani et al. (2004), *Sample classification from protein mass spectrometry
by peak probability contrasts*, Bioinformatics 20(17):3034-44.

Two pipelines are provided, one per original script:

| Pipeline | Original R script | Purpose |
|---|---|---|
| `pyrgcms.pipelines.train_test` | `Preprocessing_training_and_test_data_random_forest.R` | Stratified 70/30 split, tuned random forest, held-out evaluation, impurity importances |
| `pyrgcms.pipelines.nested_resampling` | `Nested_resampling_random_forest.R` | Nested resampling for an estimate of prediction performance |

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install -e .
# with the test and lint tooling:
python -m pip install -e ".[dev]"
```

## Data

The pyr-GC-MS data (`Cleavesetal.pyrGCMSData.zip`) are **not** in this
repository. Download them from OSF: <https://osf.io/embh8/>

Unpack the archive into a directory of your choice. Each file is a
comma-separated table with a header row, one row per scan and one column per m/z
channel:

```
scan,50,51,52,...,700
1,0,12,4,...,0
2,0,9,7,...,0
```

The pipeline expects 141 files: 134 samples plus 7 that carry the `"C"` label
and are excluded, exactly as in the R code. It refuses to run if the file count
differs from the label count, since that would shift every label.

> **Ordering matters.** Sample labels are positional, so the *i*-th label in
> `pyrgcms.species.SPECIES` describes the *i*-th file in the sorted directory
> listing. R's `list.files()` sorts by locale; `pyrgcms` sorts by file name
> code point, which agrees with R's `C` locale. Check the ordering before
> trusting the labels on a fresh copy of the data.

## Usage

### Command line

```bash
# Stratified split, tuned random forest, held-out evaluation
pyrgcms train-test /path/to/data --verbose

# Skip tuning and use the hyper-parameters reported in the paper
pyrgcms train-test /path/to/data --no-tune --importance-csv importance.csv

# Nested resampling
pyrgcms nested-resampling /path/to/data --verbose
```

Useful flags: `--tuning-evals`, `--max-trees` and `--n-jobs` trade run time
against search thoroughness; `--cluster-height` sets the peak-binning width;
`--leak-free` (nested resampling only) refits alignment and feature filtering
inside every outer fold.

### Python

```python
from pyrgcms.pipelines import train_test

result = train_test.run("/path/to/data")
print(result.summary())

result.importances.to_csv("training.rf.importance.csv", index=False)
```

## Package layout

| Module | Contents |
|---|---|
| `pyrgcms.config` | Every constant from the R scripts, plus `PreprocessConfig` / `ModelConfig` |
| `pyrgcms.species` | The hard-coded sample origin labels |
| `pyrgcms.io` | Reading raw sample files |
| `pyrgcms.maldiquant` | NumPy ports of the MALDIquant primitives |
| `pyrgcms.preprocessing` | The per-sample chromatogram chain |
| `pyrgcms.alignment` | Peak binning by hierarchical clustering of scan numbers |
| `pyrgcms.feature_selection` | Ports of `caret::nearZeroVar` and `caret::findCorrelation` |
| `pyrgcms.resampling` | Port of `caret::createDataPartition` |
| `pyrgcms.model` | scikit-learn equivalents of the mlr3 / ranger blocks |
| `pyrgcms.workflow` | Steps shared by both pipelines |
| `pyrgcms.pipelines` | The two end-to-end pipelines |

[`docs/r-to-python.md`](docs/r-to-python.md) maps every R construct to its
Python replacement and lists the places where behaviour necessarily differs.

## Tests

```bash
python -m pytest --cov=pyrgcms --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
```

The suite runs against small synthetic chromatograms, so it needs no download.
The numerical ports are checked against slow, line-by-line transliterations of
the original MALDIquant C code and caret R code (`tests/reference_impl.py`),
which pins behaviour including edge handling and tie-breaking.

## Performance

Measured on the full published geometry (3240 scans x 151 m/z channels, SNIP
with 100 iterations), on a single modern desktop core:

| Stage | Cost |
|---|---|
| Chromatogram chain, per sample | ~0.8 s (~2 min for all 134 samples) |
| Peak alignment, 134 samples | ~30 s |
| Peak array in memory | ~520 MB (`134 x 3240 x 151` float64) |

Tuning dominates the total: 20 random-search evaluations x 10 folds, with
`num.trees` drawn from 1-8000, is a few hundred forest fits. Use `--n-jobs` to
spread them across cores, or `--tuning-evals` / `--max-trees` to shrink the
search. `--no-tune` skips it entirely and uses the paper's hyper-parameters.

Each MALDIquant primitive processes all 151 chromatograms of a sample in one
vectorised call rather than looping over them as the R code does, which is
where most of the speed-up over the original comes from.

## Differences from the R version

The algorithms are faithful ports; these are the deliberate exceptions.

- **Random numbers.** R and NumPy use different generators, so `--seed 99` does
  not reproduce the R partition or the R tuning path. The *procedures* are
  identical; the specific draws are not.
- **Negative raw intensities.** `MALDIquant::transformIntensity` silently drops
  points whose square root is `NaN`, which would desynchronise the fixed scan
  grid. `transform_intensity_sqrt` clips them to zero and warns instead;
  `on_negative="raise"` makes it an error.
- **`mtry` above the feature count.** `ranger` errors out; the R scripts search
  `mtry` up to 2000 regardless of how many features survive filtering.
  `ClippedRandomForestClassifier` clips the value at fit time so the search
  space stays usable.
- **Peak clustering.** The R code runs complete-linkage clustering over scan
  numbers repeated once per sample. Duplicated points merge at height 0 and
  then behave exactly like a single point, so `pyrgcms` clusters the distinct
  values instead - an identical partition at any positive cut height, and far
  cheaper. The cluster mean stays frequency-weighted, as in R. A test asserts
  the two agree.
- **Empty m/z channels.** A channel with no peaks in any sample makes R's
  `hclust` fail; `pyrgcms` skips it with a warning.
- **Constant columns in the correlation step.** R's `cor()` returns `NA` for a
  zero-variance column, which makes `findCorrelation` error out.
  `workflow.correlation_matrix` gives such a column a correlation of `0` with
  everything else, leaving it untouched by the cutoff. In practice the
  near-zero-variance filter has already removed these, so the branch is a safety
  net rather than a routine difference.
- **Scan ordering.** Both versions index scans positionally. `pyrgcms`
  additionally checks that each file's own `scan` column is strictly increasing
  and refuses a shuffled file, which R would have accepted while silently
  mis-assigning every feature.

Two properties of the original pipeline are **preserved rather than fixed**, so
results stay comparable with the paper:

- Training features take the maximum over exactly the scan numbers seen in a
  cluster, while test features take it over the whole contiguous span
  `min..max`. Pass `span="exact"` to `PeakAligner.transform` to use the training
  rule everywhere.
- In the nested-resampling script, peak alignment and feature filtering run over
  all samples *before* the outer resampling loop, which makes the outer estimate
  mildly optimistic. Pass `leak_free=True` (or `--leak-free`) to move both
  inside the loop.

## License

Released under the GNU GPL version 3. See [LICENSE](LICENSE).

The Python port depends on NumPy (BSD-3-Clause), pandas (BSD-3-Clause),
SciPy (BSD-3-Clause) and scikit-learn (BSD-3-Clause). It reimplements
algorithms from MALDIquant (GPL >= 3) and caret (GPL-2 | GPL-3); the GPL-3
license of this repository is compatible with both.

## Citation

Cite the original code: [![DOI](https://zenodo.org/badge/623661351.svg)](https://zenodo.org/doi/10.5281/zenodo.13774782)
