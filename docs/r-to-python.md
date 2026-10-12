# R to Python mapping

Every construct in the original scripts and where it now lives. The R sources
are kept verbatim under [`reference/r/`](../reference/r/).

## Libraries

| R library | Python replacement |
|---|---|
| `dplyr` | `pandas` |
| `data.table` | `pandas` |
| `MALDIquant` | `pyrgcms.maldiquant` (NumPy ports) |
| `caret` | `pyrgcms.feature_selection`, `pyrgcms.resampling`, `sklearn.model_selection` |
| `mlr3`, `mlr3verse`, `mlr3learners`, `mlr3tuning` | `sklearn.pipeline`, `sklearn.model_selection` |
| `ranger` (via `classif.ranger`) | `sklearn.ensemble.RandomForestClassifier` |
| `ggplot2` | not needed; pipelines return data frames |

## Data preparation

| R | Python |
|---|---|
| `setwd(...)` | `data_dir` argument on every entry point |
| `species = c(rep("A", 5), ...)` | `pyrgcms.species.SPECIES` |
| `ind = which(species2 == "3")` | `pyrgcms.species.excluded_indices()` |
| `fnames <- list.files()` | `pyrgcms.io.list_sample_files` (code-point sort) |
| `lapply(fnames2, read.delim)` + comma split | `pyrgcms.io.read_sample_file` (`pandas.read_csv`) |
| `slice(1:MM)` | `n_scans` argument |
| `factor(y, labels = c("A", "B"))` | plain `"A"` / `"B"` strings; `pyrgcms.model.encode_target` maps them to `{0, 1}` |

## Chromatogram preprocessing

| R | Python |
|---|---|
| `createMassSpectrum(...)` | not needed; arrays carry the grid implicitly |
| `transformIntensity(method = "sqrt")` | `maldiquant.transform_intensity_sqrt` |
| `smoothIntensity("MovingAverage", halfWindowSize = 5)` | `maldiquant.moving_average` |
| `removeBaseline(method = "SNIP")` | `maldiquant.remove_baseline` |
| `(x - Mi) / (Ma - Mi)` | `preprocessing.normalise_sample` |
| `detectPeaks(method = "MAD", halfWindowSize = 20, SNR = 4)` | `maldiquant.detect_peaks` |
| `v <- numeric(MM); v[scan.number] <- intensity(peaks)` | `detect_peaks` returns the dense grid directly |
| `unlist` + `paste(outer(Bin, MS, paste, sep = ';'))` (wide sample x scan;m/z table) | not needed; samples stay in one `(n_samples, n_scans, n_mz)` array |

The whole chain is `preprocessing.preprocess_sample`, which processes all m/z
channels of a sample in one vectorised pass rather than looping over them.

## Peak alignment

| R | Python |
|---|---|
| `select(ends_with(";<mz>"))` | indexing the `(n_samples, n_scans, n_mz)` array |
| `rep(colnames, times = N_elements)` | per-scan sample counts inside `PeakAligner.fit` |
| `hclust(dist(vec), method = "complete")` | `scipy.cluster.hierarchy.linkage(method="complete")` |
| `cutree(hc, h = 20)` | `scipy.cluster.hierarchy.fcluster(t=20, criterion="distance")` |
| `round(mean(Lt))` | frequency-weighted mean, `numpy.round` (half-to-even, like R) |
| `apply(dataframeN2, 1, max)` | `PeakAligner.fit_transform` / `.transform` |
| `min(L.ind2):max(L.ind2)` (test path) | `span="contiguous"` |
| `make.names(...)` | `alignment.make_names` |
| `lg` / `mutate(Perc_Nnonzero)` | `alignment.nonzero_ratio`, `workflow.append_nonzero_ratio` |

## Feature filtering

| R | Python |
|---|---|
| `nearZeroVar(x)` | `feature_selection.near_zero_variance` |
| `nearZeroVar(x, saveMetrics = TRUE)` | `feature_selection.near_zero_variance_metrics` |
| `cor(x)` | `workflow.correlation_matrix` (constant columns get correlation 0 instead of R's `NA`, and an oversized matrix is refused with a clear error) |
| `findCorrelation(corr.ma, 0.85)` | `feature_selection.find_correlation` |
| `x[, -positions]` | `feature_selection.drop_columns` |

Both filter ports follow caret's own source, including the `exact = ncol(x) < 100`
default that decides between the fast and exact `findCorrelation` variants.

## Partitioning and modelling

| R | Python |
|---|---|
| `createDataPartition(y, p = 0.7, list = FALSE)` | `resampling.create_data_partition` |
| `setdiff(seq(1, Ndim), train_set)` | `resampling.train_test_indices` |
| `as_task_classif(..., positive = "B")` | `model.encode_target(y, "B")` |
| `task$col_roles$stratum = "y"` | `StratifiedKFold` |
| `rsmp("cv", folds = 10)` | `StratifiedKFold(n_splits=10, shuffle=True)` |
| `po("removeconstants")` | `model.RemoveConstants` |
| `lrn("classif.ranger", predict_type = "prob")` | `model.ClippedRandomForestClassifier` |
| `rf.mtry` | `max_features` |
| `rf.num.trees` | `n_estimators` |
| `importance = "impurity"` | `feature_importances_` (mean decrease impurity) |
| `to_tune(p_int(1, 2000))` | `scipy.stats.randint(1, 2001)` |
| `tune(method = "random_search", term_evals = 20)` | `RandomizedSearchCV(n_iter=20)` |
| `msr("classif.auc")` | `scoring="roc_auc"` |
| `graph_learner$train(task)` | `Pipeline.fit` |
| `predict_newdata` | `Pipeline.predict` / `.predict_proba` |
| `prediction_rf$confusion` | `model.evaluate(...)["confusion"]` |
| `AutoTuner` + `resample(outer_loop)` | `model.nested_resample` |
| `rr$aggregate()` (classif.ce) | `result["ce_mean"]` |
| `rr$aggregate(msr("classif.auc"))` | `result["auc_mean"]` |
| `flt("importance")` + `write.csv` | `model.feature_importances` + `--importance-csv` |

`ranger`'s classification defaults (bootstrap with replacement, full-size
samples, `min.node.size = 1`, gini splitting, unlimited depth) are set
explicitly in `model.make_learner` so the forests match.

## Things with no Python counterpart

| R | Why it is gone |
|---|---|
| `plot(graph)` | pipeline topology is two lines of Python |
| `future::plan("multisession")` | `n_jobs` on the search / forest |
| `saveRDS` / `readRDS` of the sample list | pipelines return objects the caller can pickle |
| `prop.table(table(y_train))` | reported by `TrainTestResult.summary()` |

## Verification strategy

R is not required to check the ports. `tests/reference_impl.py` contains slow,
line-by-line transliterations of

- `C_snip` and `C_localMaxima` from MALDIquant's C sources,
- `.movingAverage` / `.filter` from MALDIquant's R sources,
- `caret:::nzv` and `caret:::findCorrelation_fast`,

and the test suite asserts the fast vectorised implementations agree with them
on random and adversarial inputs (plateaus, ties, oversized windows, sparse
peak-like matrices). The clustering shortcut is likewise checked against the
literal expanded-vector form.
