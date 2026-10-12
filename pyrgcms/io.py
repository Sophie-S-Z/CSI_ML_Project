"""Reading the raw pyr-GC-MS sample files.

Each raw file is a comma-separated table with one header row, one row per
chromatogram scan and one column per m/z channel::

    scan,50,51,52,...,700
    1,0,12,4,...,0
    2,0,9,7,...,0

The R script reads every file in the data directory with ``read.delim`` and
then splits the single resulting column on commas, which is equivalent to a
plain CSV read.  It keeps the first ``MM`` rows and relabels the columns
``scan, 50 ... 700``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MZ_MAX, MZ_MIN, N_SCANS

__all__ = [
    "DATA_SUFFIXES",
    "list_sample_files",
    "load_samples",
    "read_sample_file",
]

#: File extensions treated as raw sample tables.
DATA_SUFFIXES: tuple[str, ...] = (".csv", ".txt", ".tsv", ".dat")


def list_sample_files(
    data_dir: str | Path, suffixes: Iterable[str] | None = None
) -> list[Path]:
    """Return the raw sample files in ``data_dir``, sorted by name.

    ``list.files()`` in R returns names sorted by the active locale collation.
    This function sorts by the plain code-point ordering of the file name, which
    agrees with R's ``C`` locale.  Because :mod:`pyrgcms.species` labels samples
    *by position*, always confirm that the ordering here matches the ordering
    used when the label vector was written.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist or contains no matching files.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {directory}")

    wanted = tuple(
        s.lower() for s in (suffixes if suffixes is not None else DATA_SUFFIXES)
    )
    files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in wanted),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"no files with suffix {wanted} found in {directory}")
    return files


def read_sample_file(
    path: str | Path,
    n_scans: int = N_SCANS,
    mz_min: int = MZ_MIN,
    mz_max: int = MZ_MAX,
) -> pd.DataFrame:
    """Read one raw sample file into a ``(n_scans, n_mz)`` frame.

    The returned frame is indexed by scan number (1-based, as in R) and has one
    integer-named column per m/z channel.

    Raises
    ------
    ValueError
        If the file has an unexpected number of columns or too few scans.
    """
    path = Path(path)
    expected_columns = mz_max - mz_min + 2  # scan column + one per m/z
    frame = pd.read_csv(path, header=0)

    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"{path.name}: expected {expected_columns} columns "
            f"(scan + m/z {mz_min}-{mz_max}), found {frame.shape[1]}"
        )
    if frame.shape[0] < n_scans:
        raise ValueError(
            f"{path.name}: expected at least {n_scans} scans, found {frame.shape[0]}"
        )

    frame = frame.iloc[:n_scans].copy()
    frame.columns = ["scan", *range(mz_min, mz_max + 1)]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().to_numpy().any():
        bad = int(frame.isna().to_numpy().sum())
        raise ValueError(f"{path.name}: {bad} value(s) could not be parsed as numeric")

    # The R script indexes scans purely positionally, so it would happily accept
    # a shuffled file and silently mis-assign every downstream feature.  The
    # file's own scan column is already parsed here, so check it is ordered.
    recorded_scans = frame["scan"].to_numpy()
    if not np.all(np.diff(recorded_scans) > 0):
        raise ValueError(
            f"{path.name}: the scan column is not strictly increasing; "
            "rows must be in scan order"
        )

    frame.index = pd.Index(np.arange(1, n_scans + 1), name="scan")
    return frame.drop(columns="scan")


def load_samples(
    data_dir: str | Path,
    keep: Sequence[int] | None = None,
    n_scans: int = N_SCANS,
    mz_min: int = MZ_MIN,
    mz_max: int = MZ_MAX,
    suffixes: Iterable[str] | None = None,
) -> tuple[list[pd.DataFrame], list[Path]]:
    """Read every raw sample file, optionally keeping a subset by position.

    Parameters
    ----------
    data_dir:
        Directory holding the raw sample files.
    keep:
        Zero-based positions to retain, applied to the sorted file listing.
        ``None`` keeps everything.

    Returns
    -------
    (samples, paths)
        Parallel lists of per-sample frames and the files they came from.
    """
    files = list_sample_files(data_dir, suffixes=suffixes)
    if keep is not None:
        keep = list(keep)
        out_of_range = [i for i in keep if not 0 <= i < len(files)]
        if out_of_range:
            raise IndexError(
                f"keep contains position(s) {out_of_range} but only "
                f"{len(files)} file(s) were found in {data_dir}"
            )
        files = [files[i] for i in keep]

    samples = [
        read_sample_file(path, n_scans=n_scans, mz_min=mz_min, mz_max=mz_max)
        for path in files
    ]
    return samples, files
