"""Shared fixtures: a small synthetic stand-in for the real pyr-GC-MS dataset.

The published data live on OSF and are far too large for a test suite, so the
fixtures here generate chromatograms with the same *shape* -- a noisy baseline
plus Gaussian peaks -- at a fraction of the size.  Biotic samples get a few
extra marker peaks so the classifier has real signal to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyrgcms.config import ModelConfig, PreprocessConfig

#: Geometry of the synthetic dataset: small enough to run in seconds.
TEST_N_SCANS = 240
TEST_MZ_MIN = 50
TEST_MZ_MAX = 69
TEST_N_MZ_USED = 10

#: Labels of the synthetic files, including two excluded ``"C"`` samples.
TEST_SPECIES: tuple[str, ...] = tuple(
    ["A"] * 9 + ["B"] * 9 + ["C"] + ["A"] * 8 + ["B"] * 8 + ["C"]
)

#: Scans where the biotic marker peaks sit.
_BIOTIC_MARKERS = ((60, 1), (140, 3), (200, 6))
_SHARED_PEAKS = ((30, 0), (100, 2), (170, 4))


@pytest.fixture(scope="session")
def preprocess_config() -> PreprocessConfig:
    """A :class:`PreprocessConfig` matching the synthetic dataset geometry."""
    return PreprocessConfig(
        n_scans=TEST_N_SCANS,
        n_mz_used=TEST_N_MZ_USED,
        mz_min=TEST_MZ_MIN,
        mz_max=TEST_MZ_MAX,
        smooth_half_window=2,
        snip_iterations=20,
        peak_half_window=5,
        peak_snr=4.0,
    )


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    """A cheap :class:`ModelConfig`: few evaluations, small forests."""
    return ModelConfig(
        seed=99,
        inner_folds=3,
        outer_folds=3,
        tuning_evals=3,
        mtry_range=(1, 20),
        num_trees_range=(10, 40),
        n_jobs=1,
    )


def _gaussian(
    scans: np.ndarray, centre: float, width: float, height: float
) -> np.ndarray:
    return height * np.exp(-0.5 * ((scans - centre) / width) ** 2)


def make_sample(label: str, rng: np.random.Generator) -> np.ndarray:
    """Build one synthetic ``(n_scans, n_mz)`` raw intensity block."""
    n_mz = TEST_MZ_MAX - TEST_MZ_MIN + 1
    scans = np.arange(TEST_N_SCANS, dtype=float)

    # Drifting baseline plus noise, as a real chromatogram would have.
    baseline = 40.0 + 25.0 * np.sin(scans / 90.0)[:, None]
    block = baseline + rng.random((TEST_N_SCANS, n_mz)) * 6.0

    for centre, channel in _SHARED_PEAKS:
        jitter = rng.normal(scale=2.0)
        block[:, channel] += _gaussian(scans, centre + jitter, 3.0, 900.0)

    if label == "B":
        for centre, channel in _BIOTIC_MARKERS:
            jitter = rng.normal(scale=2.0)
            block[:, channel] += _gaussian(
                scans, centre + jitter, 3.0, 700.0 + rng.random() * 200.0
            )

    return block


def write_dataset(
    directory: Path, species: tuple[str, ...] = TEST_SPECIES, seed: int = 20260929
) -> Path:
    """Write one CSV per label into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    header = ",".join(["scan", *(str(mz) for mz in range(TEST_MZ_MIN, TEST_MZ_MAX + 1))])

    for position, label in enumerate(species):
        block = make_sample(label, rng)
        scan_column = np.arange(1, TEST_N_SCANS + 1, dtype=float).reshape(-1, 1)
        rows = np.hstack([scan_column, block])
        # Zero-padded names so the sorted listing matches the label order.
        path = directory / f"sample_{position:03d}.csv"
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    return directory


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """A directory of synthetic raw sample files, built once per session."""
    return write_dataset(tmp_path_factory.mktemp("pyrgcms_data"))


@pytest.fixture(scope="session")
def synthetic_species() -> tuple[str, ...]:
    """Labels matching :func:`synthetic_data_dir`, in file order."""
    return TEST_SPECIES
