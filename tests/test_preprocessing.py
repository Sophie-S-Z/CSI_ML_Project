"""The per-sample chromatogram chain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyrgcms.config import PreprocessConfig
from pyrgcms.preprocessing import (
    normalise_sample,
    preprocess_sample,
)

from .conftest import TEST_N_MZ_USED, TEST_N_SCANS, make_sample

# --- normalisation ------------------------------------------------------------


def test_normalisation_spans_the_whole_sample_not_each_channel():
    block = np.array([[0.0, 10.0], [5.0, 20.0]])
    normalised = normalise_sample(block)

    assert normalised.min() == pytest.approx(0.0)
    assert normalised.max() == pytest.approx(1.0)
    # Per-channel scaling would have put 0.0 and 10.0 both at 0.
    assert normalised[0, 1] == pytest.approx(0.5)


def test_flat_sample_normalises_to_zeros_instead_of_nan():
    np.testing.assert_allclose(normalise_sample(np.full((3, 2), 7.0)), 0.0)


# --- full chain ---------------------------------------------------------------


def test_preprocessing_finds_the_injected_peaks(preprocess_config):
    rng = np.random.default_rng(3)
    sample = make_sample("B", rng)

    peaks = preprocess_sample(sample, preprocess_config)

    assert peaks.shape == (TEST_N_SCANS, TEST_N_MZ_USED)
    assert np.count_nonzero(peaks) > 0
    assert np.all(peaks >= 0)

    # The shared peak near scan 30 in channel 0 must survive preprocessing.
    channel_zero = np.flatnonzero(peaks[:, 0])
    assert channel_zero.size >= 1
    assert np.min(np.abs(channel_zero - 29)) <= preprocess_config.peak_half_window


def test_preprocessing_accepts_a_dataframe(preprocess_config):
    rng = np.random.default_rng(3)
    block = make_sample("A", rng)
    frame = pd.DataFrame(block, columns=range(50, 50 + block.shape[1]))

    np.testing.assert_allclose(
        preprocess_sample(frame, preprocess_config),
        preprocess_sample(block, preprocess_config),
    )


def test_preprocessing_only_uses_the_configured_channels(preprocess_config):
    rng = np.random.default_rng(5)
    block = make_sample("A", rng)
    peaks = preprocess_sample(block, preprocess_config)
    assert peaks.shape[1] == preprocess_config.n_mz_used


def test_preprocessing_is_deterministic(preprocess_config):
    block = make_sample("B", np.random.default_rng(11))
    np.testing.assert_array_equal(
        preprocess_sample(block, preprocess_config),
        preprocess_sample(block, preprocess_config),
    )


def test_preprocessing_rejects_a_short_sample(preprocess_config):
    with pytest.raises(ValueError, match="scans"):
        preprocess_sample(np.zeros((10, 20)), preprocess_config)


def test_preprocessing_rejects_too_few_channels(preprocess_config):
    with pytest.raises(ValueError, match="m/z channels"):
        preprocess_sample(np.zeros((TEST_N_SCANS, 2)), preprocess_config)


def test_preprocessing_rejects_non_matrix_input(preprocess_config):
    with pytest.raises(ValueError, match="2-D"):
        preprocess_sample(np.zeros(10), preprocess_config)


# --- config validation --------------------------------------------------------


def test_config_rejects_more_channels_than_the_files_hold():
    with pytest.raises(ValueError, match="exceeds"):
        PreprocessConfig(mz_min=50, mz_max=59, n_mz_used=20)


def test_config_rejects_nonsense_geometry():
    with pytest.raises(ValueError, match="n_mz_used must be"):
        PreprocessConfig(n_mz_used=0)
    with pytest.raises(ValueError, match="n_scans must be"):
        PreprocessConfig(n_scans=0)
