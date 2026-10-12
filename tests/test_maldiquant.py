"""The vectorised MALDIquant ports must match the literal C/R transliterations."""

from __future__ import annotations

import numpy as np
import pytest

from pyrgcms import maldiquant as mq

from .reference_impl import local_maxima_c, moving_average_r, snip_c


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20230929)


# --- sqrt transform -----------------------------------------------------------


def test_sqrt_transform_matches_numpy():
    signal = np.array([0.0, 1.0, 4.0, 9.0])
    np.testing.assert_allclose(mq.transform_intensity_sqrt(signal), [0, 1, 2, 3])


def test_sqrt_transform_clips_negatives_with_a_warning():
    with pytest.warns(RuntimeWarning, match="negative intensity"):
        result = mq.transform_intensity_sqrt(np.array([-1.0, 4.0]))
    np.testing.assert_allclose(result, [0.0, 2.0])


def test_sqrt_transform_can_raise_on_negatives():
    with pytest.raises(ValueError, match="negative intensity"):
        mq.transform_intensity_sqrt(np.array([-1.0]), on_negative="raise")


def test_sqrt_transform_rejects_unknown_policy():
    with pytest.raises(ValueError, match="on_negative"):
        mq.transform_intensity_sqrt(np.array([1.0]), on_negative="ignore")


# --- moving average -----------------------------------------------------------


@pytest.mark.parametrize("half_window_size", [1, 2, 5, 9])
def test_moving_average_matches_reference(rng, half_window_size):
    signal = rng.random(200) * 100
    np.testing.assert_allclose(
        mq.moving_average(signal, half_window_size),
        moving_average_r(signal, half_window_size),
        rtol=1e-12,
        atol=1e-10,
    )


def test_moving_average_edges_use_the_nearest_full_window():
    signal = np.arange(1.0, 11.0)
    smoothed = mq.moving_average(signal, half_window_size=2)
    first_window_mean = signal[:5].mean()
    last_window_mean = signal[-5:].mean()
    assert smoothed[0] == pytest.approx(first_window_mean)
    assert smoothed[1] == pytest.approx(first_window_mean)
    assert smoothed[-1] == pytest.approx(last_window_mean)
    assert smoothed[-2] == pytest.approx(last_window_mean)


def test_moving_average_handles_columns_independently(rng):
    block = rng.random((120, 4)) * 10
    smoothed = mq.moving_average(block, half_window_size=3)
    for column in range(block.shape[1]):
        np.testing.assert_allclose(
            smoothed[:, column], mq.moving_average(block[:, column], 3)
        )


def test_moving_average_rejects_oversized_window():
    with pytest.raises(ValueError, match="too large"):
        mq.moving_average(np.arange(5.0), half_window_size=3)


def test_moving_average_rejects_zero_window():
    with pytest.raises(ValueError, match=">= 1"):
        mq.moving_average(np.arange(50.0), half_window_size=0)


# --- SNIP baseline ------------------------------------------------------------


@pytest.mark.parametrize("decreasing", [True, False])
@pytest.mark.parametrize("iterations", [1, 5, 30])
def test_snip_matches_reference(rng, iterations, decreasing):
    signal = rng.random(150) * 50 + np.linspace(0, 20, 150)
    np.testing.assert_allclose(
        mq.snip_baseline(signal, iterations=iterations, decreasing=decreasing),
        snip_c(signal, iterations=iterations, decreasing=decreasing),
        rtol=1e-12,
        atol=1e-12,
    )


def test_snip_handles_iterations_larger_than_the_signal(rng):
    signal = rng.random(9) * 10
    np.testing.assert_allclose(
        mq.snip_baseline(signal, iterations=100),
        snip_c(signal, iterations=100),
    )


def test_snip_never_exceeds_the_signal(rng):
    signal = rng.random(200) * 100
    assert np.all(mq.snip_baseline(signal, iterations=40) <= signal + 1e-12)


def test_snip_zero_iterations_is_the_identity(rng):
    signal = rng.random(50)
    np.testing.assert_allclose(mq.snip_baseline(signal, iterations=0), signal)


def test_snip_rejects_negative_iterations():
    with pytest.raises(ValueError, match="iterations"):
        mq.snip_baseline(np.arange(10.0), iterations=-1)


def test_remove_baseline_subtracts_without_clipping(rng):
    signal = rng.random(100) * 10
    residual = mq.remove_baseline(signal, iterations=20)
    np.testing.assert_allclose(residual, signal - mq.snip_baseline(signal, 20))
    # removeBaseline does not clamp, unlike smoothIntensity.
    assert residual.min() <= 0.0 + 1e-12


def test_snip_columns_are_independent(rng):
    block = rng.random((80, 3)) * 30
    baseline = mq.snip_baseline(block, iterations=15)
    for column in range(block.shape[1]):
        np.testing.assert_allclose(
            baseline[:, column], mq.snip_baseline(block[:, column], 15)
        )


# --- MAD ----------------------------------------------------------------------


def test_mad_matches_r_definition():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    expected = 1.4826 * np.median(np.abs(signal - np.median(signal)))
    assert mq.mad(signal) == pytest.approx(expected)


def test_mad_is_per_column(rng):
    block = rng.random((50, 3))
    per_column = mq.mad(block)
    assert per_column.shape == (3,)
    for column in range(3):
        assert per_column[column] == pytest.approx(mq.mad(block[:, column]))


# --- local maxima -------------------------------------------------------------


@pytest.mark.parametrize("half_window_size", [1, 2, 5, 20])
def test_local_maxima_matches_reference(rng, half_window_size):
    signal = rng.random(400) * 10
    np.testing.assert_array_equal(
        mq.local_maxima(signal, half_window_size),
        local_maxima_c(signal, half_window_size),
    )


def test_local_maxima_matches_reference_with_plateaus(rng):
    # Heavy rounding creates ties, exercising the left-most-maximum rule.
    signal = np.round(rng.random(300) * 4)
    np.testing.assert_array_equal(mq.local_maxima(signal, 3), local_maxima_c(signal, 3))


def test_local_maxima_picks_the_leftmost_of_a_plateau():
    signal = np.array([0.0, 1.0, 5.0, 5.0, 1.0, 0.0])
    mask = mq.local_maxima(signal, half_window_size=2)
    assert mask.tolist() == [False, False, True, False, False, False]


def test_local_maxima_zero_window_flags_everything():
    signal = np.array([3.0, 1.0, 2.0])
    assert mq.local_maxima(signal, half_window_size=0).all()


def test_local_maxima_rejects_negative_window():
    with pytest.raises(ValueError, match=">= 0"):
        mq.local_maxima(np.arange(10.0), half_window_size=-1)


# --- peak detection -----------------------------------------------------------


def test_detect_peaks_keeps_only_maxima_above_the_noise(rng):
    signal = rng.random(500) * 0.1
    signal[100] = 50.0
    signal[300] = 40.0
    peaks = mq.detect_peaks(signal, half_window_size=20, snr=4)

    assert peaks[100] == pytest.approx(50.0)
    assert peaks[300] == pytest.approx(40.0)
    assert np.count_nonzero(peaks) == 2


def test_detect_peaks_on_a_flat_signal_follows_the_zero_padding_rule():
    signal = np.full(200, 5.0)
    signal[50] = 5.001
    peaks = mq.detect_peaks(signal, half_window_size=10, snr=4)

    # MAD is 0 here, so every point clears the SNR gate and only the local
    # maximum rule matters.  MALDIquant pads with zeros, so the very first
    # point beats its (padded) left neighbourhood and is flagged alongside the
    # genuine bump -- the reference transliteration agrees.
    np.testing.assert_array_equal(peaks > 0, local_maxima_c(signal, half_window_size=10))
    assert peaks[50] == pytest.approx(5.001)
    assert peaks[0] == pytest.approx(5.0)
    assert np.count_nonzero(peaks) == 2


def test_detect_peaks_is_column_wise(rng):
    block = rng.random((300, 3)) * 0.1
    block[50, 0] = 20.0
    block[150, 2] = 30.0
    peaks = mq.detect_peaks(block, half_window_size=15, snr=4)
    assert peaks[50, 0] == pytest.approx(20.0)
    assert peaks[150, 2] == pytest.approx(30.0)
    assert np.count_nonzero(peaks[:, 1]) >= 0  # no crash on a quiet channel


def test_two_dimensional_helpers_reject_three_dimensional_input():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        mq.moving_average(np.zeros((2, 3, 4)), 1)
