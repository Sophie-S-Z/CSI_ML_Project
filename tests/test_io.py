"""Reading raw sample files."""

from __future__ import annotations

import numpy as np
import pytest

from pyrgcms.io import list_sample_files, load_samples, read_sample_file

from .conftest import (
    TEST_MZ_MAX,
    TEST_MZ_MIN,
    TEST_N_SCANS,
    TEST_SPECIES,
)

_GEOMETRY = {"n_scans": TEST_N_SCANS, "mz_min": TEST_MZ_MIN, "mz_max": TEST_MZ_MAX}


def test_listing_is_sorted_by_name(synthetic_data_dir):
    files = list_sample_files(synthetic_data_dir)
    assert len(files) == len(TEST_SPECIES)
    assert [p.name for p in files] == sorted(p.name for p in files)


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_sample_files(tmp_path / "nope")


def test_empty_directory_is_reported(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no files with suffix"):
        list_sample_files(tmp_path / "empty")


def test_non_data_files_are_ignored(synthetic_data_dir, tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for source in list_sample_files(synthetic_data_dir)[:3]:
        (mixed / source.name).write_bytes(source.read_bytes())
    (mixed / "README.md").write_text("not data")

    assert len(list_sample_files(mixed)) == 3


def test_reading_one_sample_gives_the_expected_shape(synthetic_data_dir):
    path = list_sample_files(synthetic_data_dir)[0]
    frame = read_sample_file(path, **_GEOMETRY)

    assert frame.shape == (TEST_N_SCANS, TEST_MZ_MAX - TEST_MZ_MIN + 1)
    assert frame.columns[0] == TEST_MZ_MIN
    assert frame.columns[-1] == TEST_MZ_MAX
    assert frame.index[0] == 1
    assert frame.index[-1] == TEST_N_SCANS
    assert not frame.isna().to_numpy().any()


def test_only_the_first_n_scans_are_kept(synthetic_data_dir):
    path = list_sample_files(synthetic_data_dir)[0]
    frame = read_sample_file(path, n_scans=50, mz_min=TEST_MZ_MIN, mz_max=TEST_MZ_MAX)
    assert frame.shape[0] == 50


def test_wrong_column_count_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("scan,50,51\n1,2,3\n2,3,4\n")
    with pytest.raises(ValueError, match=r"expected \d+ columns"):
        read_sample_file(path, n_scans=2, mz_min=50, mz_max=69)


def test_too_few_scans_is_reported(tmp_path):
    header = ",".join(["scan", *(str(m) for m in range(50, 70))])
    path = tmp_path / "short.csv"
    path.write_text(header + "\n" + "\n".join(",".join(["1"] * 21) for _ in range(3)))
    with pytest.raises(ValueError, match="at least 100 scans"):
        read_sample_file(path, n_scans=100, mz_min=50, mz_max=69)


def test_unparseable_values_are_reported(tmp_path):
    header = ",".join(["scan", *(str(m) for m in range(50, 70))])
    rows = [",".join(["1"] * 21), ",".join(["1"] * 20 + ["oops"])]
    path = tmp_path / "text.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="could not be parsed"):
        read_sample_file(path, n_scans=2, mz_min=50, mz_max=69)


def test_load_samples_keeps_the_requested_positions(synthetic_data_dir):
    samples, paths = load_samples(synthetic_data_dir, keep=[0, 2, 4], **_GEOMETRY)
    assert len(samples) == 3
    assert [p.name for p in paths] == [
        "sample_000.csv",
        "sample_002.csv",
        "sample_004.csv",
    ]


def test_load_samples_reads_everything_by_default(synthetic_data_dir):
    samples, paths = load_samples(synthetic_data_dir, **_GEOMETRY)
    assert len(samples) == len(paths) == len(TEST_SPECIES)


def test_load_samples_rejects_out_of_range_positions(synthetic_data_dir):
    with pytest.raises(IndexError, match="only"):
        load_samples(synthetic_data_dir, keep=[0, 9999], **_GEOMETRY)


def test_values_survive_the_round_trip(synthetic_data_dir):
    path = list_sample_files(synthetic_data_dir)[0]
    frame = read_sample_file(path, **_GEOMETRY)
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(frame.to_numpy(), raw[:TEST_N_SCANS, 1:])


def test_shuffled_scan_column_is_rejected(tmp_path):
    header = ",".join(["scan", *(str(m) for m in range(50, 70))])
    rows = [",".join([str(scan), *(["1"] * 20)]) for scan in (1, 3, 2, 4)]
    path = tmp_path / "shuffled.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="strictly increasing"):
        read_sample_file(path, n_scans=4, mz_min=50, mz_max=69)


def test_scan_numbers_need_not_start_at_one(tmp_path):
    header = ",".join(["scan", *(str(m) for m in range(50, 70))])
    rows = [",".join([str(scan), *(["1"] * 20)]) for scan in (10, 20, 30)]
    path = tmp_path / "offset.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")

    frame = read_sample_file(path, n_scans=3, mz_min=50, mz_max=69)
    assert frame.index.tolist() == [1, 2, 3]
