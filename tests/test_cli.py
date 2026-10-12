"""The command line interface."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrgcms import cli

from .conftest import TEST_MZ_MAX, TEST_MZ_MIN, TEST_N_MZ_USED, TEST_N_SCANS


def test_parser_requires_a_sub_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_parser_reads_the_train_test_options():
    args = cli.build_parser().parse_args(
        ["train-test", "data", "--cluster-height", "8", "--no-tune", "--seed", "3"]
    )
    assert args.command == "train-test"
    assert args.cluster_height == 8.0
    assert args.no_tune is True
    assert args.seed == 3


def test_parser_reads_the_nested_resampling_options():
    args = cli.build_parser().parse_args(
        ["nested-resampling", "data", "--leak-free", "--max-trees", "50"]
    )
    assert args.command == "nested-resampling"
    assert args.leak_free is True
    assert args.max_trees == 50


def test_missing_data_directory_exits_non_zero(capsys, tmp_path):
    code = cli.main(["train-test", str(tmp_path / "absent")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_invalid_option_values_exit_non_zero(capsys, tmp_path):
    code = cli.main(["train-test", str(tmp_path), "--n-mz", "0"])
    assert code == 1
    assert "n_mz_used must be" in capsys.readouterr().err


def test_train_test_command_prints_a_summary(synthetic_data_dir, capsys):
    code = cli.main(_train_test_argv(synthetic_data_dir))

    assert code == 0
    output = capsys.readouterr().out
    assert "Training samples" in output
    assert "Test AUC" in output


def test_train_test_command_writes_the_importance_csv(
    synthetic_data_dir, tmp_path, capsys
):
    destination = tmp_path / "out" / "importance.csv"
    code = cli.main(
        [*_train_test_argv(synthetic_data_dir), "--importance-csv", str(destination)]
    )

    assert code == 0
    assert destination.is_file()
    table = pd.read_csv(destination)
    assert list(table.columns) == ["feature", "importance"]
    assert len(table) > 0
    assert "Importances written to" in capsys.readouterr().out


def test_nested_resampling_command_prints_a_summary(synthetic_data_dir, capsys):
    code = cli.main(
        [
            "nested-resampling",
            str(synthetic_data_dir),
            "--cluster-height",
            "8",
            "--n-mz",
            str(TEST_N_MZ_USED),
            "--tuning-evals",
            "2",
            "--max-trees",
            "25",
            "--n-jobs",
            "1",
        ]
    )
    assert code == 0
    assert "Outer AUC mean" in capsys.readouterr().out


def test_verbosity_flag_is_accepted(synthetic_data_dir, capsys):
    code = cli.main(["-v", *_train_test_argv(synthetic_data_dir)])
    assert code == 0


def _train_test_argv(data_dir) -> list[str]:
    return [
        "train-test",
        str(data_dir),
        "--cluster-height",
        "8",
        "--n-mz",
        str(TEST_N_MZ_USED),
        "--no-tune",
        "--n-jobs",
        "1",
    ]


@pytest.fixture(autouse=True)
def _synthetic_geometry(monkeypatch):
    """Point the CLI at the synthetic dataset's geometry and labels."""
    from pyrgcms.pipelines import nested_resampling, train_test

    from .conftest import TEST_SPECIES

    original_train_test = train_test.run
    original_nested = nested_resampling.run

    def train_test_run(data_dir, **kwargs):
        kwargs.setdefault("species", TEST_SPECIES)
        kwargs["preprocess_config"] = _shrink(kwargs.get("preprocess_config"))
        kwargs["final_mtry"] = min(kwargs.get("final_mtry", 5), 5)
        kwargs["final_num_trees"] = min(kwargs.get("final_num_trees", 30), 30)
        return original_train_test(data_dir, **kwargs)

    def nested_run(data_dir, **kwargs):
        kwargs.setdefault("species", TEST_SPECIES)
        kwargs["preprocess_config"] = _shrink(kwargs.get("preprocess_config"))
        return original_nested(data_dir, **kwargs)

    monkeypatch.setattr(cli.train_test, "run", train_test_run)
    monkeypatch.setattr(cli.nested_resampling, "run", nested_run)


def _shrink(config):
    """Rebuild a CLI-built config with the synthetic dataset's geometry."""
    from dataclasses import replace

    from pyrgcms.config import PreprocessConfig

    base = config or PreprocessConfig()
    return replace(
        base,
        n_scans=TEST_N_SCANS,
        mz_min=TEST_MZ_MIN,
        mz_max=TEST_MZ_MAX,
        n_mz_used=min(base.n_mz_used, TEST_N_MZ_USED),
        smooth_half_window=2,
        snip_iterations=20,
        peak_half_window=5,
    )
