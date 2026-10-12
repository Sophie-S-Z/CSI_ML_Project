"""Command line entry point: ``python -m pyrgcms`` or ``pyrgcms``."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import (
    CLUSTER_HEIGHT,
    PAPER_MTRY,
    PAPER_NUM_TREES,
    ModelConfig,
    PreprocessConfig,
)
from .pipelines import nested_resampling, train_test

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for both sub-commands."""
    parser = argparse.ArgumentParser(
        prog="pyrgcms",
        description=(
            "Preprocess pyrolysis GC-MS data and classify samples as biotic or "
            "abiotic with a random forest."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="increase logging detail (repeat for debug output)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "data_dir", type=Path, help="directory holding the raw sample files"
        )
        sub.add_argument(
            "--cluster-height",
            type=float,
            default=CLUSTER_HEIGHT,
            help=f"dendrogram cut height in scans (default: {CLUSTER_HEIGHT:g})",
        )
        sub.add_argument(
            "--seed",
            type=int,
            default=ModelConfig().seed,
            help=f"random seed (default: {ModelConfig().seed})",
        )
        sub.add_argument(
            "--tuning-evals",
            type=int,
            default=ModelConfig().tuning_evals,
            help=(
                "random-search evaluations per tuning run "
                f"(default: {ModelConfig().tuning_evals})"
            ),
        )
        sub.add_argument(
            "--max-trees",
            type=int,
            default=ModelConfig().num_trees_range[1],
            help=(
                "upper bound of the num.trees search range; lower it to trade "
                f"accuracy for speed (default: {ModelConfig().num_trees_range[1]})"
            ),
        )
        sub.add_argument(
            "--n-jobs",
            type=int,
            default=-1,
            help="parallel workers for tuning (-1 uses every core)",
        )
        sub.add_argument(
            "--n-mz",
            type=int,
            default=PreprocessConfig().n_mz_used,
            help=(
                "number of leading m/z channels to use "
                f"(default: {PreprocessConfig().n_mz_used})"
            ),
        )

    split = subparsers.add_parser(
        "train-test",
        help="stratified 70/30 split, tuned random forest, held-out evaluation",
    )
    add_common(split)
    split.add_argument(
        "--no-tune",
        action="store_true",
        help=(
            "skip the random search and use the hyper-parameters reported in the "
            f"paper (mtry={PAPER_MTRY}, num.trees={PAPER_NUM_TREES})"
        ),
    )
    split.add_argument(
        "--importance-csv",
        type=Path,
        default=None,
        help="write the impurity importance table to this CSV",
    )

    nested = subparsers.add_parser(
        "nested-resampling",
        help="nested resampling for an unbiased performance estimate",
    )
    add_common(nested)
    nested.add_argument(
        "--leak-free",
        action="store_true",
        help=(
            "refit peak alignment and feature filtering inside every outer fold "
            "instead of once over all samples (stricter than the R original)"
        ),
    )

    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        preprocess_config = replace(PreprocessConfig(), n_mz_used=args.n_mz)
        base_model_config = ModelConfig()
        model_config = replace(
            base_model_config,
            seed=args.seed,
            tuning_evals=args.tuning_evals,
            num_trees_range=(base_model_config.num_trees_range[0], args.max_trees),
            n_jobs=args.n_jobs,
        )

        if args.command == "train-test":
            result = train_test.run(
                args.data_dir,
                preprocess_config=preprocess_config,
                model_config=model_config,
                cluster_height=args.cluster_height,
                tune=not args.no_tune,
            )
            print(result.summary())
            if args.importance_csv is not None:
                args.importance_csv.parent.mkdir(parents=True, exist_ok=True)
                result.importances.to_csv(args.importance_csv, index=False)
                print(f"\nImportances written to {args.importance_csv}")
        else:
            result = nested_resampling.run(
                args.data_dir,
                preprocess_config=preprocess_config,
                model_config=model_config,
                cluster_height=args.cluster_height,
                leak_free=args.leak_free,
            )
            print(result.summary())
    except (OSError, ValueError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except MemoryError:
        print(
            "error: ran out of memory; try --n-mz with fewer channels or "
            "--n-jobs 1 to reduce the number of concurrent workers",
            file=sys.stderr,
        )
        return 1

    return 0
