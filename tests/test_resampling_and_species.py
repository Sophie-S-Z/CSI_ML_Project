"""Stratified partitioning and the hard-coded sample labels."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from pyrgcms.resampling import create_data_partition, train_test_indices
from pyrgcms.species import (
    SPECIES,
    excluded_indices,
    kept_indices,
    labels,
)

# --- species vector -----------------------------------------------------------


def test_species_vector_matches_the_published_counts():
    counts = Counter(SPECIES)
    assert len(SPECIES) == 141
    assert counts["A"] == 75
    assert counts["B"] == 59
    assert counts["C"] == 7


def test_excluded_and_kept_partition_the_vector():
    assert len(excluded_indices()) == 7
    assert len(kept_indices()) == 134
    assert set(excluded_indices()) & set(kept_indices()) == set()


def test_excluded_positions_are_the_c_samples():
    assert excluded_indices() == [11, 17, 18, 19, 20, 115, 116]


def test_labels_drop_the_excluded_samples():
    kept = labels()
    assert len(kept) == 134
    assert set(kept) == {"A", "B"}


def test_helpers_accept_a_custom_vector():
    custom = ["A", "C", "B"]
    assert excluded_indices(custom) == [1]
    assert kept_indices(custom) == [0, 2]
    assert labels(custom) == ["A", "B"]


# --- createDataPartition ------------------------------------------------------


def test_partition_takes_the_ceiling_of_each_class():
    y = ["A"] * 10 + ["B"] * 7
    (train,) = create_data_partition(y, p=0.7, random_state=0)

    chosen = Counter(y[i] for i in train)
    assert chosen["A"] == math.ceil(10 * 0.7)
    assert chosen["B"] == math.ceil(7 * 0.7)


def test_partition_is_sorted_and_free_of_duplicates():
    y = ["A"] * 20 + ["B"] * 20
    (train,) = create_data_partition(y, p=0.5, random_state=1)
    assert train.tolist() == sorted(train.tolist())
    assert len(set(train.tolist())) == len(train)


def test_partition_is_reproducible_for_a_given_seed():
    y = ["A"] * 30 + ["B"] * 20
    first = create_data_partition(y, p=0.7, random_state=7)[0]
    second = create_data_partition(y, p=0.7, random_state=7)[0]
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_partitions():
    y = ["A"] * 40 + ["B"] * 40
    first = create_data_partition(y, p=0.7, random_state=1)[0]
    second = create_data_partition(y, p=0.7, random_state=2)[0]
    assert first.tolist() != second.tolist()


def test_requesting_several_partitions_returns_several():
    y = ["A"] * 10 + ["B"] * 10
    assert len(create_data_partition(y, p=0.6, times=3, random_state=3)) == 3


def test_a_singleton_class_is_always_selected():
    y = ["A"] * 10 + ["B"]
    with pytest.warns(RuntimeWarning, match="single record"):
        (train,) = create_data_partition(y, p=0.5, random_state=0)
    assert any(y[i] == "B" for i in train)


def test_partition_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 2"):
        create_data_partition(["A"], p=0.5)
    with pytest.raises(ValueError, match="p must lie"):
        create_data_partition(["A", "B"], p=1.0)
    with pytest.raises(ValueError, match="times must be"):
        create_data_partition(["A", "B"], p=0.5, times=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        create_data_partition(np.zeros((3, 2)), p=0.5)


def test_partition_accepts_a_generator():
    y = ["A"] * 10 + ["B"] * 10
    rng = np.random.default_rng(5)
    (train,) = create_data_partition(y, p=0.5, random_state=rng)
    assert len(train) == 10


# --- train/test split ---------------------------------------------------------


def test_train_and_test_are_complementary():
    y = ["A"] * 25 + ["B"] * 15
    train, test = train_test_indices(y, p=0.7, random_state=99)

    assert set(train.tolist()) | set(test.tolist()) == set(range(40))
    assert set(train.tolist()) & set(test.tolist()) == set()


def test_both_partitions_keep_both_classes():
    y = ["A"] * 75 + ["B"] * 59
    train, test = train_test_indices(y, p=0.7, random_state=99)

    assert {y[i] for i in train} == {"A", "B"}
    assert {y[i] for i in test} == {"A", "B"}


def test_split_sizes_match_the_real_dataset():
    train, test = train_test_indices(labels(), p=0.7, random_state=99)
    # ceil(75 * .7) + ceil(59 * .7) = 53 + 42 = 95
    assert len(train) == 95
    assert len(test) == 134 - 95
