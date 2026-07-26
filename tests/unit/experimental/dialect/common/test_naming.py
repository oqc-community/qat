# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.common.naming`."""

from __future__ import annotations

from qat.experimental.dialect.common.naming import assign_unique_name, assign_unique_names


def test_names_follow_the_prefix_scheme_when_unreserved():
    keys = ["a", "b", "c"]
    assert assign_unique_names(keys, set(), "bb") == {"a": "bb0", "b": "bb1", "c": "bb2"}


def test_reserved_names_are_skipped():
    names = assign_unique_names(["a", "b"], {"bb0", "bb2"}, "bb")
    assert names == {"a": "bb1", "b": "bb3"}


def test_names_stay_distinct_across_keys_and_reservations():
    names = assign_unique_names(["a", "b", "c"], {"bb1"}, "bb")
    assert len(set(names.values())) == len(names)
    assert "bb1" not in names.values()


def test_no_keys_yields_no_names():
    assert assign_unique_names([], {"bb0"}, "bb") == {}


def test_a_distinct_prefix_is_honoured():
    assert assign_unique_names(["x"], set(), "L") == {"x": "L0"}


def test_unique_name_uses_preferred_name_when_available():
    assert assign_unique_name({"bb0"}, "bb_", preferred_name="bb_halt") == "bb_halt"


def test_unique_name_falls_back_to_prefix_counter_when_preferred_is_taken():
    assert (
        assign_unique_name({"bb_halt", "bb_halt_0"}, "bb_halt_", "bb_halt") == "bb_halt_1"
    )
