# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :class:`EnvironmentTracker`."""

from typing import cast

import pytest
from xdsl.ir import Attribute, SSAValue

from qat.experimental.frontend.importer.environment import EnvironmentTracker


def _ssa() -> SSAValue[Attribute]:
    """Create a unique placeholder SSA value for tracker tests."""
    return cast(SSAValue[Attribute], object())


class TestEnvironmentTracker:
    def test_get_by_name_returns_none_by_default_for_missing_key(self):
        """Missing names should return None when no default is provided."""
        tracker = EnvironmentTracker[Attribute]()
        assert tracker.get_by_name("missing") is None

    def test_get_by_name_returns_explicit_default_for_missing_key(self):
        """Missing names should return the caller-provided default value."""
        tracker = EnvironmentTracker[Attribute]()
        default = _ssa()
        assert tracker.get_by_name("missing", default) is default

    def test_set_by_name_then_get_by_name_returns_value(self):
        """Values set by name should be retrievable by that same name."""
        tracker = EnvironmentTracker[Attribute]()
        value = _ssa()

        tracker.set_by_name("q0", value)

        assert tracker.get_by_name("q0") is value

    def test_set_by_name_overwrites_existing_name(self):
        """Rebinding a name should replace its previous SSA value."""
        tracker = EnvironmentTracker[Attribute]()
        old_value = _ssa()
        new_value = _ssa()

        tracker.set_by_name("q0", old_value)
        tracker.set_by_name("q0", new_value)

        assert tracker.get_by_name("q0") is new_value

    def test_set_by_name_forceput_reassigns_shared_value_to_latest_name(self):
        """A shared value should move to the latest name due to forceput semantics."""
        tracker = EnvironmentTracker[Attribute]()
        shared_value = _ssa()

        tracker.set_by_name("q0", shared_value)
        tracker.set_by_name("q1", shared_value)

        assert tracker.get_by_name("q0") is None
        assert tracker.get_by_name("q1") is shared_value

    def test_set_by_value_updates_existing_binding(self):
        """Updating by old SSA value should preserve the original name key."""
        tracker = EnvironmentTracker[Attribute]()
        old_value = _ssa()
        new_value = _ssa()

        tracker.set_by_name("q0", old_value)
        tracker.set_by_value(old_value, new_value)

        assert tracker.get_by_name("q0") is new_value

    def test_set_by_value_raises_key_error_for_unknown_value(self):
        """Unknown old values should raise KeyError when rebinding by value."""
        tracker = EnvironmentTracker[Attribute]()

        with pytest.raises(KeyError):
            tracker.set_by_value(_ssa(), _ssa())

    def test_items_returns_name_value_pairs(self):
        """Items() should expose the current name-to-value mappings."""
        tracker = EnvironmentTracker[Attribute]()
        v0 = _ssa()
        v1 = _ssa()

        tracker.set_by_name("q0", v0)
        tracker.set_by_name("q1", v1)

        items = dict(tracker.items())
        assert items == {"q0": v0, "q1": v1}
