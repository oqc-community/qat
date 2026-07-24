# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the post-selection predicate attributes in the results dialect."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.results.ir import IntegerStatePredicateAttr


class TestIntegerStatePredicateAttr:
    """Tests the IntegerStatePredicateAttr, which models a predicate for post-selecting
    results based on an integer state."""

    def test_initialization_and_properties(self):
        """Tests that the IntegerStatePredicateAttr can be initialized with a key and a list
        of disallowed values, and that its properties return the expected values."""
        attr = IntegerStatePredicateAttr("state", [0, 1, 2])

        assert attr.key == StringAttr("state")
        assert isinstance(attr.disallowed_values, ArrayAttr)
        assert attr.disallowed_values == ArrayAttr([IntAttr(0), IntAttr(1), IntAttr(2)])
        attr.verify()

    def test_verify_fails_when_disallowed_values_are_not_all_int_attrs(self):
        """Tests that verification fails when the disallowed values are not all IntAttr."""
        with pytest.raises(
            VerifyException,
            match="should be of base attribute",
        ):
            IntegerStatePredicateAttr(
                "state",
                ArrayAttr([IntAttr(0), StringAttr("not-an-int")]),
            )
