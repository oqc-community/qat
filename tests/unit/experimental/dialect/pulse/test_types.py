# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the types in the pulse dialects."""

import pytest
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.pulse.ir import Pulse
from qat.experimental.dialect.pulse.ir.types import StateKeyType


class TestStateKeyType:
    """The state type carries a discriminated state that can take an integer label.

    The allowed states are expected to be consecutive, i.e., have no integer gaps that do
    not map to a possible state, allowing a representation from a minimum allowed state to a
    maximum. This class tests the methods around that, and also printing / parsing under
    those conditions.
    """

    def test_initialisation_with_valid_parameters_passes_verification(self):
        """Sanity test to check we can instantiate the type."""
        attr = StateKeyType(0, 1)
        attr.verify()

    def test_initialisation_with_invalid_inequalities_fails_verification(self):
        """Tests that when max < min, verification fails raising a VerifyException with the
        expected matching error message."""
        with pytest.raises(VerifyException, match="StateKeyType bounds invalid"):
            StateKeyType(2, 0).verify()

    def test_state_range_is_min_and_max(self):
        """Tests the state range corresponds to the minimum and maximum allowed states."""
        attr = StateKeyType(-1, 2)
        assert attr.state_range == (-1, 2)

    def test_print_returns_expected_string(self, io_stream):
        """Tests that printing the type returns the expected string, including the minimum
        and maximum range."""
        attr = StateKeyType(-1, 2)
        context = Context()
        context.load_dialect(Pulse)
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()
        assert output == "!pulse.state<-1, 2>"

    def test_parse_returns_correct_type(self):
        """Prepares a printed string and parses it into the expected type, testing that the
        properties of that type are correct."""
        attr_str = "!pulse.state<0, 1>"
        context = Context()
        context.load_dialect(Pulse)
        parser = Parser(context, attr_str)
        parsed = parser.parse_attribute()
        assert isinstance(parsed, StateKeyType)
        assert parsed.min_state.data == 0
        assert parsed.max_state.data == 1

    def test_print_and_parse_roundtrip_gives_correct_type(self, io_stream):
        """Tests that printing followed by parsing the resulting string results in an
        equivalent typed attribute."""
        attr = StateKeyType(-1, 2)
        context = Context()
        context.load_dialect(Pulse)
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()
        parser = Parser(context, output)
        parsed = parser.parse_attribute()
        assert parsed == attr
