# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer

from qat.experimental.dialect.q1 import Q1, LabelAttr


def test_label_attr_name_and_data():
    attr = LabelAttr("loop_start")

    assert LabelAttr.name == "q1.label"
    assert attr.data == "loop_start"


def test_label_attr_print_parse_roundtrip():
    attr = LabelAttr("loop_start")
    output = StringIO()
    printer = Printer(stream=output)
    printer.print_attribute(attr)

    encoded_attr = output.getvalue()
    assert encoded_attr == '#q1.label"loop_start"'

    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(Q1)
    parser = Parser(context, encoded_attr)
    parsed_attr = parser.parse_attribute()
    assert parsed_attr == attr
