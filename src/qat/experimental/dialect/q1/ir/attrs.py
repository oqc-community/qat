# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Data, ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from qat.experimental.dialect.q1.ir.arg_format import WithArgument


@irdl_attr_definition
class LabelAttr(WithArgument, Data[str]):
    name = "q1.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(self.data)

    def print_arg(self) -> str:
        return f"@{self.data}"


class DebugInfoAttr(ParametrizedAttribute, ABC):
    """Abstract base for debug information carried on a lowered Q1 op.

    Subclasses record structured debug metadata that the assembly printer uses to
    emit per-instruction inline comments.  The attribute is stripped in production
    builds by passing ``emit_debug_info=False`` to
    :func:`~qat.experimental.dialect.q1.target.emit_program`.

    Concrete implementations must provide :meth:`format_comment`.
    """

    @abstractmethod
    def format_comment(self) -> str:
        """Return a human-readable comment string for this debug info.

        :returns: Comment text to be appended to the assembly instruction.
        """


@irdl_attr_definition
class ProvenanceInfoAttr(DebugInfoAttr):
    """Provenance information carried on a lowered Q1 op.

    Records the originating pulse dialect operation and the physical channel port so
    that the assembly printer can emit per-instruction inline comments.

    :param source_op: The pulse dialect op name that this Q1 instruction was lowered
        from (e.g. ``"pulse.pulse"``, ``"pulse.acquire"``).
    :param port: The physical channel token from the enclosing
        ``q1_sequence.sequence`` symbol (e.g. ``"q0_drive"``).
    """

    name = "q1.provenance_info"

    source_op: StringAttr = param_def(StringAttr)
    port: StringAttr = param_def(StringAttr)

    def __init__(self, source_op: str, port: str) -> None:
        """
        :param source_op: Originating pulse op name.
        :param port: Physical channel token from the enclosing sequence.
        """
        super().__init__(StringAttr(source_op), StringAttr(port))

    def format_comment(self) -> str:
        """Return a human-readable comment string derived from the provenance info.

        :returns: A comment of the form ``"from <source_op> on <port>"``.
        """
        return f"from {self.source_op.data} on {self.port.data}"
