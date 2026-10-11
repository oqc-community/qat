# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Verification helpers and pass for Q1 register allocation.

This module checks that every Q1 register value in a sequence has been fully allocated to a
physical register before later lowering passes consume it.
"""

from typing import cast

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.q1.ir.reg_desc import Q1RegisterType
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass


def _format_error_message(
    sequence: SequenceOp,
    op_name: str,
    value_type: str,
    idx: int,
    issue: str,
) -> str:
    """Build a readable allocation error for a value inside a sequence."""
    return (
        f"Sequence {sequence.channel_id.data!r}: {value_type} {idx} of operation "
        f"{op_name!r} is {issue}."
    )


def _check_value(
    value: SSAValue,
    sequence: SequenceOp,
    op_name: str,
    value_type: str,
    idx: int,
) -> str | None:
    """Return an error string when a Q1 value is not a valid physical register.

    Non-Q1 values are ignored. Unallocated registers and virtual registers whose indices are
    still negative are rejected.
    """
    reg_type = cast(Q1RegisterType, value.type)

    if not reg_type.is_allocated:
        return _format_error_message(
            sequence, op_name, value_type, idx, "an unallocated Q1 register"
        )

    if reg_type.index.data < 0:
        return _format_error_message(
            sequence,
            op_name,
            value_type,
            idx,
            f"a virtual Q1 register (inf_gpr_{~reg_type.index.data})",
        )

    return None


def verify_sequence_allocation(sequence: SequenceOp) -> list[str]:
    """Verify that all registers in a sequence are properly allocated.

    :param sequence: The sequence to verify.
    :returns: A list of distinct error messages for any allocation issues found.
    """
    errors_by_value: dict[SSAValue, str] = {}

    def record_error(
        value: SSAValue,
        op_name: str,
        value_type: str,
        idx: int,
    ) -> None:
        if value in errors_by_value:
            return

        error = _check_value(value, sequence, op_name, value_type, idx)
        if error is not None:
            errors_by_value[value] = error

    for block in sequence.body.blocks:
        for block_arg_idx, block_arg in enumerate(block.args):
            if isinstance(block_arg.type, Q1RegisterType):
                record_error(block_arg, "sequence", "block_arg", block_arg_idx)

    for op in sequence.walk():
        for operand_idx, operand in enumerate(op.operands):
            if isinstance(operand.type, Q1RegisterType):
                record_error(operand, op.name, "operand", operand_idx)

        for result_idx, result in enumerate(op.results):
            if isinstance(result.type, Q1RegisterType):
                record_error(result, op.name, "result", result_idx)

        for region in op.regions:
            for block in region.blocks:
                for arg_idx, block_arg in enumerate(block.args):
                    if isinstance(block_arg.type, Q1RegisterType):
                        record_error(block_arg, op.name, "block_arg", arg_idx)

    return list(errors_by_value.values())


class VerifyQ1AllocationPass(OrderedPass, ModulePass):
    """Reject any Q1 register value that is still unallocated or virtual.

    The pass walks every :class:`~qat.experimental.dialect.q1_sequence.ir.ops.SequenceOp`
    in the module, including nested regions, and aggregates all allocation issues into a
    single :class:`~xdsl.utils.exceptions.PassFailedException`.
    """

    name = "q1-verify-allocation"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Verify all sequences in the module have properly allocated registers.

        :param ctx: The pass context required by the xDSL module-pass interface.
        :param op: The module whose sequences should be validated.
        :raises PassFailedException: If any Q1 register remains unallocated or virtual.
        """
        all_errors: list[str] = []
        for sequence in (s for s in op.walk() if isinstance(s, SequenceOp)):
            errors = verify_sequence_allocation(sequence)
            all_errors.extend(errors)

        if all_errors:
            error_report = "\n".join(all_errors)
            raise PassFailedException(
                f"Register allocation verification failed. Found {len(all_errors)} "
                f"issue(s):\n{error_report}"
            )
