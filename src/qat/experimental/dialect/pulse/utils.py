# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Structural utilities for the Pulse dialect."""

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, SSAValue
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    TimeAttr,
    WaitOp,
)


def pulse_entry_block(module: ModuleOp) -> Block:
    """Return the block that carries the Pulse instruction stream.

    Current repository producers use two concrete module shapes:

    * Frontend importers build a single ``func.func @main`` and place Pulse ops in
      its body block.
    * Some transforms and unit tests build a flat module with Pulse ops at top-level.

    TODO(COMPILER-1380): remove this dual-shape logic once the canonical module shape
    is settled.

    :param module: The Pulse module to inspect.
    :returns: The entry block containing the Pulse instruction sequence.
    :raises PassFailedException: If the module contains more than one function, or
        mixes a function with other top-level operations.
    """
    top_level_ops = list(module.body.block.ops)
    func_ops = [op for op in top_level_ops if isinstance(op, func.FuncOp)]
    if not func_ops:
        return module.body.block
    if len(func_ops) == 1 and len(top_level_ops) != 1:
        raise PassFailedException(
            "A Pulse module must be either a flat module or a module containing only "
            "one entry function and no other top-level operations."
        )
    if len(func_ops) != 1:
        raise PassFailedException(
            "A Pulse module must contain a single entry function or no functions at all."
        )
    return func_ops[0].body.block


def require_constant_operand(
    op_name: str, operand_name: str, operand: SSAValue
) -> ConstantOp:
    """Return the defining ``pulse.constant`` op for an SSA operand.

    :param op_name: Qualified name of the enclosing operation, used in error messages.
    :param operand_name: Semantic label of the operand, used in error messages.
    :param operand: The SSA value whose defining operation must be a ``ConstantOp``.
    :returns: The ``ConstantOp`` defining the operand.
    :raises PassFailedException: If the operand is not defined by a ``ConstantOp``.
    """
    owner = operand.owner
    if not isinstance(owner, ConstantOp):
        raise PassFailedException(
            f"{op_name} requires constant {operand_name} at this stage. Dynamic "
            f"{operand_name} operands require dedicated legalisation."
        )
    return owner


def extract_time_seconds(op: WaitOp) -> float:
    """Extract a constant wait duration in seconds from ``pulse.wait``.

    :param op: The wait operation to extract from.
    :returns: Duration in seconds as a Python float.
    :raises PassFailedException: If the duration operand is not a constant or its
        attribute is not a ``TimeAttr``.
    """
    const = require_constant_operand(op.name, "duration", op.duration)
    folded = const.fold()
    if not folded or not isinstance(folded[0], TimeAttr):
        raise PassFailedException(f"{op.name} expects pulse.constant time operand.")
    return float(folded[0].literal_value)


def extract_phase_radians(op: PhaseSetOp | PhaseShiftOp) -> float:
    """Extract a constant phase in radians from a phase-manipulation operation.

    :param op: The phase operation to extract from.
    :returns: Phase angle in radians as a Python float.
    :raises PassFailedException: If the phase operand is not a constant or its
        attribute is not a ``PhaseAttr``.
    """
    const = require_constant_operand(op.name, "phase", op.phase)
    folded = const.fold()
    if not folded or not isinstance(folded[0], PhaseAttr):
        raise PassFailedException(f"{op.name} expects pulse.constant phase operand.")
    return float(folded[0].literal_value)


def extract_frequency_hz(op: CreateFrameOp) -> float:
    """Extract a constant frame frequency in Hertz from ``pulse.create_frame``.

    :param op: The create-frame operation to extract from.
    :returns: Frequency in Hertz as a Python float.
    :raises PassFailedException: If the frequency operand is not a constant or its
        attribute is not a ``FrequencyAttr``.
    """
    const = require_constant_operand(op.name, "frequency", op.frequency)
    folded = const.fold()
    if not folded or not isinstance(folded[0], FrequencyAttr):
        raise PassFailedException(f"{op.name} expects pulse.constant frequency operand.")
    return float(folded[0].literal_value)
