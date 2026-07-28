# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Structural utilities for the Pulse dialect."""

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block
from xdsl.utils.exceptions import PassFailedException


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
