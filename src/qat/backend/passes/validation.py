# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from qat.core.pass_base import ValidationPass
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.measure import Acquire


class NoAcquireWeightsValidation(ValidationPass):
    """Some target machines do not support :class:`Acquire` instructions that contain weights.
    This pass can be used to validate that this is the case."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        has_filters = [inst.filter for inst in ir.instructions if isinstance(inst, Acquire)]
        if any(has_filters):
            raise NotImplementedError(
                "Acquire filters are not implemented for this target machine."
            )
        return ir


PydNoAcquireWeightsValidation = NoAcquireWeightsValidation
