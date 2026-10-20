# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Materialise Qblox whole-sequence initialisation after hardware binding."""

from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.conversion.pulse_to_q1.hardware_binding import (
    QbloxHardwareBindingPass,
)
from qat.experimental.dialect.q1 import (
    BoolImm,
    DurationImm,
    SetLatchEnImmImmOp,
    SetMrkImmOp,
    StopOp,
    UI4Imm,
    UpdParamImmOp,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass
from qat.experimental.system_data.hardware.qblox import (
    DEFAULT_QBLOX_TARGET,
    QbloxTargetDescription,
)
from qat.experimental.system_data.hardware.qblox.target import MarkerBitRole


@dataclass(frozen=True)
class QbloxSequenceInitialisationPass(OrderedPass, ModulePass):
    """Materialise Qblox whole-sequence initialisation.

    The pass owns the sequence layout ``[prologue, payload, epilogue, q1.stop]``. It
    derives the initial marker state from the bound module's physical marker routing,
    then inserts the marker, latch-enable, and applying update prologue before the
    outlined payload. ``q1.stop`` remains the final structural terminator. A future
    convention epilogue would be materialised by this pass immediately before that
    terminator; the epilogue is empty today.

    :ivar target_data: Qblox marker routing and instruction timing.
    """

    name = "qblox-sequence-initialisation"
    target_data: QbloxTargetDescription = field(default=DEFAULT_QBLOX_TARGET)

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        """Return the pass that must bind physical hardware before this pass.

        :returns: The physical hardware-binding pass.
        """

        return frozenset({QbloxHardwareBindingPass})

    def runs_before(self) -> frozenset[type[ModulePass]]:
        """Return the validation pass that must run after this pass.

        :returns: The Q1 Pulse validation pass.
        """

        from qat.experimental.conversion.pulse_to_q1.passes import Q1PulseValidationPass

        return frozenset({Q1PulseValidationPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Materialise the lowering convention in every bound sequence.

        :param ctx: The xDSL context, unused but required by the pass interface.
        :param op: Module containing outlined and bound Q1 sequences.
        :raises PassFailedException: If a sequence is empty, unbound, or does not end in
            ``q1.stop``.
        """

        for sequence in op.body.block.ops:
            if isinstance(sequence, SequenceOp):
                self._materialise(sequence)

    def _materialise(self, sequence: SequenceOp) -> None:
        """Insert one sequence's convention operations."""

        first_op = sequence.body.block.first_op
        if first_op is None:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} has an empty body."
            )

        terminator = sequence.body.block.last_op
        if not isinstance(terminator, StopOp):
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} must end with q1 stop before "
                "Qblox sequence initialisation."
            )

        if sequence.module_config is None:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} must be hardware-bound before "
                "Qblox sequence initialisation."
            )

        module_spec = self.target_data.module_spec(sequence.module_config.kind.data)
        marker_targets = (
            module_spec.marker_bit_0,
            module_spec.marker_bit_1,
            module_spec.marker_bit_2,
            module_spec.marker_bit_3,
        )
        marker_value = sum(
            1 << index
            for index, target in enumerate(marker_targets)
            if target is not None and target.role is MarkerBitRole.output_switch
        )
        duration = DurationImm(self.target_data.q1asm.min_rt_duration_ns)
        prologue_operations = (
            SetMrkImmOp(UI4Imm(marker_value)),
            SetLatchEnImmImmOp(
                BoolImm(1),
                duration,
            ),
            UpdParamImmOp(duration),
        )
        epilogue_operations = ()
        sequence.body.block.insert_ops_before(prologue_operations, first_op)
        sequence.body.block.insert_ops_before(epilogue_operations, terminator)
