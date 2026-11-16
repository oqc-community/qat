# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Phase rewrite strategies for the Pulse-to-Q1 conversion.

``PhaseLegalisation`` canonicalises constant ``pulse.phase`` operands into the
``[0, 2π)`` range.

``PhaseLowering`` consumes canonical constants and converts radians to degrees,
then to NCO phase steps.

Q1 parameter instructions (``set_ph``, ``set_ph_delta``, ``set_freq`` and
``set_awg_offs``) take no time and do not latch themselves: the next ``upd_param``,
``play`` or ``acquire`` applies them. Lowerings therefore emit parameter instructions
without an ``upd_param``, and every Pulse wait lowers to ``upd_param`` so pending changes
apply at their Pulse time. A lowering that adds ``set_freq`` must follow the same rule.

Dynamic phase operands are unsupported: Q1 phase instructions only accept an
immediate operand, so there is no numeric radians-to-steps runtime conversion for
a dynamic value. ``Q1PulseValidationPass`` rejects a dynamic ``pulse.phase_set`` or
``pulse.phase_shift`` operand upstream of both stages.
"""

from __future__ import annotations

from numpy import mod, pi, rad2deg
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
)
from qat.experimental.dialect.pulse.utils import extract_phase_radians
from qat.experimental.dialect.q1 import NcoPhaseImm, SetPhDeltaImmOp, SetPhImmOp
from qat.experimental.dialect.q1.ir.attrs import DebugInfoAttr
from qat.experimental.dialect.q1_sequence.ir.ops import find_enclosing_sequence
from qat.experimental.system_data.hardware.qblox.target import (
    Q1SequencerType,
    QbloxTargetDescription,
)


class PhaseLegalisation:
    """Callable strategy that canonicalises a ``pulse.phase_set`` or ``pulse.phase_shift``
    operand to the ``[0, 2π)`` radian range and replaces the op with an equivalent Pulse op
    carrying the normalised constant.

    The stage stays within the Pulse dialect. Constant operand finiteness is enforced
    upstream by ``Q1PulseValidationPass``, which also rejects a dynamic phase operand
    before this stage runs.
    """

    def __call__(
        self,
        op: PhaseSetOp | PhaseShiftOp,
        rewriter: PatternRewriter,
    ) -> None:
        """Legalise the phase operand of ``op`` to the canonical radian range.

        :param op: The phase operation to rewrite.
        :param rewriter: Pattern rewriter used to replace the op in the IR.
        """
        legalised_radians = mod(extract_phase_radians(op), 2 * pi)
        new_const = ConstantOp(PhaseAttr(float(legalised_radians)))
        new_op = type(op)(op.frame, new_const)
        rewriter.replace_op(op, [new_const, new_op], (new_op.result,))


class PhaseLowering:
    """Callable strategy for the lowering stage.

    Lowering is distinct from legalisation. Canonical Pulse constants map to immediate Q1
    phase instructions.

    Conversion from radians is lossy. Values are quantised into sequencer NCO phase-step
    space prior to emission.

    The emitted ``set_ph`` or ``set_ph_delta`` is not followed by ``upd_param``: the next
    timed instruction latches it, so the phase update takes no time, as in the Pulse
    schedule.
    """

    def __call__(
        self,
        op: PhaseSetOp | PhaseShiftOp,
        rewriter: PatternRewriter,
        target_data: QbloxTargetDescription,
        debug_info: DebugInfoAttr | None = None,
    ) -> None:
        """Apply the lowering-stage phase rewrite.

        :param op: The phase operation to rewrite.
        :param rewriter: Pattern rewriter used to replace the op in the IR.
        :param target_data: Qblox target data supplying NCO step-rate constants.
        :param debug_info: Optional source location copied to emitted Q1 instructions.
        """
        sequence = find_enclosing_sequence(op)
        is_readout = (
            sequence.module_config is not None
            and sequence.seq_idx is not None
            and target_data.is_readout_sequencer(
                sequence.module_config.kind.data, sequence.seq_idx.data
            )
        )
        sequencer_spec_type = (
            Q1SequencerType.readout if is_readout else Q1SequencerType.control
        )
        sequencer_data = target_data.sequencer_spec(sequencer_spec_type)
        legalised_radians = extract_phase_radians(op)
        if not (0.0 <= legalised_radians < 2 * pi):
            raise PassFailedException(
                f"{op.name} phase operand is not canonical. Run Q1PulseLegalisationPass "
                "before lowering."
            )
        phase_deg = rad2deg(legalised_radians)
        steps = int(round(phase_deg * sequencer_data.nco_phase_steps_per_degree))
        steps %= sequencer_data.nco_phase_steps
        phase_op = (
            SetPhImmOp(NcoPhaseImm(steps))
            if isinstance(op, PhaseSetOp)
            else SetPhDeltaImmOp(NcoPhaseImm(steps))
        ).with_debug_info(debug_info)
        # TODO(COMPILER-1809): Replace this per-lowering latching rule with the decided
        # parameter-latching model.
        # TODO(COMPILER-1810): Two phase updates pending at once may overwrite or add on
        # hardware. FoldContiguousPhases merges consecutive phase ops on a frame before
        # lowering, so this lowering does not emit two in a row within a block. It does
        # not merge across control flow: a loop body that ends with a phase update, then
        # starts the next iteration with one, leaves two pending. Shot loops are safe,
        # because LatchPendingParametersPass latches before each wait_sync.
        rewriter.replace_op(op, phase_op, (op.frame,))
