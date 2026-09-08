# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Phase rewrite strategies for the Pulse-to-Q1 conversion.

``PhaseLegalisation`` canonicalises constant ``pulse.phase`` operands into the
``[0, 2π)`` range.

``PhaseLowering`` consumes canonical constants and converts radians to degrees,
then to NCO phase steps.

Dynamic phase operands are unsupported: Q1 phase instructions only accept an
immediate operand, so there is no numeric radians-to-steps runtime conversion for
a dynamic value. ``Q1PulseValidationPass`` rejects a dynamic ``pulse.phase_set`` or
``pulse.phase_shift`` operand upstream of both stages.
"""

from __future__ import annotations

from numpy import mod, pi, rad2deg
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

# TODO: Migrate this lowering boundary to QbloxTargetDescription.
from qat.backend.qblox.target_data import QbloxTargetData
from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
)
from qat.experimental.dialect.pulse.utils import extract_phase_radians
from qat.experimental.dialect.q1 import (
    DurationImm,
    NcoPhaseImm,
    SetPhDeltaImmOp,
    SetPhImmOp,
    UpdParamImmOp,
)
from qat.experimental.dialect.q1.ir.attrs import DebugInfoAttr


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
    """

    def __call__(
        self,
        op: PhaseSetOp | PhaseShiftOp,
        rewriter: PatternRewriter,
        target_data: QbloxTargetData,
        debug_info: DebugInfoAttr | None = None,
    ) -> None:
        """Apply the lowering-stage phase rewrite.

        :param op: The phase operation to rewrite.
        :param rewriter: Pattern rewriter used to replace the op in the IR.
        :param target_data: QBlox target description supplying NCO step-rate constants.
        """
        seq_data = target_data.CONTROL_SEQUENCER_DATA
        legalised_radians = extract_phase_radians(op)
        if not (0.0 <= legalised_radians < 2 * pi):
            raise PassFailedException(
                f"{op.name} phase operand is not canonical. Run Q1PulseLegalisationPass "
                "before lowering."
            )
        phase_deg = rad2deg(legalised_radians)
        steps = int(round(phase_deg * seq_data.nco_phase_steps_per_deg))
        steps %= seq_data.nco_max_phase_steps
        primary = (
            SetPhImmOp(NcoPhaseImm(steps))
            if isinstance(op, PhaseSetOp)
            else SetPhDeltaImmOp(NcoPhaseImm(steps))
        ).with_debug_info(debug_info)
        rewriter.replace_op(
            op,
            [primary, UpdParamImmOp(DurationImm(seq_data.grid_time))],
            (op.frame,),
        )
