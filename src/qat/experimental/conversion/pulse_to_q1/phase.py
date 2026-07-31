# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Phase rewrite strategies for the Pulse-to-Q1 conversion.

``PhaseLegalisation`` canonicalises constant ``pulse.phase`` operands into the
``[0, 2π)`` range. Dynamic operands are rewritten with
``builtin.unrealized_conversion_cast`` to an unallocated integer register.
This cast is a staged type bridge. It does not define numeric conversion.

``PhaseLowering`` consumes canonical constants and casted dynamic operands.
The constant path converts radians to degrees, then to NCO phase steps.
The dynamic path expects an integer register already interpreted as NCO phase
steps and normalises it into the valid sequencer interval.
"""

from __future__ import annotations

import numpy as np
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.target_data import QbloxTargetData
from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
)
from qat.experimental.dialect.pulse.utils import extract_phase_radians
from qat.experimental.dialect.q1 import (
    AddRsImmRdOp,
    CmpRsImmOp,
    DurationImm,
    JaeImmOp,
    JbImmOp,
    JgeImmOp,
    JlImmOp,
    LabelOp,
    NcoPhaseImm,
    NopOp,
    Registers,
    SetPhDeltaImmOp,
    SetPhDeltaRsOp,
    SetPhImmOp,
    SetPhRsOp,
    SU32Imm,
    SubRsImmRdOp,
    UpdParamImmOp,
)


class PhaseLegalisation:
    """Callable strategy that canonicalises a ``pulse.phase_set`` or ``pulse.phase_shift``
    operand to the ``[0, 2π)`` radian range and replaces the op with an equivalent Pulse op
    carrying the normalised constant.

    The stage stays within the Pulse dialect. Constant operand finiteness is enforced
    upstream by ``Q1PulseValidationPass``.

    Dynamic operands are rewritten with ``builtin.unrealized_conversion_cast``
    into an unallocated integer register. This is a staged typing contract.
    It is not a numeric radians-to-steps conversion.
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
        if not isinstance(op.phase.owner, ConstantOp):
            cast_op = UnrealizedConversionCastOp.get(
                [op.phase], [Registers.UNALLOCATED_INT]
            )
            dynamic_op = type(op)(op.frame, cast_op.results[0])
            rewriter.replace_op(op, [cast_op, dynamic_op], (dynamic_op.result,))
            return
        legalised_radians = np.mod(extract_phase_radians(op), 2 * np.pi)
        new_const = ConstantOp(PhaseAttr(float(legalised_radians)))
        new_op = type(op)(op.frame, new_const)
        rewriter.replace_op(op, [new_const, new_op], (new_op.result,))


class PhaseLowering:
    """Callable strategy for the lowering stage.

    Lowering is distinct from legalisation. Canonical Pulse constants map to immediate Q1
    phase instructions. Dynamic operands map to register Q1 phase instructions with modulo-
    loop normalisation.

    Constant conversion from radians is lossy. Values are quantised into sequencer NCO
    phase-step space prior to emission.
    """

    def __call__(
        self,
        op: PhaseSetOp | PhaseShiftOp,
        rewriter: PatternRewriter,
        target_data: QbloxTargetData,
    ) -> None:
        """Apply the lowering-stage phase rewrite.

        :param op: The phase operation to rewrite.
        :param rewriter: Pattern rewriter used to replace the op in the IR.
        :param target_data: QBlox target description supplying NCO step-rate constants.
            Dynamic operands must arrive as integer-register SSA values interpreted as NCO
            phase steps. Lowering then applies modulo-loop normalisation and emits register-
            form Q1 phase instructions.
        """
        seq_data = target_data.CONTROL_SEQUENCER_DATA
        if not isinstance(op.phase.owner, ConstantOp):
            phase_mod_loop_base = f"{op.name.replace('.', '_')}_{id(op)}"
            negative_loop_label = f"{phase_mod_loop_base}_negative_loop"
            non_negative_label = f"{phase_mod_loop_base}_non_negative"
            reduce_loop_label = f"{phase_mod_loop_base}_reduce_loop"
            done_label = f"{phase_mod_loop_base}_done"
            phase_steps_imm = SU32Imm(seq_data.nco_max_phase_steps)
            zero_imm = SU32Imm(0)
            phase_steps = op.phase
            dynamic_phase_ops = [
                CmpRsImmOp(phase_steps, zero_imm),
                JgeImmOp(non_negative_label),
                LabelOp(negative_loop_label),
                AddRsImmRdOp(phase_steps, phase_steps_imm, phase_steps),
                NopOp(),
                CmpRsImmOp(phase_steps, zero_imm),
                JlImmOp(negative_loop_label),
                LabelOp(non_negative_label),
                CmpRsImmOp(phase_steps, phase_steps_imm),
                JbImmOp(done_label),
                LabelOp(reduce_loop_label),
                SubRsImmRdOp(phase_steps, phase_steps_imm, phase_steps),
                NopOp(),
                CmpRsImmOp(phase_steps, phase_steps_imm),
                JaeImmOp(reduce_loop_label),
                LabelOp(done_label),
                SetPhRsOp(phase_steps)
                if isinstance(op, PhaseSetOp)
                else SetPhDeltaRsOp(phase_steps),
                UpdParamImmOp(DurationImm(seq_data.grid_time)),
            ]
            rewriter.replace_op(op, dynamic_phase_ops, (op.frame,))
            return
        legalised_radians = extract_phase_radians(op)
        if not (0.0 <= legalised_radians < 2 * np.pi):
            raise PassFailedException(
                f"{op.name} phase operand is not canonical. Run Q1PulseLegalisationPass "
                "before lowering."
            )
        phase_deg = np.rad2deg(legalised_radians)
        steps = int(round(phase_deg * seq_data.nco_phase_steps_per_deg))
        steps %= seq_data.nco_max_phase_steps
        rewriter.replace_op(
            op,
            [
                SetPhImmOp(NcoPhaseImm(steps)),
                UpdParamImmOp(DurationImm(seq_data.grid_time)),
            ]
            if isinstance(op, PhaseSetOp)
            else [
                SetPhDeltaImmOp(NcoPhaseImm(steps)),
                UpdParamImmOp(DurationImm(seq_data.grid_time)),
            ],
            (op.frame,),
        )
