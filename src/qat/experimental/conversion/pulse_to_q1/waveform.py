# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Waveform rewrite strategies for Pulse-to-Q1 conversion."""

from typing import cast

from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.target_data import QbloxTargetData
from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    ConstantOp,
    PulseOp,
    SquareWaveformOp,
    TimeAttr,
)
from qat.experimental.dialect.pulse.units import TimeUnits
from qat.experimental.dialect.q1 import (
    DurationImm,
    SetAwgOffsImmImmOp,
    SI16Imm,
    UpdParamImmOp,
    WaitImmOp,
)
from qat.experimental.dialect.q1.ir.attrs import DebugInfoAttr
from qat.experimental.dialect.q1_sequence.ir.ops import find_enclosing_sequence
from qat.experimental.system_data.hardware.qblox.target import DEFAULT_QBLOX_TARGET


def _square_waveform(op: PulseOp) -> SquareWaveformOp:
    waveform = op.waveform.owner
    if not isinstance(waveform, SquareWaveformOp):
        raise TypeError(f"Expected pulse.square_waveform, got {waveform}.")
    return waveform


class SquareWaveformLegalisation:
    """Canonicalise a square waveform's width to integer nanoseconds."""

    def __call__(
        self,
        pulse_op: PulseOp,
        rewriter: PatternRewriter,
    ) -> None:
        """Legalise the square waveform consumed by ``op``.

        :param pulse_op: Square-waveform pulse operation to legalise.
        :param rewriter: Pattern rewriter used to replace the operation.
        """
        waveform = _square_waveform(pulse_op)
        width = waveform.width.owner
        if not isinstance(width, ConstantOp) or not isinstance(width.value, TimeAttr):
            raise PassFailedException(
                "pulse.square_waveform width must be a pulse.time constant."
            )

        width_ns = round(width.value.value_in_unit(TimeUnits.NANOSECOND))
        legalised_width = ConstantOp(TimeAttr(width_ns, TimeUnits.NANOSECOND))
        legalised_waveform = SquareWaveformOp(legalised_width, waveform.amplitude)
        legalised_pulse = PulseOp(pulse_op.frame, legalised_waveform)
        rewriter.replace_op(
            pulse_op,
            [legalised_width, legalised_waveform, legalised_pulse],
            (legalised_pulse.result,),
        )
        if not waveform.result.uses:
            rewriter.erase_op(waveform)


class SquareWaveformLowering:
    """Lower a canonical square pulse to latched Q1 AWG offsets.

    The non-zero offset remains live for exactly the Pulse waveform width. Latching the
    falling edge then advances the Q1 timeline by one additional sequencer grid interval.
    """

    def __call__(
        self,
        pulse_op: PulseOp,
        rewriter: PatternRewriter,
        target_data: QbloxTargetData,
        debug_info: DebugInfoAttr | None = None,
    ) -> None:
        """Lower the square waveform consumed by ``op``.

        :param pulse_op: Square-waveform pulse operation to lower.
        :param rewriter: Pattern rewriter used to replace the operation.
        :param target_data: Qblox limits used during lowering.
        :param debug_info: Optional source information copied to emitted instructions.
        """
        waveform = _square_waveform(pulse_op)
        width = waveform.width.owner
        if (
            not isinstance(width, ConstantOp)
            or not isinstance(width.value, TimeAttr)
            or width.value.unit.data is not TimeUnits.NANOSECOND
            or not isinstance(width.value.value.data, int)
        ):
            raise PassFailedException(
                "pulse.square_waveform width is not canonical. Run "
                "Q1PulseLegalisationPass before lowering."
            )

        amplitude = cast(ConstantOp, waveform.amplitude.owner)
        amplitude_value = cast(AmplitudeAttr, amplitude.value)

        sequence_op = find_enclosing_sequence(pulse_op)
        is_readout = (
            sequence_op.module_config is not None
            and sequence_op.seq_idx is not None
            and DEFAULT_QBLOX_TARGET.is_readout_sequencer(
                sequence_op.module_config.kind.data, sequence_op.seq_idx.data
            )
        )
        sequencer_data = (
            target_data.READOUT_SEQUENCER_DATA
            if is_readout
            else target_data.CONTROL_SEQUENCER_DATA
        )
        grid_time = sequencer_data.grid_time
        width_ns = width.value.value.data
        rise_duration = width_ns if width_ns < 2 * grid_time else grid_time
        remaining_duration = width_ns - rise_duration

        value = amplitude_value.literal_value
        max_offset = target_data.Q1ASM_DATA.max_offset
        q1_ops: list[Operation] = [
            SetAwgOffsImmImmOp(
                SI16Imm(int(value.real * max_offset)),
                SI16Imm(int(value.imag * max_offset)),
            ).with_debug_info(debug_info),
            UpdParamImmOp(DurationImm(rise_duration)).with_debug_info(debug_info),
        ]
        minimum_duration = DurationImm._MIN
        maximum_wait = target_data.Q1ASM_DATA.max_wait_time
        maximum_wait -= maximum_wait % grid_time
        if remaining_duration and maximum_wait < minimum_duration:
            raise PassFailedException(
                f"Q1 wait limit {target_data.Q1ASM_DATA.max_wait_time} ns cannot represent "
                f"the required {grid_time} ns sequencer alignment."
            )
        while remaining_duration > maximum_wait:
            wait_duration = maximum_wait
            if remaining_duration - wait_duration < minimum_duration:
                wait_duration -= minimum_duration - (remaining_duration - wait_duration)
            q1_ops.append(WaitImmOp(DurationImm(wait_duration)).with_debug_info(debug_info))
            remaining_duration -= wait_duration
        if remaining_duration:
            q1_ops.append(
                WaitImmOp(DurationImm(remaining_duration)).with_debug_info(debug_info)
            )
        q1_ops.extend(
            [
                SetAwgOffsImmImmOp(SI16Imm(0), SI16Imm(0)).with_debug_info(debug_info),
                UpdParamImmOp(DurationImm(grid_time)).with_debug_info(debug_info),
            ]
        )
        rewriter.replace_op(pulse_op, q1_ops, new_results=[pulse_op.frame])
        if not waveform.result.uses:
            rewriter.erase_op(waveform)
