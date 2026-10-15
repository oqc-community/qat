# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Waveform rewrite strategies for Pulse-to-Q1 conversion."""

from typing import cast

from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    ConstantOp,
    PulseOp,
    SquareWaveformOp,
    TimeAttr,
)
from qat.experimental.dialect.q1 import (
    DurationImm,
    SetAwgOffsImmImmOp,
    SI16Imm,
    UpdParamImmOp,
    WaitDurationImm,
    WaitImmOp,
)
from qat.experimental.dialect.q1.ir.attrs import DebugInfoAttr
from qat.experimental.system_data.hardware.qblox.target import (
    Q1AsmSpec,
    QbloxTargetDescription,
)


def split_wait_duration(duration_ns: int, q1asm: Q1AsmSpec) -> tuple[int, ...]:
    """Split a pulse-level wait into exactly representable normal Q1 waits.

    The special ``wait 0`` encoding is never emitted here. A final remainder below the
    minimum non-zero RT duration is made up by shortening the preceding chunks.

    :param duration_ns: Total pulse-level wait duration in nanoseconds.
    :param q1asm: Q1 instruction timing limits.
    :returns: Wait-immediate durations whose exact sum is ``duration_ns``.
    :raises PassFailedException: If the duration cannot be represented using normal waits.
    """

    if duration_ns == 0:
        return ()

    minimum = q1asm.min_rt_duration_ns
    maximum = q1asm.max_wait_immediate_ns
    resolution = q1asm.rt_duration_resolution_ns
    if duration_ns < minimum or duration_ns % resolution:
        raise PassFailedException(
            f"Q1 wait duration {duration_ns} ns is not representable by normal waits "
            f"with minimum {minimum} ns and resolution {resolution} ns."
        )

    full_chunks, remainder = divmod(duration_ns, maximum)
    chunks = [maximum] * full_chunks
    if remainder == 0:
        return tuple(chunks)
    if remainder >= minimum:
        chunks.append(remainder)
        return tuple(chunks)

    deficit = minimum - remainder
    for index in range(len(chunks) - 1, -1, -1):
        transfer = min(deficit, chunks[index] - minimum)
        chunks[index] -= transfer
        deficit -= transfer
        if deficit == 0:
            break
    if deficit:
        raise PassFailedException(
            f"Q1 wait duration {duration_ns} ns cannot be split into waits in "
            f"[{minimum}, {maximum}] ns at {resolution} ns resolution."
        )
    chunks.append(minimum)
    return tuple(chunks)


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

        The waveform width is expected to be in picoseconds (canonical Pulse IR unit) and
        will be rounded to integer nanoseconds for Q1 consumption.

        :param pulse_op: Square-waveform pulse operation to legalise.
        :param rewriter: Pattern rewriter used to replace the operation.
        """
        waveform = _square_waveform(pulse_op)
        width_ps_op = waveform.width.owner
        if not isinstance(width_ps_op, ConstantOp) or not isinstance(
            width_ps_op.value, TimeAttr
        ):
            raise PassFailedException(
                "pulse.square_waveform width must be a pulse.time constant."
            )

        width_ps = round(width_ps_op.value.literal_value, -3)
        legalised_width = ConstantOp(TimeAttr(width_ps))
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
    falling edge then advances the Q1 timeline by one minimum RT duration.
    """

    def __call__(
        self,
        pulse_op: PulseOp,
        rewriter: PatternRewriter,
        target_data: QbloxTargetDescription,
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
            or width.value.literal_value % 1000 != 0
        ):
            raise PassFailedException(
                "pulse.square_waveform width must be an integer number of nanoseconds "
                "(a multiple of 1000 ps). Run Q1PulseLegalisationPass before lowering."
            )

        amplitude = cast(ConstantOp, waveform.amplitude.owner)
        amplitude_value = cast(AmplitudeAttr, amplitude.value)

        width_ns = width.value.literal_value // 1000
        minimum_duration = target_data.q1asm.min_rt_duration_ns
        rise_duration = width_ns if width_ns < 2 * minimum_duration else minimum_duration
        remaining_duration = width_ns - rise_duration
        if rise_duration > target_data.q1asm.max_rt_duration_ns:
            raise PassFailedException(
                f"Square waveform rising-edge duration {rise_duration} ns exceeds the "
                f"maximum RT duration {target_data.q1asm.max_rt_duration_ns} ns."
            )

        value = amplitude_value.literal_value
        max_offset = target_data.q1asm.max_offset
        q1_ops: list[Operation] = [
            SetAwgOffsImmImmOp(
                SI16Imm(int(value.real * max_offset)),
                SI16Imm(int(value.imag * max_offset)),
            ).with_debug_info(debug_info),
            UpdParamImmOp(DurationImm(rise_duration)).with_debug_info(debug_info),
        ]
        for wait_duration in split_wait_duration(remaining_duration, target_data.q1asm):
            q1_ops.append(
                WaitImmOp(WaitDurationImm(wait_duration)).with_debug_info(debug_info)
            )
        q1_ops.extend(
            [
                SetAwgOffsImmImmOp(SI16Imm(0), SI16Imm(0)).with_debug_info(debug_info),
                UpdParamImmOp(DurationImm(minimum_duration)).with_debug_info(debug_info),
            ]
        )
        rewriter.replace_op(pulse_op, q1_ops, new_results=[pulse_op.frame])
        if not waveform.result.uses:
            rewriter.erase_op(waveform)
