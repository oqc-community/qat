# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Q1 operation visitor for the generic schedule ledger."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import singledispatchmethod
from math import sqrt, tau

import numpy as np
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation

from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1.ir.ops import (
    AcquireImmImmImmOp,
    AcquireImmRsImmOp,
    AcquireTtlImmImmImmImmOp,
    AcquireTtlImmRsImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    AcquireWeightedImmRsRsRsImmOp,
    LatchRstImmOp,
    MoveImmRdOp,
    NopOp,
    PlayImmImmImmOp,
    PlayRsRsImmOp,
    ResetPhOp,
    SetAwgGainImmImmOp,
    SetAwgOffsImmImmOp,
    SetCondImmImmImmImmOp,
    SetCondRsRsRsImmOp,
    SetFreqImmOp,
    SetLatchEnImmImmOp,
    SetLatchEnRsImmOp,
    SetMrkImmOp,
    SetMrkRsOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    StopImmOp,
    StopOp,
    StopRsOp,
    UpdParamImmOp,
    WaitImmOp,
    WaitSyncImmOp,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    Q1SequencerType,
    QbloxTargetDescription,
)
from qat.experimental.tools.schedule import ResourceKind, ScheduleTracker


class Q1ScheduleVisitor:
    """Translate statically known Q1 instructions into ledger updates."""

    def __init__(
        self,
        tracker: ScheduleTracker,
        sequence: str = "sequence",
        target_description: QbloxTargetDescription = DEFAULT_QBLOX_TARGET,
        waveforms: Mapping[int, Iterable[float]] | None = None,
        sequencer_type: Q1SequencerType = Q1SequencerType.control,
    ) -> None:
        self.tracker = tracker
        self.sequence = sequence
        self._sequencer_spec = target_description.sequencer_spec(sequencer_type)
        self._awg_gain_scale = -target_description.q1asm.min_gain
        self._awg_offset_scale = -target_description.q1asm.min_offset
        self._waveforms = {
            index: np.asarray(
                samples if isinstance(samples, np.ndarray) else tuple(samples),
                dtype=float,
            )
            for index, samples in (waveforms or {}).items()
        }
        for index, samples in self._waveforms.items():
            if (
                not np.all(np.isfinite(samples))
                or np.any(samples < self._sequencer_spec.min_waveform_sample)
                or np.any(samples > self._sequencer_spec.max_waveform_sample)
            ):
                raise ValueError(
                    f"Q1 waveform {index} contains samples outside "
                    f"[{self._sequencer_spec.min_waveform_sample}, "
                    f"{self._sequencer_spec.max_waveform_sample}]"
                )
        self._active_waveform: np.ndarray | None = None
        self._waveform_cursor = 0
        self._active_phase_offset = 0.0
        self._pending_frequency: float | None = None
        self._pending_phase_offset: float | None = None
        self._pending_phase_delta = 0.0
        self._pending_phase_reset = False
        self._pending_gains: tuple[float, float] | None = None
        self._pending_offsets: tuple[float, float] | None = None
        self._active_gains = (1.0, 1.0)
        self._active_offsets = (0.0, 0.0)
        self.tracker.resource(
            sequence,
            ResourceKind.SEQUENCE,
            "ns",
            "DAC/ADC range",
            phase_label="NCO phase",
            phase_unit="steps",
            phase_scale=self._sequencer_spec.nco_phase_steps / tau,
            signal_limits=(
                self._sequencer_spec.min_waveform_sample,
                self._sequencer_spec.max_waveform_sample,
            ),
        )

    @singledispatchmethod
    def visit(self, operation: Operation) -> None:
        raise NotImplementedError(
            f"Q1 schedule visitor does not support {type(operation).__name__}"
        )

    @visit.register
    def _(self, operation: NopOp) -> None:
        self._advance(self._sequencer_spec.clock_period_ns, "nop")

    @visit.register(StopOp)
    @visit.register(StopImmOp)
    @visit.register(StopRsOp)
    def _(self, operation: StopOp | StopImmOp | StopRsOp) -> None:
        return None

    @visit.register
    def _(self, operation: WaitImmOp) -> None:
        self._advance(operation.duration.data, "wait")

    @visit.register
    def _(self, operation: WaitSyncImmOp) -> None:
        self._advance(operation.duration.data, "wait_sync")

    @visit.register
    def _(self, operation: PlayImmImmImmOp) -> None:
        self._play(
            operation.wave0.data,
            operation.wave1.data,
            operation.duration.data,
        )

    @visit.register
    def _(self, operation: PlayRsRsImmOp) -> None:
        self._play(
            self._static_register_value(operation.wave0),
            self._static_register_value(operation.wave1),
            operation.duration.data,
        )

    def _play(self, wave0: int, wave1: int, duration: int) -> None:
        self._apply_pending_parameters()
        self._active_waveform = self._waveform_pair(wave0, wave1)
        self._waveform_cursor = 0
        self._advance(duration, "play")
        self._active_waveform = None
        self._waveform_cursor = 0

    @staticmethod
    def _static_register_value(value) -> int:
        owner = value.owner
        if isinstance(owner, MoveImmRdOp):
            return owner.source.data
        owner_name = getattr(owner, "name", "block argument")
        raise ValueError(
            f"Q1 schedule plotting requires a static register value, got {owner_name}"
        )

    @visit.register(AcquireImmImmImmOp)
    @visit.register(AcquireImmRsImmOp)
    @visit.register(AcquireTtlImmImmImmImmOp)
    @visit.register(AcquireTtlImmRsImmImmOp)
    @visit.register(AcquireWeightedImmImmImmImmImmOp)
    @visit.register(AcquireWeightedImmRsRsRsImmOp)
    def _(
        self,
        operation: (
            AcquireImmImmImmOp
            | AcquireImmRsImmOp
            | AcquireTtlImmImmImmImmOp
            | AcquireTtlImmRsImmImmOp
            | AcquireWeightedImmImmImmImmImmOp
            | AcquireWeightedImmRsRsRsImmOp
        ),
    ) -> None:
        self._apply_pending_parameters()
        self._advance(operation.duration.data, "acquire")

    @visit.register
    def _(self, operation: SetFreqImmOp) -> None:
        frequency = (
            operation.nco_freq.data / self._sequencer_spec.nco_frequency_steps_per_hz
        )
        if not (
            self._sequencer_spec.nco_min_frequency_hz
            <= frequency
            <= self._sequencer_spec.nco_max_frequency_hz
        ):
            raise ValueError(
                f"Q1 NCO frequency {frequency} Hz is outside "
                f"[{self._sequencer_spec.nco_min_frequency_hz}, "
                f"{self._sequencer_spec.nco_max_frequency_hz}]"
            )
        self._pending_frequency = frequency

    @visit.register
    def _(self, operation: SetPhImmOp) -> None:
        self._pending_phase_offset = (
            tau * operation.nco_po.data / self._sequencer_spec.nco_phase_steps
        )

    @visit.register
    def _(self, operation: SetPhDeltaImmOp) -> None:
        self._pending_phase_delta = (
            tau * operation.nco_delta_po.data / self._sequencer_spec.nco_phase_steps
        )

    @visit.register(SetCondImmImmImmImmOp)
    @visit.register(SetCondRsRsRsImmOp)
    @visit.register(MoveImmRdOp)
    @visit.register(SetMrkImmOp)
    @visit.register(SetMrkRsOp)
    def _(
        self,
        operation: (
            SetCondImmImmImmImmOp
            | SetCondRsRsRsImmOp
            | MoveImmRdOp
            | SetMrkImmOp
            | SetMrkRsOp
        ),
    ) -> None:
        return None

    @visit.register(SetLatchEnImmImmOp)
    @visit.register(SetLatchEnRsImmOp)
    def _(self, operation: SetLatchEnImmImmOp | SetLatchEnRsImmOp) -> None:
        self._advance(operation.duration.data, "set_latch_en")

    @visit.register
    def _(self, operation: LatchRstImmOp) -> None:
        self._advance(operation.duration.data, "latch_rst")

    @visit.register
    def _(self, operation: ResetPhOp) -> None:
        self._pending_phase_reset = True
        self._pending_phase_offset = None
        self._pending_phase_delta = 0.0

    @visit.register
    def _(self, operation: SetAwgGainImmImmOp) -> None:
        self._pending_gains = (
            operation.gain0.data / self._awg_gain_scale,
            operation.gain1.data / self._awg_gain_scale,
        )

    @visit.register
    def _(self, operation: SetAwgOffsImmImmOp) -> None:
        self._pending_offsets = (
            operation.offs0.data / self._awg_offset_scale,
            operation.offs1.data / self._awg_offset_scale,
        )

    @visit.register
    def _(self, operation: UpdParamImmOp) -> None:
        self._apply_pending_parameters()
        self._advance(operation.duration.data, "upd_param")

    def _apply_pending_parameters(self) -> None:
        resource = self.tracker.select(self.sequence)
        if self._pending_phase_reset:
            resource.set_phase(0.0)
            self._active_phase_offset = 0.0
        if self._pending_frequency is not None:
            resource.set_frequency(self._pending_frequency)
        if self._pending_phase_offset is not None:
            resource.set_phase(
                self.tracker.phase + self._pending_phase_offset - self._active_phase_offset
            )
            self._active_phase_offset = self._pending_phase_offset
        if self._pending_phase_delta:
            resource.set_phase(self.tracker.phase + self._pending_phase_delta)
        if self._pending_gains is not None:
            self._active_gains = self._pending_gains
        if self._pending_offsets is not None:
            self._active_offsets = self._pending_offsets

        self._pending_frequency = None
        self._pending_phase_offset = None
        self._pending_phase_delta = 0.0
        self._pending_phase_reset = False
        self._pending_gains = None
        self._pending_offsets = None

    def _waveform_pair(
        self,
        wave0: int,
        wave1: int,
    ) -> np.ndarray:
        path0 = self._waveforms.get(wave0)
        path1 = self._waveforms.get(wave1)
        if path0 is None or path1 is None:
            missing = wave0 if path0 is None else wave1
            raise ValueError(f"Q1 play references missing waveform index {missing}")

        sample_count = max(path0.size, path1.size)
        envelope = np.zeros(sample_count, dtype=complex)
        envelope[: path0.size] = path0
        envelope[: path1.size] += 1j * path1
        return envelope

    def _advance(self, duration: int, label: str) -> None:
        sample_count = round(duration * self._sequencer_spec.sample_rate_hz * 1e-9)
        signal = np.zeros(sample_count, dtype=complex)
        if self._active_waveform is not None:
            remaining = self._active_waveform.size - self._waveform_cursor
            copied = min(sample_count, remaining)
            signal[:copied] = self._active_waveform[
                self._waveform_cursor : self._waveform_cursor + copied
            ]
            self._waveform_cursor += copied
            if self._waveform_cursor >= self._active_waveform.size:
                self._active_waveform = None
                self._waveform_cursor = 0
        gain0, gain1 = self._active_gains
        offset0, offset1 = self._active_offsets
        signal = (
            gain0 * signal.real + offset0 + 1j * (gain1 * signal.imag + offset1)
        ) / sqrt(2)
        self.tracker.select(self.sequence).advance(duration, signal=signal, label=label)


def build_q1_schedule(
    module: ModuleOp,
    sequence: str = "sequence",
    target: QbloxTargetDescription = DEFAULT_QBLOX_TARGET,
    waveforms: Mapping[int, Iterable[float]] | None = None,
    sequencer_type: Q1SequencerType = Q1SequencerType.control,
) -> ScheduleTracker:
    """Walk a Q1 module and record its linear instruction schedule.

    :param module: Q1 module containing Q1 assembly operations.
    :param sequence: Name assigned to the tracked Q1 sequence.
    :param target: Qblox target constants used for Q1 timing and NCO conversion.
    :param waveforms: Real waveform-memory samples keyed by Q1 table index for a flat
        module. Enclosed Q1 sequence operations use their own waveform tables.
    :param sequencer_type: Sequencer type used for flat or unbound Q1 IR. Bound sequence
        operations derive their type from their physical module configuration.
    :returns: The populated schedule tracker.
    """
    tracker = ScheduleTracker()
    from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp

    sequence_ops = [
        operation for operation in module.body.walk() if isinstance(operation, SequenceOp)
    ]
    if sequence_ops:
        if any(len(sequence_op.body.blocks) != 1 for sequence_op in sequence_ops):
            raise ValueError(
                "Q1 schedule plotting requires linearised single-block sequences"
            )
        if len(sequence_ops) > 1 and any(
            isinstance(operation, WaitSyncImmOp)
            for sequence_op in sequence_ops
            for operation in sequence_op.body.walk()
        ):
            raise ValueError(
                "Q1 schedule plotting does not support wait_sync across multiple sequences"
            )
        for sequence_op in sequence_ops:
            operations = tuple(sequence_op.body.block.ops)
            unsupported = next(
                (
                    operation
                    for operation in operations
                    if not isinstance(operation, Q1AsmOperation)
                ),
                None,
            )
            if unsupported is not None:
                raise ValueError(
                    f"Q1 schedule plotting requires flat Q1 operations, found "
                    f"{unsupported.name}"
                )
            waveform_table = {
                waveform.index.data: tuple(
                    float(value) for value in waveform.data.iter_values()
                )
                for waveform in sequence_op.waveforms
            }
            visitor = Q1ScheduleVisitor(
                tracker,
                sequence_op.channel_id.data,
                target,
                waveform_table,
                (
                    target.sequencer(
                        sequence_op.module_config.kind.data,
                        sequence_op.seq_idx.data,
                    ).sequencer_spec.type
                    if sequence_op.module_config is not None
                    and sequence_op.seq_idx is not None
                    else sequencer_type
                ),
            )
            for operation in operations:
                operation.accept(visitor)
        return tracker

    operations = tuple(module.body.block.ops)
    unsupported = next(
        (
            operation
            for operation in operations
            if not isinstance(operation, Q1AsmOperation)
        ),
        None,
    )
    if unsupported is not None:
        raise ValueError(
            f"Q1 schedule plotting requires flat Q1 operations, found {unsupported.name}"
        )
    visitor = Q1ScheduleVisitor(
        tracker,
        sequence,
        target,
        waveforms,
        sequencer_type,
    )
    for operation in operations:
        operation.accept(visitor)
    return tracker
