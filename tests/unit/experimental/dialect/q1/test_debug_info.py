# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for debug info auto-attachment during pulse-to-Q1 lowering.

Covers:
- :class:`~qat.experimental.dialect.q1.ir.attrs.ProvenanceInfoAttr` construction
  (isolation).
- Automatic ``debug_info`` attachment to Q1 ops by the lowering rewrite patterns.
"""

import numpy as np
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation

from qat.backend.qblox.target_data import TARGET_DATA
from qat.experimental.conversion.pulse_to_q1.passes import (
    PulseToQ1LoweringPass,
    Q1PulseLegalisationPass,
)
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    AmplitudeType,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    SampledWaveformAttr,
    StartContinuousWaveformOp,
    TimeAttr,
    WaitOp,
    WaveformType,
)
from qat.experimental.dialect.pulse.units import TimeUnits
from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    EmissionContext,
    PlayImmImmImmOp,
    ProvenanceInfoAttr,
    SetAwgOffsImmImmOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    StopOp,
    WaitImmOp,
)
from qat.experimental.dialect.q1_sequence import SequenceOp


def _sequence_module(*ops, channel_id: str = "q0_drive") -> ModuleOp:
    return ModuleOp([SequenceOp(channel_id, [*ops, StopOp()])])


def _frame(channel_id: str = "q0/drive"):
    freq = ConstantOp(FrequencyAttr(4.8e9))
    return freq, CreateFrameOp(freq, StringAttr(channel_id))


def _run_q1_pipeline(module: ModuleOp) -> None:
    Q1PulseLegalisationPass().apply(Context(), module)
    PulseToQ1LoweringPass().apply(Context(), module)


def _sequence_body_ops(module: ModuleOp) -> list[Operation]:
    [seq] = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
    return list(seq.body.block.ops)


class TestProvenanceInfoAttr:
    """Isolation: ProvenanceInfoAttr construction and comment formatting."""

    def test_format_comment(self):
        info = ProvenanceInfoAttr("pulse.pulse", "q0_drive")
        assert info.format_comment() == "from pulse.pulse on q0_drive"

    def test_parameters_accessible(self):
        info = ProvenanceInfoAttr("pulse.wait", "q1_drive")
        assert info.source_op.data == "pulse.wait"
        assert info.port.data == "q1_drive"


class TestDebugInfoAutoAttach:
    """Verify debug info is auto-attached to Q1 ops during pulse-to-Q1 lowering."""

    def _assert_provenance(self, op, source_op: str, port: str) -> None:
        debug_info = op.properties.get("debug_info")
        assert isinstance(debug_info, ProvenanceInfoAttr), (
            f"{type(op).__name__} has no ProvenanceInfoAttr attached"
        )
        assert debug_info.source_op.data == source_op
        assert debug_info.port.data == port

    def test_wait_op(self):
        freq, frame = _frame()
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        module = _sequence_module(freq, frame, duration, wait)
        _run_q1_pipeline(module)
        [wait_op] = [op for op in _sequence_body_ops(module) if isinstance(op, WaitImmOp)]
        self._assert_provenance(wait_op, "pulse.wait", "q0_drive")

    def test_pulse_op(self):
        samples = np.ones(16, dtype=complex)
        waveform = ConstantOp(
            SampledWaveformAttr(samples, TimeAttr(16e-9), TimeAttr(1e-9)), WaveformType()
        )
        freq, frame = _frame()
        pulse = PulseOp(frame, waveform)
        module = _sequence_module(freq, frame, waveform, pulse)
        _run_q1_pipeline(module)
        [play_op] = [
            op for op in _sequence_body_ops(module) if isinstance(op, PlayImmImmImmOp)
        ]
        self._assert_provenance(play_op, "pulse.pulse", "q0_drive")

    def test_phase_set_op(self):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(0.5))
        phase_set = PhaseSetOp(frame, phase)
        module = _sequence_module(freq, frame, phase, phase_set)
        _run_q1_pipeline(module)
        [set_ph] = [op for op in _sequence_body_ops(module) if isinstance(op, SetPhImmOp)]
        self._assert_provenance(set_ph, "pulse.phase_set", "q0_drive")

    def test_phase_shift_op(self):
        freq, frame = _frame()
        phase = ConstantOp(PhaseAttr(0.5))
        phase_shift = PhaseShiftOp(frame, phase)
        module = _sequence_module(freq, frame, phase, phase_shift)
        _run_q1_pipeline(module)
        [set_ph_delta] = [
            op for op in _sequence_body_ops(module) if isinstance(op, SetPhDeltaImmOp)
        ]
        self._assert_provenance(set_ph_delta, "pulse.phase_shift", "q0_drive")

    def test_acquire_op(self):
        freq, frame = _frame("q0/measure")
        duration = ConstantOp(TimeAttr(1000, TimeUnits.NANOSECOND))
        acquire = AcquireOp(frame, duration, weights=None)
        module = _sequence_module(freq, frame, duration, acquire, channel_id="q0_measure")
        _run_q1_pipeline(module)
        [acq_op] = [
            op for op in _sequence_body_ops(module) if isinstance(op, AcquireImmImmImmOp)
        ]
        self._assert_provenance(acq_op, "pulse.acquire", "q0_measure")

    def test_start_continuous_waveform_op(self):
        amp = ConstantOp(AmplitudeAttr(0.5 + 0.25j), AmplitudeType())
        freq, frame = _frame()
        start = StartContinuousWaveformOp(frame, amp)
        module = _sequence_module(freq, frame, amp, start)
        _run_q1_pipeline(module)
        offs_ops = [
            op for op in _sequence_body_ops(module) if isinstance(op, SetAwgOffsImmImmOp)
        ]
        self._assert_provenance(offs_ops[0], "pulse.start_continuous_waveform", "q0_drive")

    def test_port_matches_sequence_channel(self):
        """The port in the attached debug info matches the enclosing SequenceOp channel."""
        freq, frame = _frame("q1/drive")
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        module = _sequence_module(freq, frame, duration, wait, channel_id="q1_drive")
        _run_q1_pipeline(module)
        [wait_op] = [op for op in _sequence_body_ops(module) if isinstance(op, WaitImmOp)]
        assert wait_op.properties.get("debug_info").port.data == "q1_drive"

    def test_long_wait_all_chunks_annotated(self):
        """Every WaitImmOp in a chunked long wait carries provenance from the source op."""
        max_wait_time = TARGET_DATA.Q1ASM_DATA.max_wait_time
        freq, frame = _frame()
        duration = ConstantOp(TimeAttr((2 * max_wait_time + 16) * 1e-9))
        wait = WaitOp(frame, duration)
        module = _sequence_module(freq, frame, duration, wait)
        _run_q1_pipeline(module)
        wait_ops = [op for op in _sequence_body_ops(module) if isinstance(op, WaitImmOp)]
        assert len(wait_ops) == 3
        for op in wait_ops:
            self._assert_provenance(op, "pulse.wait", "q0_drive")

    def test_debug_info_visible_in_emitted_assembly(self):
        """assembly_line with emit_debug_info=True shows provenance in the output."""
        freq, frame = _frame()
        duration = ConstantOp(TimeAttr(16e-9))
        wait = WaitOp(frame, duration)
        module = _sequence_module(freq, frame, duration, wait)
        _run_q1_pipeline(module)
        [wait_op] = [op for op in _sequence_body_ops(module) if isinstance(op, WaitImmOp)]
        line = wait_op.assembly_line(EmissionContext(emit_debug_info=True))
        assert "from pulse.wait on q0_drive" in line
