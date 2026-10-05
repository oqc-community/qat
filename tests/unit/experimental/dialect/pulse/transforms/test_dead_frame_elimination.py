# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for dead frame elimination pass."""

from __future__ import annotations

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, StringAttr
from xdsl.dialects.scf import ForOp
from xdsl.ir import Block

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    Pulse,
    PulseOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.pulse.transforms.dead_frame_elimination import (
    DeadFrameEliminationPass,
)

from tests.unit.utils.ir import (
    build_module_from_ops,
    create_context,
    get_operations_with_type,
)

_CONTEXT = create_context(Pulse)


class TestDeadFrameEliminationPass:
    """Tests for DeadFrameEliminationPass.

    Tests verify frame elimination logic across multiple scenarios:

    - Empty and frame-less modules (edge cases)
    - Dead frames: lineages with only non-live operations (CreateFrameOp, PhaseSetOp,
      PhaseShiftOp, WaitOp)
    - Live frames: lineages containing at least one hardware-visible operation
      (AcquireOp, PulseOp, SynchronizeOp, etc.)
    - Complex scenarios: synchronized frames, mixed dead/live frames, multiple
      independent frames

    Strategy: Black-box verification of module state after pass application. Tests
    assert operation counts rather than rewriter call sequences, verifying that
    eliminated frames and their operations are removed from the module.
    """

    def test_empty_module_passes_without_error(self):
        """Verify pass handles empty modules gracefully."""
        module = build_module_from_ops([])
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)
        # No assertions needed; just verifies no exception is raised

    def test_module_with_no_frames_is_unchanged(self):
        """Verify pass leaves modules with no frames untouched."""
        freq = ConstantOp(FrequencyAttr(5.0e9))
        module = build_module_from_ops([freq])
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)
        assert len(get_operations_with_type(module, ConstantOp)) == 1

    def test_dead_frame_with_phase_set_shift_and_wait_is_eliminated(self):
        """Verify dead frames with only metadata operations are eliminated.

        Non-live operations (phase set, shift, wait) do not constitute hardware-visible
        work, so frames carrying only these operations should be completely removed.
        """
        # Create a frame and operations that don't use it for pulse-level work
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        phase_set = PhaseSetOp(frame, ConstantOp(PhaseAttr(0.0)))
        phase_shift = PhaseShiftOp(phase_set, ConstantOp(PhaseAttr(1.5)))
        wait = WaitOp(phase_shift, ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops([freq, frame, phase_set, phase_shift, wait])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify all frame operations were removed
        frames = get_operations_with_type(module, CreateFrameOp)
        phase_sets = get_operations_with_type(module, PhaseSetOp)
        phase_shifts = get_operations_with_type(module, PhaseShiftOp)
        waits = get_operations_with_type(module, WaitOp)

        assert len(frames) == 0, "Expected CreateFrameOp to be eliminated"
        assert len(phase_sets) == 0, "Expected PhaseSetOp to be eliminated"
        assert len(phase_shifts) == 0, "Expected PhaseShiftOp to be eliminated"
        assert len(waits) == 0, "Expected WaitOp to be eliminated"

    def test_live_frame_with_acquire_is_not_eliminated(self):
        """Verify frames with AcquireOp (hardware-visible) are preserved.

        Acquire is a live operation that performs measurement, so any frame in its lineage
        must be kept regardless of other non-live operations.
        """
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        phase_shift = PhaseShiftOp(frame, ConstantOp(PhaseAttr(1.5)))
        acquire = AcquireOp(phase_shift, ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops([freq, frame, phase_shift, acquire])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify frame is preserved because of acquire
        frames = get_operations_with_type(module, CreateFrameOp)
        acquires = get_operations_with_type(module, AcquireOp)

        assert len(frames) == 1, "Expected CreateFrameOp to be preserved"
        assert len(acquires) == 1, "Expected AcquireOp to be preserved"

    def test_live_frame_with_pulse_is_not_eliminated(self):
        """Verify frames with PulseOp (hardware-visible) are preserved.

        Pulse is a live operation that generates a waveform, so any frame in its lineage
        must be kept to maintain correct pulse timing and phase.
        """
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        duration = ConstantOp(TimeAttr(100e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(duration, amplitude)
        pulse = PulseOp(frame, waveform)

        module = build_module_from_ops([freq, frame, duration, amplitude, waveform, pulse])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify frame is preserved because of pulse
        frames = get_operations_with_type(module, CreateFrameOp)
        pulses = get_operations_with_type(module, PulseOp)

        assert len(frames) == 1, "Expected CreateFrameOp to be preserved"
        assert len(pulses) == 1, "Expected PulseOp to be preserved"

    def test_live_frame_with_synchronize_is_not_eliminated(self):
        """Verify frames in SynchronizeOp are preserved (sync is live).

        SynchronizeOp is live because timing delays on one synchronized frame can affect the
        timeline of other synchronized frames, making cross-frame timing dependent. Removing
        a frame's delays could alter synchronized timing.
        """
        freq_0 = ConstantOp(FrequencyAttr(5.0e9))
        freq_1 = ConstantOp(FrequencyAttr(6.0e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))

        # Synchronize both frames
        sync = SynchronizeOp(frame_0, frame_1)
        # Use synchronized frames with acquire (live)
        acquire_0 = AcquireOp(sync.results[0], ConstantOp(TimeAttr(100e-9)))
        acquire_1 = AcquireOp(sync.results[1], ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops(
            [freq_0, freq_1, frame_0, frame_1, sync, acquire_0, acquire_1]
        )

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify both frames are preserved
        frames = get_operations_with_type(module, CreateFrameOp)
        syncs = get_operations_with_type(module, SynchronizeOp)

        assert len(frames) == 2, "Expected both CreateFrameOps to be preserved"
        assert len(syncs) == 1, "Expected SynchronizeOp to be preserved"

    def test_live_frame_with_synchronize_without_result_use_is_preserved(self):
        """Verify frames are preserved by SynchronizeOp even without consuming results.

        SynchronizeOp itself is live regardless of whether its output is consumed.
        Synchronization coordinates frame timing across multiple frames, making it hardware-
        visible work even without downstream operations on the synchronized results.
        """
        freq_0 = ConstantOp(FrequencyAttr(5.0e9))
        freq_1 = ConstantOp(FrequencyAttr(6.0e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))

        # Synchronize both frames (synchronize itself is a live operation)
        sync = SynchronizeOp(frame_0, frame_1)

        module = build_module_from_ops([freq_0, freq_1, frame_0, frame_1, sync])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify both frames are preserved because SynchronizeOp is live
        frames = get_operations_with_type(module, CreateFrameOp)
        syncs = get_operations_with_type(module, SynchronizeOp)

        assert len(frames) == 2, "Expected CreateFrameOps to be preserved (sync is live)"
        assert len(syncs) == 1, "Expected SynchronizeOp to be preserved"

    def test_wait_only_frame_is_eliminated(self):
        """Verify frames with only non-live operations are eliminated."""
        # Verify that a frame with only Wait operations (non-live) is correctly
        # eliminated, not preserved.
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        # Just wait on the frame - no live ops
        wait = WaitOp(frame, ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops([freq, frame, wait])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify frame is eliminated (wait is not live)
        frames = get_operations_with_type(module, CreateFrameOp)
        waits = get_operations_with_type(module, WaitOp)
        assert len(frames) == 0, "Expected CreateFrameOp to be eliminated (wait only)"
        assert len(waits) == 0, "Expected WaitOp to be eliminated"

    def test_multiple_dead_frames_all_eliminated(self):
        """Verify multiple independent dead frames are all eliminated."""
        freq_0 = ConstantOp(FrequencyAttr(5.0e9))
        freq_1 = ConstantOp(FrequencyAttr(6.0e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))

        # Dead operations on each frame
        wait_0 = WaitOp(frame_0, ConstantOp(TimeAttr(100e-9)))
        wait_1 = WaitOp(frame_1, ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops([freq_0, freq_1, frame_0, frame_1, wait_0, wait_1])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify all frames are eliminated
        frames = get_operations_with_type(module, CreateFrameOp)
        waits = get_operations_with_type(module, WaitOp)

        assert len(frames) == 0, "Expected all CreateFrameOps to be eliminated"
        assert len(waits) == 0, "Expected all WaitOps to be eliminated"

    def test_mixed_dead_and_live_frames_preserves_live_eliminates_dead(self):
        """Verify independent dead and live frames are handled separately.

        Frames with only non-live operations should be eliminated even when other live
        frames exist in the same module. Frame preservation is per-lineage, not module-wide.
        """
        # Dead frame
        freq_dead = ConstantOp(FrequencyAttr(5.0e9))
        frame_dead = CreateFrameOp(freq_dead, StringAttr("q0/drive"))
        wait_dead = WaitOp(frame_dead, ConstantOp(TimeAttr(100e-9)))

        # Live frame
        freq_live = ConstantOp(FrequencyAttr(6.0e9))
        frame_live = CreateFrameOp(freq_live, StringAttr("q1/drive"))
        acquire_live = AcquireOp(frame_live, ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops(
            [freq_dead, frame_dead, wait_dead, freq_live, frame_live, acquire_live]
        )

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify only dead frame is eliminated
        frames = get_operations_with_type(module, CreateFrameOp)
        waits = get_operations_with_type(module, WaitOp)
        acquires = get_operations_with_type(module, AcquireOp)

        assert len(frames) == 1, "Expected only live frame to be preserved"
        assert len(waits) == 0, "Expected dead wait operation to be eliminated"
        assert len(acquires) == 1, "Expected live acquire operation to be preserved"

    def test_live_frame_synced_with_pulse_frame_both_preserved(self):
        """Verify frames synchronized together are preserved if any has a live op."""
        # Frame 1: has a pulse (live)
        freq_0 = ConstantOp(FrequencyAttr(5.0e9))
        frame_0 = CreateFrameOp(freq_0, StringAttr("q0/drive"))
        duration_0 = ConstantOp(TimeAttr(100e-9))
        amplitude_0 = ConstantOp(AmplitudeAttr(1.0))
        waveform_0 = SquareWaveformOp(duration_0, amplitude_0)
        pulse_0 = PulseOp(frame_0, waveform_0)

        # Frame 2: only phase operations (dead on its own)
        freq_1 = ConstantOp(FrequencyAttr(6.0e9))
        frame_1 = CreateFrameOp(freq_1, StringAttr("q1/drive"))
        phase_shift_1 = PhaseShiftOp(frame_1, ConstantOp(PhaseAttr(1.5)))

        # Synchronize both frames
        sync = SynchronizeOp(pulse_0.result, phase_shift_1.result)
        acquire_0 = AcquireOp(sync.results[0], ConstantOp(TimeAttr(100e-9)))
        acquire_1 = AcquireOp(sync.results[1], ConstantOp(TimeAttr(100e-9)))

        module = build_module_from_ops(
            [
                freq_0,
                frame_0,
                duration_0,
                amplitude_0,
                waveform_0,
                pulse_0,
                freq_1,
                frame_1,
                phase_shift_1,
                sync,
                acquire_0,
                acquire_1,
            ]
        )

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify both frames are preserved (frame_1 is kept because of sync with frame_0)
        frames = get_operations_with_type(module, CreateFrameOp)
        syncs = get_operations_with_type(module, SynchronizeOp)
        phase_shifts = get_operations_with_type(module, PhaseShiftOp)

        assert len(frames) == 2, "Expected both CreateFrameOps to be preserved"
        assert len(syncs) == 1, "Expected SynchronizeOp to be preserved"
        # Phase shift on frame_1 should be preserved because frame_1 is kept
        assert len(phase_shifts) == 1, "Expected PhaseShiftOp to be preserved"

    def test_unknown_operations_are_warned_once_when_lineage_is_dead(self, caplog):
        """Verify unknown ops in a dead lineage are collected and reported once.

        Unknown operations should preserve the frame, but only one warning should be logged
        summarising all unknown operations seen in the lineage.
        """
        from xdsl.ir import Operation, SSAValue

        class UnknownOpA(Operation):
            name = "unknown.a"

            def __init__(self, frame_operand: SSAValue):
                super().__init__(operands=[frame_operand])

        class UnknownOpB(Operation):
            name = "unknown.b"

            def __init__(self, frame_operand: SSAValue):
                super().__init__(operands=[frame_operand])

        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        unknown_a = UnknownOpA(frame.result)
        unknown_b = UnknownOpB(frame.result)

        module = build_module_from_ops([freq, frame, unknown_a, unknown_b])

        with caplog.at_level("WARNING"):
            pass_instance = DeadFrameEliminationPass()
            pass_instance.apply(_CONTEXT, module)

        frames = get_operations_with_type(module, CreateFrameOp)
        purr_records = [record for record in caplog.records if record.name == "purr"]
        assert len(frames) == 1, "Expected unknown operations to preserve the frame"
        assert len(purr_records) == 1, "Expected a single warning for the lineage"
        assert "unknown.a" in purr_records[0].message
        assert "unknown.b" in purr_records[0].message

    def test_known_live_operation_suppresses_unknown_warning(self, caplog):
        """Verify a live operation suppresses the unknown-op warning entirely."""
        from xdsl.ir import Operation, SSAValue

        class UnknownOp(Operation):
            name = "unknown.live_followed_by_unknown"

            def __init__(self, frame_operand: SSAValue):
                super().__init__(operands=[frame_operand])

        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        duration = ConstantOp(TimeAttr(100e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(duration, amplitude)
        pulse = PulseOp(frame, waveform)
        unknown = UnknownOp(pulse.result)

        module = build_module_from_ops(
            [freq, frame, duration, amplitude, waveform, pulse, unknown]
        )

        with caplog.at_level("WARNING"):
            pass_instance = DeadFrameEliminationPass()
            pass_instance.apply(_CONTEXT, module)

        frames = get_operations_with_type(module, CreateFrameOp)
        purr_records = [record for record in caplog.records if record.name == "purr"]
        assert len(frames) == 1, "Expected live pulse to keep the frame alive"
        assert not purr_records, "Expected no warning when a known live op is present"

    def test_frame_with_only_create_frame_op_is_dead(self):
        """Verify frames with no downstream operations are eliminated.

        A frame creation with no operations on it is dead code that should be eliminated
        entirely.
        """
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))

        module = build_module_from_ops([freq, frame])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify frame is eliminated
        frames = get_operations_with_type(module, CreateFrameOp)

        assert len(frames) == 0, (
            "Expected CreateFrameOp with no operations to be eliminated"
        )

    def test_frame_with_start_continuous_waveform_is_preserved(self):
        """Verify frames with StartContinuousWaveformOp (live) are preserved.

        StartContinuousWaveformOp is a live operation that initiates continuous output, so
        frames in its lineage must be kept.
        """
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        start_cw = StartContinuousWaveformOp(frame, amplitude)

        module = build_module_from_ops([freq, frame, amplitude, start_cw])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify frame is preserved because StartContinuousWaveformOp is live
        frames = get_operations_with_type(module, CreateFrameOp)
        start_cws = get_operations_with_type(module, StartContinuousWaveformOp)

        assert len(frames) == 1, "Expected CreateFrameOp to be preserved"
        assert len(start_cws) == 1, "Expected StartContinuousWaveformOp to be preserved"

    def test_frame_with_stop_continuous_waveform_is_preserved(self):
        """Verify frames with StopContinuousWaveformOp (live) are preserved.

        StopContinuousWaveformOp is a live operation that halts continuous output, so frames
        in its lineage must be kept.
        """
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))
        stop_cw = StopContinuousWaveformOp(frame)

        module = build_module_from_ops([freq, frame, stop_cw])

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify frame is preserved because StopContinuousWaveformOp is live
        frames = get_operations_with_type(module, CreateFrameOp)
        stop_cws = get_operations_with_type(module, StopContinuousWaveformOp)

        assert len(frames) == 1, "Expected CreateFrameOp to be preserved"
        assert len(stop_cws) == 1, "Expected StopContinuousWaveformOp to be preserved"

    def test_frame_passed_to_unknown_operation_preserves_frame(self, caplog):
        """Verify frames passed to unknown operations are preserved (fail-safe).

        Operations outside the recognized live/non-live types are conservatively treated as
        live to prevent incorrect elimination of frames passed to external dialects or
        operations the pass doesn't understand.
        """
        from xdsl.ir import Operation, SSAValue

        # Create a custom unknown operation that takes a frame as an operand
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(freq, StringAttr("q0/drive"))

        # Create a mock unknown operation that references the frame
        # We'll create a dummy operation class that isn't in the recognized lists
        class UnknownExternalOp(Operation):
            name = "unknown.external_op"

            def __init__(self, frame_operand: SSAValue):
                super().__init__(operands=[frame_operand])

        unknown_op = UnknownExternalOp(frame.result)

        module = build_module_from_ops([freq, frame, unknown_op])

        with caplog.at_level("WARNING"):
            pass_instance = DeadFrameEliminationPass()
            pass_instance.apply(_CONTEXT, module)

        # Verify frame is preserved because it's passed to an unknown operation
        # (conservative fail-safe behavior)
        frames = get_operations_with_type(module, CreateFrameOp)
        purr_records = [record for record in caplog.records if record.name == "purr"]

        assert len(frames) == 1, (
            "Expected CreateFrameOp to be preserved (unknown op triggers fail-safe)"
        )
        assert len(purr_records) == 1, "Expected a single warning for the lineage"
        assert "unknown.external_op" in purr_records[0].message

    def test_frame_with_scf_for_in_entry_block_is_preserved_or_eliminated_correctly(
        self,
    ):
        """Verify dead frame elimination works correctly with SCF for loops present.

        When SCF control flow is present in the same module, dead frame elimination should
        still correctly identify and eliminate dead frames that are not used by any live
        operations. This tests that presence of region-bearing ops (like scf.for) doesn't
        interfere with the analysis.
        """
        # Create two frames: one dead, one... well, also dead in this test
        freq = ConstantOp(FrequencyAttr(5.0e9))
        frame1 = CreateFrameOp(freq, StringAttr("q0/drive"))
        frame2 = CreateFrameOp(freq, StringAttr("q1/drive"))

        # Add only a phase shift to frame1 (dead)
        phase_shift = PhaseShiftOp(frame1.result, ConstantOp(PhaseAttr(1.57)))

        # Create an scf.for loop that is NOT used as an iter_arg
        # (just to show it can coexist with frame elimination)
        c0 = arith.ConstantOp.from_int_and_width(0, IndexType())
        c10 = arith.ConstantOp.from_int_and_width(10, IndexType())
        c1 = arith.ConstantOp.from_int_and_width(1, IndexType())
        body_block = Block(arg_types=[IndexType()])
        for_loop = ForOp(c0, c10, c1, [], body_block)

        module = build_module_from_ops(
            [freq, frame1, phase_shift, frame2, c0, c10, c1, for_loop]
        )

        # Apply elimination
        pass_instance = DeadFrameEliminationPass()
        pass_instance.apply(_CONTEXT, module)

        # Verify both frames are eliminated (they're both dead)
        frames = get_operations_with_type(module, CreateFrameOp)
        assert len(frames) == 0, "Expected all CreateFrameOps to be eliminated"

        # Verify scf.for is still present
        for_loops = get_operations_with_type(module, ForOp)
        assert len(for_loops) == 1, "Expected scf.for to be preserved"
