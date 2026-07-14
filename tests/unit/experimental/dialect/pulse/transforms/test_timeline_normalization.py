# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Implements tests for timeline normalization.

These tests exercise timeline normalization using the pass, followed by canonicalization to
ensure that the time values are properly resolved and simplified.

Some things that are tested are:

* Situations where time is completely deterministic gets resolved to constant times.
* Situations with a variable time (e.g. through a function argument or loop) gets resolved
  into expressions over times, but all synchronizes are rewritten to wait instructions where possible.
* Operations with unknown timing semantics result in affected frames being unknown, and
  synchronizes on those frames remain.
* Operations with bodies, such as control flows, means that we treat all frames as unknown,
  meaning that synchronizes on those frames remain.
* Synchronizing unknowns cause them to become known relative to each other, allowing us to
  resolve subsequent synchronizes between them to constants.
* Block argument frames are treated as unknowns.
"""

import numpy as np
import pytest
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i32
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.transforms.canonicalize import CanonicalizePass

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AddOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrameType,
    FrequencyAttr,
    MaxTimeOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    Pulse,
    PulseOp,
    SampledWaveformAttr,
    SquareWaveformOp,
    SubOp,
    SynchronizeOp,
    TimeAttr,
    TimeType,
    WaitOp,
    WaveformType,
)
from qat.experimental.dialect.pulse.transforms.timeline_normalization import (
    TimelineNormalization,
)
from qat.experimental.dialect.pulse.units import TimeUnits

from tests.unit.utils.ir import (
    build_module_from_ops,
    build_module_with_arguments,
    create_context,
)

_CONTEXT = create_context(Pulse)


def _make_block_with_frames(num_frames: int) -> tuple[list[Operation], list[SSAValue]]:
    """Helper function to initialize frame ops and return them with their frame SSA
    values."""

    ops = []
    frames = []
    for i in range(num_frames):
        freq_op = ConstantOp(value=FrequencyAttr(5.5e9))
        frame_op = CreateFrameOp(freq_op.result, StringAttr(f"port{i}"))
        ops.extend([freq_op, frame_op])
        frames.append(frame_op.result)
    return ops, frames


def _get_ops_of_types(
    module: ModuleOp, op_types: type[Operation] | list[type[Operation]]
) -> dict[type[Operation], list[Operation]]:
    """Helper function to get all operations of a given type or types from a module."""

    if isinstance(op_types, type):
        op_types = [op_types]

    ops = {op_type: [] for op_type in op_types}
    for op in module.walk():
        for op_type in op_types:
            if isinstance(op, op_type):
                ops[op_type].append(op)
    return ops


def _track_lineage_of_frame(frame: SSAValue) -> list[Operation]:
    """Helper function to track the lineage of a frame through the program, returning the
    list of operations that the frame goes through."""

    lineage = []
    current_frame = frame
    while current_frame.uses.get_length() > 0:
        assert current_frame.uses.get_length() == 1, (
            "Expected the frame to have only one use, otherwise the lineage is ambiguous."
        )
        use = next(iter(current_frame.uses))
        op = use.operation
        lineage.append(op)
        current_frame = op.results[op.operands.index(current_frame)]
    return lineage


class TestTimelineNormalizationWithKeyOperations:
    """Tests the standard operations from the pulse dialect that are expected to have well
    defined timing semantics.

    By testing these operations individually we can test the bigger picture behaviour in
    other tests without worrying about the details of how these operations are handled.
    """

    def test_with_no_sync_ops_does_not_change_program(self):
        """Tests that if there are no synchronize operations, then the program is not
        changed by the pass."""

        ops, _ = _make_block_with_frames(2)
        module_op = build_module_from_ops(ops)
        _, clone_module_op = TimelineNormalization().apply_to_clone(_CONTEXT, module_op)
        assert module_op.is_structurally_equivalent(clone_module_op)

    def test_with_pulse_op_increments_frame_time(self):
        """Tests that a frame operation with a pulse op increments the time of the frame by
        the duration of the pulse."""

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)
        sync_op = SynchronizeOp(frames[1], pulse_op.result)

        module_op = build_module_from_ops(
            ops + [time_op, amp_op, square_op, pulse_op, sync_op]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the pulse operation has been resolved to a pulse with a known duration
        ops = _get_ops_of_types(module_op, [PulseOp, SynchronizeOp, WaitOp])
        assert len(ops[PulseOp]) == 1
        assert ops[PulseOp][0].waveform.owner == square_op
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 1
        assert ops[WaitOp][0].duration.owner == time_op
        assert ops[WaitOp][0].frame == frames[1]

    def test_with_sampled_waveform_increments_frame_time(self):
        """Tests that a frame operation with a pulse op with a sampled waveform increments
        the time of the frame by the duration of the sampled waveform."""

        ops, frames = _make_block_with_frames(2)
        samples = np.linspace(0, 1, 64)
        sample_time = TimeAttr(1, TimeUnits.NANOSECOND)
        width = TimeAttr(64, TimeUnits.NANOSECOND)
        sampled_waveform_op = ConstantOp(
            SampledWaveformAttr(samples=samples, sample_time=sample_time, width=width)
        )
        pulse_op = PulseOp(frames[0], sampled_waveform_op.result)
        sync_op = SynchronizeOp(frames[1], pulse_op.result)

        module_op = build_module_from_ops(ops + [sampled_waveform_op, pulse_op, sync_op])
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the pulse operation has been resolved to a pulse with a known duration
        ops = _get_ops_of_types(module_op, [PulseOp, SynchronizeOp, WaitOp])
        assert len(ops[PulseOp]) == 1
        assert ops[PulseOp][0].waveform.owner == sampled_waveform_op
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 1
        constant_time_op = ops[WaitOp][0].duration.owner
        assert isinstance(constant_time_op, ConstantOp)
        assert constant_time_op.value.value.data == 64

    def test_with_acquire_op_increments_frame_time(self):
        """Tests that a frame operation with an acquire op increments the time of the frame
        by the duration of the acquire."""

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        acquire_op = AcquireOp(frames[0], time_op.result)
        sync_op = SynchronizeOp(frames[1], acquire_op.frame_result)

        module_op = build_module_from_ops(ops + [time_op, acquire_op, sync_op])
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the acquire operation has been resolved to an acquire with a known
        # duration
        ops = _get_ops_of_types(module_op, [AcquireOp, SynchronizeOp, WaitOp])
        assert len(ops[AcquireOp]) == 1
        assert ops[AcquireOp][0].duration.owner == time_op
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 1
        assert ops[WaitOp][0].duration.owner == time_op
        assert ops[WaitOp][0].frame == frames[1]

    def test_with_wait_op_increments_frame_time(self):
        """Tests that a frame operation with a wait op increments the time of the frame by
        the duration of the wait."""

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        wait_op = WaitOp(frames[0], time_op.result)
        sync_op = SynchronizeOp(frames[1], wait_op.result)

        module_op = build_module_from_ops(ops + [time_op, wait_op, sync_op])
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the wait operation has been resolved to a wait with a known duration
        ops = _get_ops_of_types(module_op, [WaitOp, SynchronizeOp])
        assert len(ops[WaitOp]) == 2
        assert ops[WaitOp][0].duration.owner == time_op
        assert ops[WaitOp][1].duration.owner == time_op
        assert len(ops[SynchronizeOp]) == 0
        assert {wait.frame for wait in ops[WaitOp]} == {frames[0], frames[1]}

    def test_with_phase_operations_does_not_advance_frame_time(self):
        """Tests that frame operations with phase operations do not advance the time of the
        frame, since phase operations are expected to be instantaneous."""

        ops, frames = _make_block_with_frames(2)
        phase_op = ConstantOp(PhaseAttr(0.25))
        phase_shift_op = PhaseShiftOp(frames[0], phase_op.result)
        phase_set_op = PhaseSetOp(frames[1], phase_op.result)
        sync_op = SynchronizeOp(phase_shift_op.result, phase_set_op.result)
        phase_set_op_2 = PhaseSetOp(sync_op.frames[0], phase_op.result)
        phase_shift_op_2 = PhaseShiftOp(sync_op.frames[1], phase_op.result)

        module_op = build_module_from_ops(
            [
                *ops,
                phase_op,
                phase_shift_op,
                phase_set_op,
                sync_op,
                phase_set_op_2,
                phase_shift_op_2,
            ]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the phase shift operation has been resolved to a phase shift with a
        # known phase
        ops = _get_ops_of_types(
            module_op, [PhaseShiftOp, PhaseSetOp, SynchronizeOp, WaitOp]
        )
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 0
        assert len(ops[PhaseShiftOp]) == 2
        assert len(ops[PhaseSetOp]) == 2
        assert ops[PhaseShiftOp] == [phase_shift_op, phase_shift_op_2]
        assert ops[PhaseSetOp] == [phase_set_op, phase_set_op_2]

    def test_with_outer_for_loop_resolves_as_expected(self):
        """Tests that if we have an outer for loop, then all synchronizes inside the loop
        resolve to waits.

        Essentially meaning that the loop is of no consequence, since it surrounds the
        quantum kernel.
        """

        # Make the inner loop
        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)
        sync_op = SynchronizeOp(frames[1], pulse_op.result)
        for_body = ops + [time_op, amp_op, square_op, pulse_op, sync_op]
        for_body = Block(for_body, arg_types=(i32,))

        # Create the loop
        lower_op = ArithConstantOp(IntegerAttr(0, i32))
        upper_op = ArithConstantOp(IntegerAttr(10, i32))
        step_op = ArithConstantOp(IntegerAttr(1, i32))
        for_op = ForOp(
            lb=lower_op.result,
            ub=upper_op.result,
            step=step_op.result,
            iter_args=[],
            body=for_body,
        )

        module_op = build_module_from_ops([for_op])
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check the body is still in the op
        module_ops = _get_ops_of_types(module_op, [ForOp, WaitOp, SynchronizeOp])
        assert len(module_ops[ForOp]) == 1
        for_block = module_ops[ForOp][0].regions[0].blocks[0]
        assert len(for_block.ops) == 9

        # Check there is a single wait op, and it lives in the for block
        assert len(module_ops[SynchronizeOp]) == 0
        assert len(module_ops[WaitOp]) == 1
        wait_op = module_ops[WaitOp][0]
        assert wait_op.parent == for_block
        assert wait_op.duration.owner == time_op


class TestTimelineNormalizationResolvesAllSynchronizesToConstants:
    """This tests a number of situations where we expect every synchronize to be resolved to
    a constant time.

    Many of the tests will inspect the IR after applying the pass, looking at the algebraic
    expressions created, and then also inspect after applying canonicalization to inspect
    the constants are as expected.

    Sets up and durations just using square waveforms to distinguish from lowered waits.
    Doesn't get too concerned with phase operations, as they're tested separately. This test
    just concerns whether timing is resolved to deterministic expressions.
    """

    @staticmethod
    def _apply_canonicalization_and_check_ops_and_times_are_as_expected(
        module_op: ModuleOp,
        frames: list[SSAValue],
        expected_ops: dict[SSAValue, list[type[Operation]]],
        expected_times: dict[SSAValue, list[int]],
    ):
        """Helper function to apply canonicalization to a module, and check that the
        expected operations and times are as expected."""

        CanonicalizePass().apply(_CONTEXT, module_op)
        for frame in frames:
            ops = expected_ops[frame]
            times = expected_times[frame]
            for i in range(len(ops)):
                assert frame.uses.get_length() == 1
                op = next(iter(frame.uses)).operation
                assert isinstance(op, ops[i])
                if ops[i] == WaitOp:
                    assert isinstance(op.duration.owner, ConstantOp)
                    assert op.duration.owner.value.value.data == times[i]
                elif ops[i] == PulseOp:
                    assert isinstance(op.waveform.owner, SquareWaveformOp)
                    assert isinstance(op.waveform.owner.width.owner, ConstantOp)
                    assert op.waveform.owner.width.owner.value.value.data == times[i]
                else:
                    raise ValueError(f"Unexpected operation type {ops[i]} in expected ops.")
                frame = op.result

    def test_two_frames_with_one_zero_time_resolves_into_correct_times(self):
        """Creates a program with two frames, where one evolves by no time, and the other
        evolves under multiple time operations, and applies synchronization.

        Tests that the synchronization resolves to a single wait on the second frame, with
        an operand that is an addition over the two time operations.
        """

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[1], square_op.result)
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        amp_op_2 = ConstantOp(AmplitudeAttr(1.0))
        square_op_2 = SquareWaveformOp(time_op_2.result, amp_op_2.result)
        pulse_op_2 = PulseOp(pulse_op.result, square_op_2.result)
        sync_op = SynchronizeOp(frames[0], pulse_op_2.result)

        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                pulse_op,
                time_op_2,
                amp_op_2,
                square_op_2,
                pulse_op_2,
                sync_op,
            ]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the synchronize has been resolved to a wait on the second frame, with
        # an operand that is an addition over the two time operations.
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp, AddOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 1
        assert len(ops[AddOp]) == 1

        # Checks that the lowered ops are as expected
        lowered_wait_op = ops[WaitOp][0]
        lowered_add_op = ops[AddOp][0]
        assert lowered_wait_op.frame == frames[0]
        assert lowered_wait_op.duration == lowered_add_op.results[0]
        assert set(lowered_add_op.operands) == {time_op.result, time_op_2.result}

        # Run canonicalization and check it simplifies nicely
        CanonicalizePass().apply(_CONTEXT, module_op)
        assert isinstance(lowered_wait_op.duration.owner, ConstantOp)
        assert lowered_wait_op.duration.owner.value.value.data == 192

    def test_two_frames_with_equal_time_removes_synchronize_operation_with_no_waits(self):
        """Creates a program with two frames, where both evolve under the same time
        operation, and applies synchronization.

        Since we know from resolved timing expressions that the two frames evolve under the
        same times, we expect the synchronization is redundant, and simply gets removed.
        """

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op_1 = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op_1 = PulseOp(frames[0], square_op_1.result)
        square_op_2 = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op_2 = PulseOp(frames[1], square_op_2.result)
        sync_op = SynchronizeOp(pulse_op_1.result, pulse_op_2.result)

        module_op = build_module_from_ops(
            ops
            + [time_op, amp_op, square_op_1, pulse_op_1, square_op_2, pulse_op_2, sync_op]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the synchronize has been removed because the frame times are equal.
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp, PulseOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 0
        assert set(ops[PulseOp]) == {pulse_op_1, pulse_op_2}

    def test_two_frames_with_equal_sum_of_times_resolves_into_zero_time_waits(self):
        """Creates a program with two frames, where both evolve under different time
        operations, but the sum of the time operations is the same for both frames, and
        applies synchronization.

        Since we know from resolved timing expressions that the two frames evolve under the
        same total time, we expect the synchronization to resolve to a wait with a constant
        zero time.
        """

        ops, frames = _make_block_with_frames(2)
        time_op_1 = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        time_op_3 = ConstantOp(TimeAttr(192, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op_1 = SquareWaveformOp(time_op_1.result, amp_op.result)
        pulse_op_1 = PulseOp(frames[0], square_op_1.result)
        square_op_2 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_2 = PulseOp(pulse_op_1.result, square_op_2.result)
        square_op_3 = SquareWaveformOp(time_op_3.result, amp_op.result)
        pulse_op_3 = PulseOp(frames[1], square_op_3.result)
        sync_op = SynchronizeOp(pulse_op_2.result, pulse_op_3.result)

        module_op = build_module_from_ops(
            ops
            + [
                time_op_1,
                time_op_2,
                time_op_3,
                amp_op,
                square_op_1,
                pulse_op_1,
                square_op_2,
                pulse_op_2,
                square_op_3,
                pulse_op_3,
                sync_op,
            ]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check that the synchronize has been resolved into two waits.
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 2

        # Check the wait ops are subtractions
        wait_op_4 = ops[WaitOp][0]
        wait_op_5 = ops[WaitOp][1]

        # Check the wait operations take on the same frames
        last_frames = {pulse_op_2.result, pulse_op_3.result}
        assert last_frames == {wait_op_4.frame, wait_op_5.frame}

        # Check the durations are the expected expressions
        sub_op_1 = wait_op_4.duration.owner
        sub_op_2 = wait_op_5.duration.owner
        assert isinstance(wait_op_4.duration.owner, SubOp)
        assert isinstance(wait_op_5.duration.owner, SubOp)
        assert isinstance(sub_op_1.lhs.owner, MaxTimeOp)
        assert isinstance(sub_op_2.lhs.owner, MaxTimeOp)
        assert sub_op_1.lhs.owner is sub_op_2.lhs.owner
        assert any(isinstance(sub_op.rhs.owner, AddOp) for sub_op in [sub_op_1, sub_op_2])
        assert any(sub_op.rhs.owner is time_op_3 for sub_op in [sub_op_1, sub_op_2])

        self._apply_canonicalization_and_check_ops_and_times_are_as_expected(
            module_op,
            frames,
            {frames[0]: [PulseOp, PulseOp], frames[1]: [PulseOp]},
            {frames[0]: [64, 128], frames[1]: [192]},
        )

    def test_multiple_synchronizations_on_two_frames_resolves_into_correct_times(self):
        """Creates a program with two frames, where both evolve under different time
        operations, and with many synchronizations between them.

        Makes sure that all synchronizations resolve into Waits, and that the time operands
        of the synchronizes are the expected deterministic expressions.
        """

        time_1 = 60
        time_2 = 132
        time_3 = 234

        ops, frames = _make_block_with_frames(2)
        time_op_1 = ConstantOp(TimeAttr(time_1, TimeUnits.NANOSECOND))
        time_op_2 = ConstantOp(TimeAttr(time_2, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op_1 = SquareWaveformOp(time_op_1.result, amp_op.result)
        pulse_op_1 = PulseOp(frames[0], square_op_1.result)
        square_op_2 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_2 = PulseOp(frames[1], square_op_2.result)
        sync_op_1 = SynchronizeOp(pulse_op_1.result, pulse_op_2.result)
        time_op_3 = ConstantOp(TimeAttr(time_3, TimeUnits.NANOSECOND))
        amp_op_3 = ConstantOp(AmplitudeAttr(1.0))
        square_op_3 = SquareWaveformOp(time_op_3.result, amp_op_3.result)
        pulse_op_3 = PulseOp(sync_op_1.result[0], square_op_3.result)

        sync_op_2 = SynchronizeOp(pulse_op_3.result, sync_op_1.result[1])

        module_op = build_module_from_ops(
            ops
            + [
                time_op_1,
                time_op_2,
                amp_op,
                square_op_1,
                pulse_op_1,
                square_op_2,
                pulse_op_2,
                sync_op_1,
                time_op_3,
                amp_op_3,
                square_op_3,
                pulse_op_3,
                sync_op_2,
            ]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 3

        lineage_0 = _track_lineage_of_frame(frames[0])
        assert len(lineage_0) == 3
        assert lineage_0[0] is pulse_op_1
        assert isinstance(lineage_0[1], WaitOp)
        duration = lineage_0[1].duration
        assert isinstance(duration.owner, SubOp)
        max_op_1 = duration.owner.lhs.owner
        assert isinstance(max_op_1, MaxTimeOp)
        assert len(max_op_1.times) == 2
        assert {duration.owner for duration in max_op_1.times} == {time_op_1, time_op_2}
        assert duration.owner.rhs.owner == time_op_1
        assert lineage_0[2] is pulse_op_3

        lineage_1 = _track_lineage_of_frame(frames[1])
        assert len(lineage_1) == 3
        assert lineage_1[0] is pulse_op_2
        assert isinstance(lineage_1[1], WaitOp)
        duration = lineage_1[1].duration
        assert isinstance(duration.owner, SubOp)
        assert duration.owner.lhs.owner == max_op_1
        assert duration.owner.rhs.owner == time_op_2
        assert isinstance(lineage_1[2], WaitOp)
        assert lineage_1[2].duration.owner is time_op_3

        CanonicalizePass().apply(_CONTEXT, module_op)
        self._apply_canonicalization_and_check_ops_and_times_are_as_expected(
            module_op,
            frames,
            {frames[0]: [PulseOp, WaitOp, PulseOp], frames[1]: [PulseOp, WaitOp]},
            {frames[0]: [time_1, time_2 - time_1, time_3], frames[1]: [time_2, time_3]},
        )

    def test_many_frames_simplifies_down_as_expected_after_canonicalization(self):
        """Creates a program with many frames, and different many different synchronization
        groups, and checks the results are entirely as expected."""

        times = [34, 78, 23, 46, 102, 4]

        ops, frames = _make_block_with_frames(5)
        time_ops = [ConstantOp(TimeAttr(time, TimeUnits.NANOSECOND)) for time in times]
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_ops = [
            SquareWaveformOp(time_op.result, amp_op.result) for time_op in time_ops
        ]

        pulse_on_frame_1_1 = PulseOp(frames[0], square_ops[0].result)
        pulse_on_frame_1_2 = PulseOp(pulse_on_frame_1_1.result, square_ops[1].result)
        pulse_on_frame_2_1 = PulseOp(frames[1], square_ops[2].result)
        synchronize_op_1_2 = SynchronizeOp(
            pulse_on_frame_1_2.result, pulse_on_frame_2_1.result
        )
        pulse_on_frame_3_1 = PulseOp(frames[2], square_ops[3].result)
        synchronize_op_2_3 = SynchronizeOp(
            synchronize_op_1_2.result[1], pulse_on_frame_3_1.result
        )
        synchronize_op_1_4 = SynchronizeOp(synchronize_op_1_2.result[0], frames[3])
        pulse_on_frame_4_1 = PulseOp(synchronize_op_1_4.result[1], square_ops[4].result)
        pulse_on_frame_1_3 = PulseOp(synchronize_op_1_4.result[0], square_ops[5].result)
        synchronize_all_op = SynchronizeOp(
            pulse_on_frame_1_3.result,
            synchronize_op_2_3.result[0],
            synchronize_op_2_3.result[1],
            pulse_on_frame_4_1.result,
            frames[4],
        )

        module_op = build_module_from_ops(
            ops
            + [
                *time_ops,
                amp_op,
                *square_ops,
                pulse_on_frame_1_1,
                pulse_on_frame_1_2,
                pulse_on_frame_2_1,
                synchronize_op_1_2,
                pulse_on_frame_3_1,
                synchronize_op_2_3,
                synchronize_op_1_4,
                pulse_on_frame_4_1,
                pulse_on_frame_1_3,
                synchronize_all_op,
            ]
        )
        TimelineNormalization().apply(_CONTEXT, module_op)

        expected_ops = {
            frames[0]: [PulseOp, PulseOp, PulseOp, WaitOp],
            frames[1]: [PulseOp, WaitOp, WaitOp],
            frames[2]: [PulseOp, WaitOp, WaitOp],
            frames[3]: [WaitOp, PulseOp],
            frames[4]: [WaitOp],
        }
        expected_times = {
            frames[0]: [times[0], times[1], times[5], times[4] - times[5]],
            frames[1]: [times[2], times[0] + times[1] - times[2], times[4]],
            frames[2]: [times[3], times[0] + times[1] - times[3], times[4]],
            frames[3]: [times[0] + times[1], times[4]],
            frames[4]: [times[0] + times[1] + times[4]],
        }
        assert len({sum(times) for times in expected_times.values()}) == 1
        self._apply_canonicalization_and_check_ops_and_times_are_as_expected(
            module_op, frames, expected_ops, expected_times
        )


class TestTimelineNormalizationWithNonConstantTimes:
    """Tests timeline normalization when one or more times are not constant.

    Tests this where times emerge as function arguments or from loop arguments, and mixes
    constant times within the program to ensure that we can still resolve as much as
    possible.
    """

    def test_function_with_single_time_argument_resolves_as_expected(self):
        """Tests a function with a single time argument.

        It uses two frames, doing the variable time on one frame, and nothing on the other.
        It then applies a synchronization, followed by a constant time on both frames, and a
        second synchronization. We expect the first synchronization to resolve to a wait
        with the variable time, and the second synchronization to resolve waits that only
        depend on the constants.
        """

        module, arguments, builder = build_module_with_arguments(TimeType())
        time_arg = arguments[0]
        ops, frames = _make_block_with_frames(2)

        amp_op = ConstantOp(AmplitudeAttr(1.0))
        waveform_op_1 = SquareWaveformOp(time_arg, amp_op.result)
        pulse_op = PulseOp(frames[0], waveform_op_1.result)
        sync_op_1 = SynchronizeOp(frames[1], pulse_op.result)
        time_op_1 = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        waveform_op_2 = SquareWaveformOp(time_op_1.result, amp_op.result)
        waveform_op_3 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_2 = PulseOp(sync_op_1.result[0], waveform_op_2.result)
        pulse_op_3 = PulseOp(sync_op_1.result[1], waveform_op_3.result)
        sync_op_2 = SynchronizeOp(pulse_op_2.result, pulse_op_3.result)

        ops += [
            amp_op,
            waveform_op_1,
            pulse_op,
            sync_op_1,
            time_op_1,
            time_op_2,
            waveform_op_2,
            waveform_op_3,
            pulse_op_2,
            pulse_op_3,
            sync_op_2,
        ]
        for op in ops:
            builder.insert(op)

        TimelineNormalization().apply(_CONTEXT, module)

        # Check synchronises are resolved
        ops = _get_ops_of_types(module, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 3

        # Check the lineage of the first frame
        lineage_0 = _track_lineage_of_frame(frames[0])
        assert len(lineage_0) == 3
        assert lineage_0[0] is pulse_op
        assert lineage_0[1] is pulse_op_3
        assert isinstance(lineage_0[2], WaitOp)
        duration = lineage_0[2].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs.owner == time_op_2
        assert isinstance(duration.lhs.owner, MaxTimeOp)
        max_op = duration.lhs.owner
        assert len(max_op.times) == 2
        assert {time_op_1, time_op_2} == {time.owner for time in max_op.times}

        # Check the lineage of the second frame
        lineage_1 = _track_lineage_of_frame(frames[1])
        assert len(lineage_1) == 3
        assert isinstance(lineage_1[0], WaitOp)
        assert lineage_1[0].duration == time_arg
        assert lineage_1[1] is pulse_op_2
        assert isinstance(lineage_1[2], WaitOp)
        duration = lineage_1[2].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs.owner == time_op_1
        assert duration.lhs.owner == max_op

    def test_function_with_two_time_arguments_resolves_as_expected(self):
        """Tests a function with two time arguments.

        Tests with a variable time on both frames, followed by a synchronization, followed
        by constant times on both frames, and a second synchronization. We expect the first
        synchronization to resolve to waits with a maximum over the two variable times, and
        the second synchronization to resolve waits that only depend on the constants.
        """

        module, arguments, builder = build_module_with_arguments(TimeType(), TimeType())
        time_arg_1, time_arg_2 = arguments
        ops, frames = _make_block_with_frames(2)

        amp_op = ConstantOp(AmplitudeAttr(1.0))
        waveform_op_1 = SquareWaveformOp(time_arg_1, amp_op.result)
        pulse_op_1 = PulseOp(frames[0], waveform_op_1.result)
        waveform_op_2 = SquareWaveformOp(time_arg_2, amp_op.result)
        pulse_op_2 = PulseOp(frames[1], waveform_op_2.result)
        sync_op_1 = SynchronizeOp(pulse_op_1.result, pulse_op_2.result)
        time_op_1 = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        waveform_op_3 = SquareWaveformOp(time_op_1.result, amp_op.result)
        waveform_op_4 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_3 = PulseOp(sync_op_1.result[0], waveform_op_3.result)
        pulse_op_4 = PulseOp(sync_op_1.result[1], waveform_op_4.result)
        sync_op_2 = SynchronizeOp(pulse_op_3.result, pulse_op_4.result)

        ops += [
            amp_op,
            waveform_op_1,
            pulse_op_1,
            waveform_op_2,
            pulse_op_2,
            sync_op_1,
            time_op_1,
            time_op_2,
            waveform_op_3,
            waveform_op_4,
            pulse_op_3,
            pulse_op_4,
            sync_op_2,
        ]
        for op in ops:
            builder.insert(op)

        TimelineNormalization().apply(_CONTEXT, module)

        # Check synchronises are resolved
        ops = _get_ops_of_types(module, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 4

        # Check the lineage of the first frame
        lineage_0 = _track_lineage_of_frame(frames[0])
        assert len(lineage_0) == 4
        assert lineage_0[0] is pulse_op_1
        assert isinstance(lineage_0[1], WaitOp)
        duration = lineage_0[1].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs == time_arg_1
        assert isinstance(duration.lhs.owner, MaxTimeOp)
        max_op1 = duration.lhs.owner
        assert len(max_op1.times) == 2
        assert {time_arg_1, time_arg_2} == set(max_op1.times)
        assert lineage_0[2] is pulse_op_3
        assert isinstance(lineage_0[3], WaitOp)
        duration = lineage_0[3].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs.owner == time_op_1
        assert isinstance(duration.lhs.owner, MaxTimeOp)
        max_op2 = duration.lhs.owner
        assert len(max_op2.times) == 2
        assert {time_op_1, time_op_2} == {time.owner for time in max_op2.times}

        # Check the lineage of the second frame
        lineage_1 = _track_lineage_of_frame(frames[1])
        assert len(lineage_1) == 4
        assert lineage_1[0] is pulse_op_2
        assert isinstance(lineage_1[1], WaitOp)
        duration = lineage_1[1].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs == time_arg_2
        assert duration.lhs.owner == max_op1
        assert lineage_1[2] is pulse_op_4
        assert isinstance(lineage_1[3], WaitOp)
        duration = lineage_1[3].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs.owner == time_op_2
        assert duration.lhs.owner == max_op2

    def test_loop_with_time_argument_resolves_as_expected(self):
        """Tests a loop with a time argument.

        Does the same one argument test as the function case, but with a loop instead
        providing the variable instead.
        """

        lower_bound = ArithConstantOp(IntegerAttr(0, i32))
        upper_bound = ArithConstantOp(IntegerAttr(10, i32))
        step = ArithConstantOp(IntegerAttr(1, i32))
        time_start = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        time_increment = ConstantOp(TimeAttr(8, TimeUnits.NANOSECOND))
        for_body = Block([], arg_types=(i32, TimeType()))

        time_arg = for_body.args[1]

        ops, frames = _make_block_with_frames(2)

        time_arg = AddOp(time_arg, time_increment.result, TimeType())
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        waveform_op_1 = SquareWaveformOp(time_arg, amp_op.result)
        pulse_op = PulseOp(frames[0], waveform_op_1.result)
        sync_op_1 = SynchronizeOp(frames[1], pulse_op.result)
        time_op_1 = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        waveform_op_2 = SquareWaveformOp(time_op_1.result, amp_op.result)
        waveform_op_3 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_2 = PulseOp(sync_op_1.result[0], waveform_op_2.result)
        pulse_op_3 = PulseOp(sync_op_1.result[1], waveform_op_3.result)
        sync_op_2 = SynchronizeOp(pulse_op_2.result, pulse_op_3.result)
        yield_op = YieldOp(time_arg)

        ops = ops + [
            time_arg,
            amp_op,
            waveform_op_1,
            pulse_op,
            sync_op_1,
            time_op_1,
            time_op_2,
            waveform_op_2,
            waveform_op_3,
            pulse_op_2,
            pulse_op_3,
            sync_op_2,
            yield_op,
        ]

        for op in ops:
            for_body.add_op(op)

        for_loop = ForOp(
            lb=lower_bound.result,
            ub=upper_bound.result,
            step=step.result,
            iter_args=[time_start.result],
            body=for_body,
        )

        module = build_module_from_ops(
            [lower_bound, upper_bound, step, time_start, time_increment, for_loop]
        )

        TimelineNormalization().apply(_CONTEXT, module)

        # Check synchronises are resolved
        ops = _get_ops_of_types(module, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 3

        # Check the lineage of the first frame
        lineage_0 = _track_lineage_of_frame(frames[0])
        assert len(lineage_0) == 3
        assert lineage_0[0] is pulse_op
        assert lineage_0[1] is pulse_op_3
        assert isinstance(lineage_0[2], WaitOp)
        duration = lineage_0[2].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs.owner == time_op_2
        assert isinstance(duration.lhs.owner, MaxTimeOp)
        max_op = duration.lhs.owner
        assert len(max_op.times) == 2
        assert {time_op_1.result, time_op_2.result} == set(max_op.times)

        # Check the lineage of the second frame
        lineage_1 = _track_lineage_of_frame(frames[1])
        assert len(lineage_1) == 3
        assert isinstance(lineage_1[0], WaitOp)
        assert lineage_1[0].duration == time_arg.result
        assert lineage_1[0].frame == frames[1]
        assert lineage_1[1] is pulse_op_2
        assert isinstance(lineage_1[2], WaitOp)
        duration = lineage_1[2].duration.owner
        assert isinstance(duration, SubOp)
        assert duration.rhs.owner == time_op_1
        assert duration.lhs.owner == max_op


class TestTimelineNormalizationWithUnknowns:
    """Tests with operations that have unknown timing semantics.

    Checks when timing is unknown, synchronization is left in tact, whether that be because
    of unknown frame consumers or nested operations.
    """

    @irdl_op_definition
    class _UnknownFrameConsumerOp(IRDLOperation):
        """An operation that consumes a frame, but has unknown timing semantics."""

        name = "unknown_frame_consumer"

        frames = var_operand_def(FrameType)
        result = var_result_def(FrameType)

        def __init__(self, *frames: SSAValue | Operation):
            """
            :param frames: A variable number of SSA values representing the frames to be
                synchronized.
            """
            frame_types = [SSAValue.get(frame, type=FrameType).type for frame in frames]
            return super().__init__(operands=[frames], result_types=[frame_types])

    @irdl_op_definition
    class _UnknownRegionOp(IRDLOperation):
        """An operation that has a region, but unknown timing semantics."""

        name = "unknown_region"

        body = region_def("single_block")

        def __init__(self):
            return super().__init__(regions=[Region()])

    def test_unknown_frame_consumer_leaves_time_as_unknown(self):
        """Tests that if we have an an operation that consumes frames with unspecified
        scheduling semantics, then the frames subsequent timings are treated as unknown, and
        synchronizations are not resolved.

        Adds two synchronizations to test the second one can be resolved, given the context
        of the first.
        """

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)
        unknown_op = self._UnknownFrameConsumerOp(pulse_op.result, frames[1])
        sync_op_1 = SynchronizeOp(*unknown_op.result)
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        square_op_2 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_2 = PulseOp(sync_op_1.result[0], square_op_2.result)
        sync_op_2 = SynchronizeOp(sync_op_1.result[1], pulse_op_2.result)
        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                pulse_op,
                unknown_op,
                sync_op_1,
                time_op_2,
                square_op_2,
                pulse_op_2,
                sync_op_2,
            ]
        )

        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check only one synchronize is resolved, and there's one wait
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 1
        assert len(ops[WaitOp]) == 1
        wait_op = ops[WaitOp][0]

        # Make sure lineages are as expected
        lineage_0 = _track_lineage_of_frame(frames[0])
        assert len(lineage_0) == 4
        assert lineage_0[0] is pulse_op
        assert lineage_0[1] is unknown_op
        assert lineage_0[2] is sync_op_1
        assert lineage_0[3] is pulse_op_2

        lineage = _track_lineage_of_frame(frames[1])
        assert len(lineage) == 3
        assert lineage[0] is unknown_op
        assert lineage[1] is sync_op_1
        assert lineage[2] is wait_op
        assert wait_op.duration.owner == time_op_2

    def test_unknown_frame_consumer_does_not_invalidate_other_frames(self):
        """Does the same test as the previous, but with a third and fourth frame that are
        then synchronized together, to check that the unknown frame consumer doesn't
        invalidate other frames."""

        ops, frames = _make_block_with_frames(4)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)
        pulse_op_2 = PulseOp(frames[2], square_op.result)
        unknown_op = self._UnknownFrameConsumerOp(pulse_op.result, frames[1])
        sync_op_1 = SynchronizeOp(pulse_op_2.result, frames[3])
        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                pulse_op,
                pulse_op_2,
                unknown_op,
                sync_op_1,
            ]
        )

        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check the synchronize is resolved, and there is one wait op
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 0
        assert len(ops[WaitOp]) == 1
        wait_op = ops[WaitOp][0]
        assert wait_op.duration.owner is time_op

    def test_unknown_frame_consumer_with_synchronize_between_affected_and_unaffected(self):
        """Tests an unknown frame consumer followed by a synchronize that synchronizes the
        affected frame with an unaffected frame, to check that the synchronize is left in
        tact."""

        ops, frames = _make_block_with_frames(3)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)
        unknown_op = self._UnknownFrameConsumerOp(pulse_op.result, frames[1])
        sync_op_1 = SynchronizeOp(unknown_op.result[0], frames[2])
        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                pulse_op,
                unknown_op,
                sync_op_1,
            ]
        )

        _, module_op_clone = TimelineNormalization().apply_to_clone(_CONTEXT, module_op)

        assert module_op.is_structurally_equivalent(module_op_clone)

    def test_unknown_region_leaves_synchronization_unresolved(self):
        """Tests that if we have an an operation with a region with unspecified scheduling
        semantics, then the frames within that region have unknown timings, and
        synchronizations are not resolved."""

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)
        unknown_region_op = self._UnknownRegionOp()
        sync_op_1 = SynchronizeOp(frames[1], pulse_op.result)
        time_op_2 = ConstantOp(TimeAttr(128, TimeUnits.NANOSECOND))
        square_op_2 = SquareWaveformOp(time_op_2.result, amp_op.result)
        pulse_op_2 = PulseOp(sync_op_1.result[0], square_op_2.result)
        sync_op_2 = SynchronizeOp(sync_op_1.result[1], pulse_op_2.result)
        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                pulse_op,
                unknown_region_op,
                sync_op_1,
                time_op_2,
                square_op_2,
                pulse_op_2,
                sync_op_2,
            ]
        )

        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check no synchronizes are resolved, and there are no waits
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 1
        assert len(ops[WaitOp]) == 1

        # Check the wait is on the correct frame, and has the correct duration
        wait_op = ops[WaitOp][0]
        assert wait_op.frame == sync_op_1.result[1]
        assert wait_op.duration.owner == time_op_2

    def test_ops_within_region_cant_be_immediately_resolved(self):
        """Creates IR with a ForLoop interweaved, and checks the first synchronization
        within can't be resolved."""

        ops, frames = _make_block_with_frames(2)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        pulse_op = PulseOp(frames[0], square_op.result)

        # Create a for loop  which acts on the two frames
        for_body = Block(arg_types=[i32, pulse_op.result.type, frames[1].type])
        lb = ArithConstantOp(IntegerAttr(0, i32))
        ub = ArithConstantOp(IntegerAttr(10, i32))
        step = ArithConstantOp(IntegerAttr(1, i32))
        for_loop = ForOp(
            lb=lb.result,
            ub=ub.result,
            step=step.result,
            iter_args=[pulse_op.result, frames[1]],
            body=for_body,
        )

        # Add pulse ops
        sync_op_1 = SynchronizeOp(for_body.args[1], for_body.args[2])
        for_body.add_op(sync_op_1)
        pulse_op_2 = PulseOp(sync_op_1.result[0], square_op.result)
        for_body.add_op(pulse_op_2)
        sync_op_2 = SynchronizeOp(sync_op_1.result[1], pulse_op_2.result)
        for_body.add_op(sync_op_2)
        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                pulse_op,
                for_loop,
            ]
        )

        TimelineNormalization().apply(_CONTEXT, module_op)

        # Find the synchronizations and waits
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 1
        assert len(ops[WaitOp]) == 1

        # Check the wait op has the correct properties
        wait_op = ops[WaitOp][0]
        assert wait_op.frame == sync_op_1.result[1]
        assert wait_op.duration.owner == time_op

    def test_unknown_pulse_timing_invalidates_frame(self):
        """Tests that if the source of a pulse isn't known, then the frame it acts on is
        treated as having unknown timing, and synchronizations aren't resolved.

        Tests by adding two waveforms together.
        """

        ops, frames = _make_block_with_frames(4)
        time_op = ConstantOp(TimeAttr(64, TimeUnits.NANOSECOND))
        amp_op = ConstantOp(AmplitudeAttr(1.0))
        square_op = SquareWaveformOp(time_op.result, amp_op.result)
        square_op_2 = SquareWaveformOp(time_op.result, amp_op.result)
        waveform_op = AddOp(square_op.result, square_op_2.result, WaveformType())
        pulse_op_1 = PulseOp(frames[0], waveform_op.result)
        pulse_op_2 = PulseOp(frames[3], square_op.result)

        sync_op_1 = SynchronizeOp(frames[1], pulse_op_1.result)
        sync_op_2 = SynchronizeOp(frames[2], pulse_op_2.result)
        module_op = build_module_from_ops(
            ops
            + [
                time_op,
                amp_op,
                square_op,
                square_op_2,
                waveform_op,
                pulse_op_1,
                pulse_op_2,
                sync_op_1,
                sync_op_2,
            ]
        )

        TimelineNormalization().apply(_CONTEXT, module_op)

        # Check only one synchronize is resolved, and on the right frames
        ops = _get_ops_of_types(module_op, [SynchronizeOp, WaitOp])
        assert len(ops[SynchronizeOp]) == 1
        assert ops[SynchronizeOp][0] is sync_op_1
        assert len(ops[WaitOp]) == 1
        wait_op = ops[WaitOp][0]
        assert wait_op.frame == frames[2]
        assert wait_op.duration.owner == time_op


class TestErrors:
    """Tests errors are raised for unsupported situations."""

    def test_multiple_blocks_in_region_raises_error(self):
        """Tests that if we have a region with multiple blocks, then we raise an error."""

        block_1 = Block([])
        block_2 = Block([])
        region = Region([block_1, block_2])
        module_op = ModuleOp(ops=region)

        with pytest.raises(NotImplementedError, match="multiple blocks"):
            TimelineNormalization().apply(_CONTEXT, module_op)
