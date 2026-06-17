# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import random

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import FloatAttr, ModuleOp, StringAttr, f64
from xdsl.ir import Block

from qat.experimental.dialect.pulse.ir.attributes import (
    AmplitudeAttr,
    FrequencyAttr,
    PhaseAttr,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.ops import (
    AcquireOp,
    AddOp,
    ConstantOp,
    CreateFrameOp,
    GaussianWaveformOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.transforms.optimize_contiguous_squashable_instructions import (
    ApplySquashContiguousOptimizations,
)


def _build_waveform_ops():
    amplitude = ConstantOp(AmplitudeAttr(1.0))
    width = ConstantOp(TimeAttr(800e-9))
    rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)
    waveform = GaussianWaveformOp(width, amplitude, rise)
    return [amplitude, width, rise, waveform], waveform


def _build_acquire_ops():
    duration = ConstantOp(TimeAttr(1.0e-6))
    return duration


def _phase_shift_ops(module: ModuleOp) -> list[PhaseShiftOp]:
    return [node for node in module.walk() if isinstance(node, PhaseShiftOp)]


def _wait_ops(module: ModuleOp) -> list[WaitOp]:
    return [node for node in module.walk() if isinstance(node, WaitOp)]


def _add_ops(module: ModuleOp) -> list[AddOp]:
    return [node for node in module.walk() if isinstance(node, AddOp)]


def _phase_operands(module: ModuleOp):
    return [op.phase for op in _phase_shift_ops(module)]


class TestFoldContiguousPhaseShifts:
    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_fold_contiguous_phase_shifts(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(frame.results[0], phase1.results[0])
        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseShiftOp(phase_op1.results[0], phase2.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=phase_op2.results[0], waveform=waveform.results[0])
            module_ops = [frame, phase1, phase_op1, phase2, phase_op2, *wf_ops, terminal_op]
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(
                frame=phase_op2.results[0], duration=duration.results[0]
            )
            module_ops = [
                frame,
                phase1,
                phase_op1,
                phase2,
                phase_op2,
                duration,
                terminal_op,
            ]

        module = ModuleOp(module_ops)
        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 2
        assert _phase_operands(module) == [phase1.results[0], phase2.results[0]]

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 1
        node = phase_ops[0].phase.owner
        assert isinstance(node, AddOp)

        lhs, rhs = node.operands[0].owner, node.operands[1].owner
        assert isinstance(lhs, ConstantOp) and isinstance(rhs, ConstantOp)
        assert lhs.value == PhaseAttr(0.1) or lhs.value == PhaseAttr(0.2)
        assert rhs.value == PhaseAttr(0.1) or rhs.value == PhaseAttr(0.2)

        assert isinstance(phase_ops[0].frame.owner, CreateFrameOp)

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_fold_contiguous_phase_shifts_with_multiple_intermediate_pulse_or_acquires(
        self, terminal_op_type
    ):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(frame.results[0], phase1.results[0])
        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseShiftOp(phase_op1.results[0], phase2.results[0])

        # Intermediate terminal is always PulseOp — only pulse continues the frame chain.
        interm1_extra_ops, interm1_waveform = _build_waveform_ops()
        intermediate_terminal_op1 = PulseOp(
            frame=phase_op2.results[0],
            waveform=interm1_waveform.results[0],
        )

        phase3 = ConstantOp(PhaseAttr(0.1))
        phase_op3 = PhaseShiftOp(intermediate_terminal_op1.results[0], phase3.results[0])
        phase4 = ConstantOp(PhaseAttr(0.2))
        phase_op4 = PhaseShiftOp(phase_op3.results[0], phase4.results[0])

        interm2_extra_ops, interm2_waveform = _build_waveform_ops()
        intermediate_terminal_op2 = PulseOp(
            frame=phase_op4.results[0],
            waveform=interm2_waveform.results[0],
        )

        phase5 = ConstantOp(PhaseAttr(0.1))
        phase_op5 = PhaseShiftOp(intermediate_terminal_op2.results[0], phase5.results[0])
        phase6 = ConstantOp(PhaseAttr(0.2))
        phase_op6 = PhaseShiftOp(phase_op5.results[0], phase6.results[0])

        # Final terminal parametrised — pulse or acquire.
        if terminal_op_type == "pulse":
            final_extra_ops, final_waveform = _build_waveform_ops()
            final_terminal_op = PulseOp(
                frame=phase_op6.results[0],
                waveform=final_waveform.results[0],
            )
        else:
            final_duration = _build_acquire_ops()
            final_extra_ops = [final_duration]
            final_terminal_op = AcquireOp(
                frame=phase_op6.results[0],
                duration=final_duration.results[0],
            )

        module_ops = [
            frame,
            phase1,
            phase_op1,
            phase2,
            phase_op2,
            *interm1_extra_ops,
            intermediate_terminal_op1,
            phase3,
            phase_op3,
            phase4,
            phase_op4,
            *interm2_extra_ops,
            intermediate_terminal_op2,
            phase5,
            phase_op5,
            phase6,
            phase_op6,
            *final_extra_ops,
            final_terminal_op,
        ]
        module = ModuleOp(module_ops)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 6

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 3

        assert all(isinstance(op.phase.owner, AddOp) for op in phase_ops)
        add_ops = _add_ops(module)
        assert len(add_ops) == 3

        frame_owner_types = [type(op.frame.owner) for op in phase_ops]
        assert frame_owner_types.count(CreateFrameOp) == 1
        assert frame_owner_types.count(PulseOp) == 2

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_does_not_fold_with_no_contiguous_phase_shifts(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase = ConstantOp(PhaseAttr(0.1))
        phase_op = PhaseShiftOp(frame.results[0], phase.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=phase_op.results[0], waveform=waveform.results[0])
            module_ops = [frame, phase, phase_op, *wf_ops, terminal_op]
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=phase_op.results[0], duration=duration.results[0])
            module_ops = [frame, phase, phase_op, duration, terminal_op]
        module = ModuleOp(module_ops)
        pass_instance = ApplySquashContiguousOptimizations()
        _, new_module = pass_instance.apply_to_clone(Context(), module)
        assert module.is_structurally_equivalent(new_module)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 1

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 1

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_does_not_fold_with_no_phase_shifts(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=frame.results[0], waveform=waveform.results[0])
            module_ops = [frame, *wf_ops, terminal_op]
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=frame.results[0], duration=duration.results[0])
            module_ops = [frame, duration, terminal_op]
        module = ModuleOp(module_ops)
        pass_instance = ApplySquashContiguousOptimizations()
        _, new_module = pass_instance.apply_to_clone(Context(), module)
        assert module.is_structurally_equivalent(new_module)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 0

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops = _phase_shift_ops(module)
        assert len(phase_ops) == 0
        assert _add_ops(module) == []

        # Backtrack through IR manually to confirm nothing changed.
        if terminal_op_type == "pulse":
            terminal_ops = [n for n in module.walk() if isinstance(n, PulseOp)]
            assert len(terminal_ops) == 1
            # PulseOp.frame should still point directly at CreateFrameOp.
            assert isinstance(terminal_ops[0].frame.owner, CreateFrameOp)
        else:
            terminal_ops = [n for n in module.walk() if isinstance(n, AcquireOp)]
            assert len(terminal_ops) == 1
            # AcquireOp.frame should still point directly at CreateFrameOp.
            assert isinstance(terminal_ops[0].frame.owner, CreateFrameOp)
            # AcquireOp.duration should still point directly at the original ConstantOp.
            assert isinstance(terminal_ops[0].duration.owner, ConstantOp)

    def test_does_not_fold_when_phase_shifts_are_separated_by_pulse(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(frame.results[0], phase1.results[0])

        wf_ops, waveform = _build_waveform_ops()
        pulse_mid = PulseOp(frame=phase_op1.results[0], waveform=waveform.results[0])

        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseShiftOp(pulse_mid.results[0], phase2.results[0])
        pulse_out = PulseOp(frame=phase_op2.results[0], waveform=waveform.results[0])

        module = ModuleOp(
            [frame, phase1, phase_op1, *wf_ops, pulse_mid, phase2, phase_op2, pulse_out]
        )
        pass_instance = ApplySquashContiguousOptimizations()
        _, new_module = pass_instance.apply_to_clone(Context(), module)
        assert module.is_structurally_equivalent(new_module)

        phase_ops_before = [
            node for node in module.walk() if isinstance(node, PhaseShiftOp)
        ]
        phase_operands_before = [phase_op.phase for phase_op in phase_ops_before]
        assert len(phase_ops_before) == 2

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops_after = _phase_shift_ops(module)
        phase_operands_after = _phase_operands(module)

        assert len(phase_ops_after) == 2
        assert phase_operands_after == phase_operands_before
        assert _add_ops(module) == []

    def test_does_not_fold_when_phase_shifts_are_separated_by_acquire(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(frame.results[0], phase1.results[0])

        duration = _build_acquire_ops()
        acquire_mid = AcquireOp(frame=phase_op1.results[0], duration=duration.results[0])

        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseShiftOp(acquire_mid.results[0], phase2.results[0])

        wf_ops, waveform = _build_waveform_ops()

        pulse_out = PulseOp(frame=phase_op2.results[0], waveform=waveform.results[0])

        module = ModuleOp(
            [
                frame,
                phase1,
                phase_op1,
                duration,
                acquire_mid,
                phase2,
                phase_op2,
                *wf_ops,
                pulse_out,
            ]
        )
        pass_instance = ApplySquashContiguousOptimizations()
        _, new_module = pass_instance.apply_to_clone(Context(), module)
        assert module.is_structurally_equivalent(new_module)

        phase_ops_before = [
            node for node in module.walk() if isinstance(node, PhaseShiftOp)
        ]
        phase_operands_before = [phase_op.phase for phase_op in phase_ops_before]
        assert len(phase_ops_before) == 2

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops_after = _phase_shift_ops(module)
        phase_operands_after = _phase_operands(module)

        assert len(phase_ops_after) == 2
        assert phase_operands_after == phase_operands_before
        assert _add_ops(module) == []

    def test_folds_large_contiguous_chain_and_creates_expected_addops(self):
        n_phases = 100

        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase = ConstantOp(PhaseAttr(2.0 * np.pi / n_phases))

        current_frame = frame.results[0]
        phase_ops = []
        for _ in range(n_phases):
            shift = PhaseShiftOp(current_frame, phase.results[0])
            phase_ops.append(shift)
            current_frame = shift.results[0]

        wf_ops, waveform = _build_waveform_ops()
        pulse = PulseOp(frame=current_frame, waveform=waveform.results[0])

        module = ModuleOp([frame, phase, *phase_ops, *wf_ops, pulse])

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        merged_phase_ops = _phase_shift_ops(module)
        add_ops = _add_ops(module)

        assert len(merged_phase_ops) == 1
        assert len(add_ops) == n_phases - 1

        # Walk the AddOp chain backwards from the final phase expression.
        # Each AddOp should have one ConstantOp operand and one AddOp operand,
        # except the deepest which has two ConstantOp operands.

        node = merged_phase_ops[0].phase.owner
        assert isinstance(node, AddOp)

        for i in range(n_phases - 1):
            assert isinstance(node, AddOp)
            lhs, rhs = node.operands[0].owner, node.operands[1].owner
            if i < n_phases - 2:
                # Interior nodes: one side is ConstantOp, other is next AddOp.
                assert (isinstance(lhs, ConstantOp) and isinstance(rhs, AddOp)) or (
                    isinstance(rhs, ConstantOp) and isinstance(lhs, AddOp)
                )
                node = rhs if isinstance(rhs, AddOp) else lhs
            else:
                # Leaf node: both operands are ConstantOp.
                assert isinstance(lhs, ConstantOp)
                assert isinstance(rhs, ConstantOp)

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_folds_phase_shifts_interleaved_with_waits(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(frame.results[0], phase1.results[0])

        wait_duration = ConstantOp(TimeAttr(1.0e-9))
        wait_op = WaitOp(phase_op1.results[0], wait_duration.results[0])

        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseShiftOp(wait_op.results[0], phase2.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=phase_op2.results[0], waveform=waveform.results[0])
            module_ops = [
                frame,
                phase1,
                phase_op1,
                wait_duration,
                wait_op,
                phase2,
                phase_op2,
                *wf_ops,
                terminal_op,
            ]
            terminal_op_cls = PulseOp
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(
                frame=phase_op2.results[0], duration=duration.results[0]
            )
            module_ops = [
                frame,
                phase1,
                phase_op1,
                wait_duration,
                wait_op,
                phase2,
                phase_op2,
                duration,
                terminal_op,
            ]
            terminal_op_cls = AcquireOp

        module = ModuleOp(module_ops)

        wait_ops_before = _wait_ops(module)

        assert len(wait_ops_before) == 1

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        add_ops = _add_ops(module)
        phase_ops = _phase_shift_ops(module)
        wait_ops_after = _wait_ops(module)
        terminal_ops = [node for node in module.walk() if isinstance(node, terminal_op_cls)]

        assert len(add_ops) == 1
        assert len(phase_ops) == 1
        assert len(wait_ops_after) == len(wait_ops_before)
        assert len(terminal_ops) == 1

    def test_commuting_wait_before_phase_shift_preserves_duration_dominance(self):
        """The new WaitOp must not be moved before its duration operand definition."""
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        phase = ConstantOp(PhaseAttr(0.1))
        phase_op = PhaseShiftOp(frame.results[0], phase.results[0])

        wait_duration = ConstantOp(TimeAttr(1.0e-9))
        wait_op = WaitOp(phase_op.results[0], wait_duration.results[0])

        acquire_duration = ConstantOp(TimeAttr(1.0e-6))
        acquire = AcquireOp(wait_op.results[0], acquire_duration.results[0])

        module = ModuleOp(
            [
                frame,
                phase,
                phase_op,
                wait_duration,
                wait_op,
                acquire_duration,
                acquire,
            ]
        )

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops = [node for node in module.walk() if isinstance(node, WaitOp)]
        assert len(wait_ops) == 1

        rewritten_wait = wait_ops[0]
        duration_owner = rewritten_wait.duration.owner

        assert duration_owner.parent is rewritten_wait.parent

        block_ops = list(rewritten_wait.parent.ops)
        assert block_ops.index(duration_owner) < block_ops.index(rewritten_wait)

    def test_phase_set_resets_phase_shifts(self):
        """PhaseSetOp acting as terminal should produce a PhaseSetOp, not PhaseShiftOp."""
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        # Base phase set — establishes an absolute phase.
        phase_set_val = ConstantOp(PhaseAttr(1.0))
        phase_set_op = PhaseSetOp(frame.results[0], phase_set_val.results[0])

        # Two subsequent relative shifts on top of the set.
        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(phase_set_op.results[0], phase1.results[0])
        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseSetOp(phase_op1.results[0], phase2.results[0])
        phase3 = ConstantOp(PhaseAttr(0.1))
        phase_op3 = PhaseShiftOp(phase_op2.results[0], phase3.results[0])

        wf_ops, waveform = _build_waveform_ops()
        pulse = PulseOp(frame=phase_op3.results[0], waveform=waveform.results[0])

        module = ModuleOp(
            [
                frame,
                phase_set_val,
                phase_set_op,
                phase1,
                phase_op1,
                phase2,
                phase_op2,
                phase3,
                phase_op3,
                *wf_ops,
                pulse,
            ]
        )

        phase_sets_before = [node for node in module.walk() if isinstance(node, PhaseSetOp)]
        phase_shifts_before = _phase_shift_ops(module)
        assert len(phase_sets_before) == 2
        assert len(phase_shifts_before) == 2

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_sets_after = [node for node in module.walk() if isinstance(node, PhaseSetOp)]
        phase_shifts_after = _phase_shift_ops(module)

        assert len(phase_sets_after) == 1
        assert len(phase_shifts_after) == 0

    def test_phase_set_resets_accumulated_shifts_and_folds_post_phase_set_shifts(self):
        """PhaseSetOp should reset accumulated shifts and fold post-set phase operations
        separately."""
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        # Base phase set — establishes an absolute phase.
        phase_set_val = ConstantOp(PhaseAttr(1.0))
        phase_set_op = PhaseSetOp(frame.results[0], phase_set_val.results[0])

        # Two subsequent relative shifts on top of the set.
        phase1 = ConstantOp(PhaseAttr(0.1))
        phase_op1 = PhaseShiftOp(phase_set_op.results[0], phase1.results[0])
        phase2 = ConstantOp(PhaseAttr(0.2))
        phase_op2 = PhaseSetOp(phase_op1.results[0], phase2.results[0])
        phase3 = ConstantOp(PhaseAttr(0.3))
        phase_op3 = PhaseShiftOp(phase_op2.results[0], phase3.results[0])

        wf_ops, waveform = _build_waveform_ops()
        pulse = PulseOp(frame=phase_op3.results[0], waveform=waveform.results[0])

        module = ModuleOp(
            [
                frame,
                phase_set_val,
                phase_set_op,
                phase1,
                phase_op1,
                phase2,
                phase_op2,
                phase3,
                phase_op3,
                *wf_ops,
                pulse,
            ]
        )

        phase_sets_before = [node for node in module.walk() if isinstance(node, PhaseSetOp)]
        phase_shifts_before = _phase_shift_ops(module)
        assert len(phase_sets_before) == 2
        assert len(phase_shifts_before) == 2

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_sets_after = [node for node in module.walk() if isinstance(node, PhaseSetOp)]
        phase_shifts_after = _phase_shift_ops(module)

        assert len(phase_sets_after) == 1
        assert len(phase_shifts_after) == 0

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_does_not_fold_when_backtracking_crosses_block_boundary(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        dur1 = ConstantOp(TimeAttr(1.0e-9))
        wait_op1 = WaitOp(frame.results[0], dur1.results[0])

        dur2 = ConstantOp(TimeAttr(2.0e-9))
        wait_op2 = WaitOp(wait_op1.results[0], dur2.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=wait_op2.results[0], waveform=waveform.results[0])
            terminal_side_ops = [dur2, wait_op2, *wf_ops, terminal_op]
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=wait_op2.results[0], duration=duration.results[0])
            terminal_side_ops = [dur2, wait_op2, duration, terminal_op]

        module = ModuleOp([])
        first_block = module.body.block
        first_block.add_ops([frame, dur1, wait_op1])

        second_block = Block()
        module.body.add_block(second_block)
        second_block.add_ops(terminal_side_ops)

        wait_ops_before = _wait_ops(module)
        add_ops_before = _add_ops(module)
        assert len(wait_ops_before) == 2
        assert len(add_ops_before) == 0

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops_after = _wait_ops(module)
        add_ops_after = _add_ops(module)

        # No folding across block boundaries.
        assert len(wait_ops_after) == 2
        assert len(add_ops_after) == 0

    def test_folds_independent_phase_branches_but_not_across_shared_root(self):
        """Two phase branches from one shared frame result fold independently.

        The shared root phase op has multiple uses, so it must not be merged with either
        branch head due to the single-use guard.
        """
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        p0 = ConstantOp(PhaseAttr(0.05))
        shared_root = PhaseShiftOp(frame.results[0], p0.results[0])

        p1a = ConstantOp(PhaseAttr(0.1))
        p2a = ConstantOp(PhaseAttr(0.2))
        branch_a_1 = PhaseShiftOp(shared_root.results[0], p1a.results[0])
        branch_a_2 = PhaseShiftOp(branch_a_1.results[0], p2a.results[0])

        p1b = ConstantOp(PhaseAttr(0.3))
        p2b = ConstantOp(PhaseAttr(0.4))
        branch_b_1 = PhaseShiftOp(shared_root.results[0], p1b.results[0])
        branch_b_2 = PhaseShiftOp(branch_b_1.results[0], p2b.results[0])

        wf_ops_a, waveform_a = _build_waveform_ops()
        terminal_a = PulseOp(frame=branch_a_2.results[0], waveform=waveform_a.results[0])

        wf_ops_b, waveform_b = _build_waveform_ops()
        terminal_b = PulseOp(frame=branch_b_2.results[0], waveform=waveform_b.results[0])

        module = ModuleOp(
            [
                frame,
                p0,
                shared_root,
                p1a,
                p2a,
                branch_a_1,
                branch_a_2,
                p1b,
                p2b,
                branch_b_1,
                branch_b_2,
                *wf_ops_a,
                terminal_a,
                *wf_ops_b,
                terminal_b,
            ]
        )

        phase_ops_before = [
            node for node in module.walk() if isinstance(node, PhaseShiftOp)
        ]
        assert len(phase_ops_before) == 5

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        phase_ops_after = _phase_shift_ops(module)
        add_ops_after = _add_ops(module)

        # shared_root remains (multi-use), each branch pair folds to one.
        assert len(phase_ops_after) == 3
        assert len(add_ops_after) == 2

    def test_phase_set_discards_pre_set_shift_across_wait_and_keeps_post_set_shift(self):
        """Check:
        PhaseShift(p2) -> Wait(w1) -> PhaseSet(p3) -> PhaseShift(p4)
        becomes:
        Wait(w1) -> PhaseSet(p3 + p4).
        """
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        phase2 = ConstantOp(PhaseAttr(0.2))
        shift2 = PhaseShiftOp(frame.results[0], phase2.results[0])

        wait_1 = ConstantOp(TimeAttr(1.0e-9))
        wait_op = WaitOp(shift2.results[0], wait_1.results[0])

        phase3 = ConstantOp(PhaseAttr(0.3))
        set3 = PhaseSetOp(wait_op.results[0], phase3.results[0])

        phase4 = ConstantOp(PhaseAttr(0.4))
        shift4 = PhaseShiftOp(set3.results[0], phase4.results[0])

        wf_ops, waveform = _build_waveform_ops()
        terminal = PulseOp(frame=shift4.results[0], waveform=waveform.results[0])

        module = ModuleOp(
            [
                frame,
                phase2,
                shift2,
                wait_1,
                wait_op,
                phase3,
                set3,
                phase4,
                shift4,
                *wf_ops,
                terminal,
            ]
        )

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops = [node for node in module.walk() if isinstance(node, WaitOp)]
        phase_set_ops = [node for node in module.walk() if isinstance(node, PhaseSetOp)]
        phase_shift_ops = [node for node in module.walk() if isinstance(node, PhaseShiftOp)]
        add_ops = [node for node in module.walk() if isinstance(node, AddOp)]

        assert len(wait_ops) == 1
        assert len(phase_set_ops) == 1
        assert len(phase_shift_ops) == 0
        assert len(add_ops) == 1

        pulse_after = next(node for node in module.walk() if isinstance(node, PulseOp))
        rebuilt_set = pulse_after.frame.owner
        assert isinstance(rebuilt_set, PhaseSetOp)

        rebuilt_wait = rebuilt_set.frame.owner
        assert isinstance(rebuilt_wait, WaitOp)


class TestFoldContiguousWaits:
    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_fold_contiguous_waits(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        dur1 = ConstantOp(TimeAttr(1.0e-9))
        wait_op1 = WaitOp(frame.results[0], dur1.results[0])
        dur2 = ConstantOp(TimeAttr(2.0e-9))
        wait_op2 = WaitOp(wait_op1.results[0], dur2.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=wait_op2.results[0], waveform=waveform.results[0])
            module_ops = [frame, dur1, wait_op1, dur2, wait_op2, *wf_ops, terminal_op]
        elif terminal_op_type == "acquire":
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=wait_op2.results[0], duration=duration.results[0])
            module_ops = [frame, dur1, wait_op1, dur2, wait_op2, duration, terminal_op]

        module = ModuleOp(module_ops)

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops = _wait_ops(module)
        expected_waits = 1
        assert len(wait_ops) == expected_waits
        assert wait_ops[0].duration.owner is not dur1
        assert wait_ops[0].duration.owner is not dur2
        assert isinstance(wait_ops[0].duration.owner, AddOp)

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_does_not_fold_single_wait(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        dur = ConstantOp(TimeAttr(1.0e-9))
        wait_op = WaitOp(frame.results[0], dur.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=wait_op.results[0], waveform=waveform.results[0])
            module_ops = [frame, dur, wait_op, *wf_ops, terminal_op]
        elif terminal_op_type == "acquire":
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=wait_op.results[0], duration=duration.results[0])
            module_ops = [frame, dur, wait_op, duration, terminal_op]

        module = ModuleOp(module_ops)

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops = _wait_ops(module)
        expected_waits = 1
        assert len(wait_ops) == expected_waits
        assert wait_ops[0].duration == dur.results[0]

    def test_does_not_fold_when_waits_are_separated_by_pulse(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        dur1 = ConstantOp(TimeAttr(1.0e-9))
        wait_op1 = WaitOp(frame.results[0], dur1.results[0])

        wf_ops, waveform = _build_waveform_ops()
        pulse_mid = PulseOp(frame=wait_op1.results[0], waveform=waveform.results[0])

        dur2 = ConstantOp(TimeAttr(2.0e-9))
        wait_op2 = WaitOp(pulse_mid.results[0], dur2.results[0])
        pulse_out = PulseOp(frame=wait_op2.results[0], waveform=waveform.results[0])

        module = ModuleOp(
            [frame, dur1, wait_op1, *wf_ops, pulse_mid, dur2, wait_op2, pulse_out]
        )

        wait_ops_before = [node for node in module.walk() if isinstance(node, WaitOp)]
        dur_operands_before = [w.duration for w in wait_ops_before]
        assert len(wait_ops_before) == 2

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops_after = _wait_ops(module)
        dur_operands_after = [w.duration for w in wait_ops_after]

        assert len(wait_ops_after) == 2
        assert dur_operands_after == dur_operands_before

    def test_folds_large_contiguous_chain_and_creates_expected_addops(self):
        n_waits = 100

        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        dur = ConstantOp(TimeAttr(1.0e-9))

        current_frame = frame.results[0]
        wait_ops = []
        for _ in range(n_waits):
            wait = WaitOp(current_frame, dur.results[0])
            wait_ops.append(wait)
            current_frame = wait.results[0]

        wf_ops, waveform = _build_waveform_ops()
        pulse = PulseOp(frame=current_frame, waveform=waveform.results[0])

        module = ModuleOp([frame, dur, *wait_ops, *wf_ops, pulse])

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        merged_wait_ops = _wait_ops(module)
        add_ops = _add_ops(module)

        assert len(merged_wait_ops) == 1
        assert len(add_ops) == n_waits - 1

        node = merged_wait_ops[0].duration.owner
        assert isinstance(node, AddOp)

        for i in range(n_waits - 1):
            assert isinstance(node, AddOp)
            lhs, rhs = node.operands[0].owner, node.operands[1].owner
            if i < n_waits - 2:
                # Interior nodes: one side is ConstantOp, other is next AddOp.
                assert (isinstance(lhs, ConstantOp) and isinstance(rhs, AddOp)) or (
                    isinstance(rhs, ConstantOp) and isinstance(lhs, AddOp)
                )
                node = rhs if isinstance(rhs, AddOp) else lhs
            else:
                # Leaf node: both operands are ConstantOp.
                assert isinstance(lhs, ConstantOp)
                assert isinstance(rhs, ConstantOp)

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_folds_waits_interleaved_with_phase_shifts(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))

        dur1 = ConstantOp(TimeAttr(1.0e-9))
        wait_op1 = WaitOp(frame.results[0], dur1.results[0])

        phase = ConstantOp(PhaseAttr(0.1))
        phase_op = PhaseShiftOp(wait_op1.results[0], phase.results[0])

        dur2 = ConstantOp(TimeAttr(2.0e-9))
        wait_op2 = WaitOp(phase_op.results[0], dur2.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=wait_op2.results[0], waveform=waveform.results[0])
            module_ops = [
                frame,
                dur1,
                wait_op1,
                phase,
                phase_op,
                dur2,
                wait_op2,
                *wf_ops,
                terminal_op,
            ]
            terminal_op_cls = PulseOp
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=wait_op2.results[0], duration=duration.results[0])
            module_ops = [
                frame,
                dur1,
                wait_op1,
                phase,
                phase_op,
                dur2,
                wait_op2,
                duration,
                terminal_op,
            ]
            terminal_op_cls = AcquireOp

        module = ModuleOp(module_ops)

        phase_ops_before = _phase_shift_ops(module)
        assert len(phase_ops_before) == 1

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        add_ops = _add_ops(module)
        wait_ops = _wait_ops(module)
        phase_ops_after = _phase_shift_ops(module)
        terminal_ops = [node for node in module.walk() if isinstance(node, terminal_op_cls)]

        assert len(add_ops) == 1
        assert len(wait_ops) == 1
        assert len(phase_ops_after) == len(phase_ops_before)
        assert len(terminal_ops) == 1

        terminal_op_after = terminal_ops[0]
        rebuilt_phase_op = terminal_op_after.frame.owner
        assert isinstance(rebuilt_phase_op, PhaseShiftOp)

        rebuilt_wait_op = rebuilt_phase_op.frame.owner
        assert isinstance(rebuilt_wait_op, WaitOp)

        accumulated_duration = rebuilt_wait_op.duration.owner
        assert isinstance(accumulated_duration, AddOp)

    @pytest.mark.parametrize("terminal_op_type", ["pulse", "acquire"])
    def test_does_not_fold_when_backtracking_crosses_block_boundary(self, terminal_op_type):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        dur1 = ConstantOp(TimeAttr(1.0e-9))
        wait_op1 = WaitOp(frame.results[0], dur1.results[0])

        dur2 = ConstantOp(TimeAttr(2.0e-9))
        wait_op2 = WaitOp(wait_op1.results[0], dur2.results[0])

        if terminal_op_type == "pulse":
            wf_ops, waveform = _build_waveform_ops()
            terminal_op = PulseOp(frame=wait_op2.results[0], waveform=waveform.results[0])
            terminal_side_ops = [dur2, wait_op2, *wf_ops, terminal_op]
        else:
            duration = _build_acquire_ops()
            terminal_op = AcquireOp(frame=wait_op2.results[0], duration=duration.results[0])
            terminal_side_ops = [dur2, wait_op2, duration, terminal_op]

        module = ModuleOp([])
        first_block = module.body.block
        first_block.add_ops([frame, dur1, wait_op1])

        second_block = Block()
        module.body.add_block(second_block)
        second_block.add_ops(terminal_side_ops)

        wait_ops_before = _wait_ops(module)
        add_ops_before = _add_ops(module)
        assert len(wait_ops_before) == 2
        assert len(add_ops_before) == 0

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops_after = _wait_ops(module)
        add_ops_after = _add_ops(module)

        # No folding across block boundaries
        assert len(wait_ops_after) == 2
        assert len(add_ops_after) == 0

    def test_folds_independent_wait_branches_but_not_across_shared_root(self):
        """Two wait branches from one shared frame result fold independently.

        The shared root wait has multiple uses, so it must not be merged with either branch
        head due to the single-use guard.
        """
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        d0 = ConstantOp(TimeAttr(1.0e-9))
        shared_root = WaitOp(frame.results[0], d0.results[0])

        d1a = ConstantOp(TimeAttr(2.0e-9))
        d2a = ConstantOp(TimeAttr(3.0e-9))
        branch_a_1 = WaitOp(shared_root.results[0], d1a.results[0])
        branch_a_2 = WaitOp(branch_a_1.results[0], d2a.results[0])

        d1b = ConstantOp(TimeAttr(4.0e-9))
        d2b = ConstantOp(TimeAttr(5.0e-9))
        branch_b_1 = WaitOp(shared_root.results[0], d1b.results[0])
        branch_b_2 = WaitOp(branch_b_1.results[0], d2b.results[0])

        wf_ops_a, waveform_a = _build_waveform_ops()
        terminal_a = PulseOp(frame=branch_a_2.results[0], waveform=waveform_a.results[0])

        wf_ops_b, waveform_b = _build_waveform_ops()
        terminal_b = PulseOp(frame=branch_b_2.results[0], waveform=waveform_b.results[0])

        module = ModuleOp(
            [
                frame,
                d0,
                shared_root,
                d1a,
                d2a,
                branch_a_1,
                branch_a_2,
                d1b,
                d2b,
                branch_b_1,
                branch_b_2,
                *wf_ops_a,
                terminal_a,
                *wf_ops_b,
                terminal_b,
            ]
        )

        wait_ops_before = _wait_ops(module)
        assert len(wait_ops_before) == 5

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        wait_ops_after = _wait_ops(module)
        add_ops_after = _add_ops(module)

        assert len(wait_ops_after) == 3
        assert len(add_ops_after) == 2


class TestRandomizedInterleaving:
    def test_random_interleaving_full_pass_folds_phases_and_waits(
        self, function_seed: int
    ) -> None:
        """Randomly interleave phase shifts and waits, then fold both with full pass."""
        n_phases = 100
        n_waits = 100

        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase = ConstantOp(PhaseAttr(2.0 * np.pi / n_phases))
        duration = ConstantOp(TimeAttr(1.0e-9))

        op_kinds = ["phase"] * n_phases + ["wait"] * n_waits
        random.Random(function_seed).shuffle(op_kinds)

        current_frame = frame.results[0]
        chain_ops = []
        for op_kind in op_kinds:
            if op_kind == "phase":
                op = PhaseShiftOp(current_frame, phase.results[0])
            else:
                op = WaitOp(current_frame, duration.results[0])
            chain_ops.append(op)
            current_frame = op.results[0]

        wf_ops, waveform = _build_waveform_ops()
        terminal = PulseOp(frame=current_frame, waveform=waveform.results[0])

        module = ModuleOp([frame, phase, duration, *chain_ops, *wf_ops, terminal])

        assert len([n for n in module.walk() if isinstance(n, AddOp)]) == 0
        assert len([n for n in module.walk() if isinstance(n, PhaseShiftOp)]) == n_phases
        assert len([n for n in module.walk() if isinstance(n, WaitOp)]) == n_waits

        pass_instance = ApplySquashContiguousOptimizations()
        pass_instance.apply(Context(), module)

        add_ops = _add_ops(module)
        phase_ops = _phase_shift_ops(module)
        wait_ops = _wait_ops(module)

        assert len(phase_ops) == 1
        assert len(wait_ops) == 1
        assert len(add_ops) == (n_phases - 1) + (n_waits - 1)

        node = phase_ops[0].phase.owner
        assert isinstance(node, AddOp)

        for i in range(n_phases - 1):
            assert isinstance(node, AddOp)
            lhs, rhs = node.operands[0].owner, node.operands[1].owner
            if i < n_phases - 2:
                # Interior nodes: one side is ConstantOp, other is next AddOp.
                assert (isinstance(lhs, ConstantOp) and isinstance(rhs, AddOp)) or (
                    isinstance(rhs, ConstantOp) and isinstance(lhs, AddOp)
                )
                node = rhs if isinstance(rhs, AddOp) else lhs
            else:
                # Leaf node: both operands are ConstantOp.
                assert isinstance(lhs, ConstantOp)
                assert isinstance(rhs, ConstantOp)
