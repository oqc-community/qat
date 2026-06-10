# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests no-op elimination on waits with zero duration and phase shifts with modulo 2pi
equal to zero."""

from math import pi

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.irdl import AnyOf, IRDLOperation, irdl_op_definition, result_def
from xdsl.transforms.canonicalize import CanonicalizePass

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    PhaseShiftOp,
    PhaseType,
    Pulse,
    TimeAttr,
    TimeType,
    WaitOp,
)

from tests.unit.utils.ir import (
    build_module_from_ops,
    create_context,
    get_operations_with_type,
)

_CONTEXT = create_context(Pulse)


@irdl_op_definition
class _DummyOp(IRDLOperation):
    """A dummy operation that returns a PhaseType or TimeType, used to test that non-
    constant phases and durations are not eliminated."""

    name = "dummy"
    result = result_def(AnyOf((PhaseType, TimeType)))

    def __init__(self, result_type):
        super().__init__(result_types=[result_type])


class TestPhaseShiftElimination:
    """Tests that phase shifts with modulo 2pi equal to zero are removed."""

    @pytest.mark.parametrize("phase_value", [0.0, 2 * pi, -2 * pi, 4 * pi, 1e-12, -1e-12])
    def test_canonicalization_with_zero_phase_shift_is_eliminated(self, phase_value):
        """Tests that a phase shift with the given value is removed by canonicalization."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        constant_phase = ConstantOp(PhaseAttr(phase_value))
        shifted_frame = PhaseShiftOp(frame, constant_phase)
        time_const = ConstantOp(TimeAttr(100e-9))
        acquire = AcquireOp(shifted_frame, time_const)
        module = build_module_from_ops(
            [freq_const, frame, constant_phase, shifted_frame, time_const, acquire]
        )

        # Apply canonicalization, check it's removed
        CanonicalizePass().apply(_CONTEXT, module)
        phase_shift_ops = get_operations_with_type(module, PhaseShiftOp)
        assert len(phase_shift_ops) == 0, (
            f"Expected no PhaseShiftOps, but found {len(phase_shift_ops)}"
        )
        assert acquire.frame is frame.result

    def test_canonicalization_with_no_phase_shift(self):
        """Tests that a frame with no phase shift is not affected by canonicalization."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        module = build_module_from_ops([freq_const, frame])

        # Apply canonicalization, check the frame is still there and unchanged
        _, new_module = CanonicalizePass().apply_to_clone(_CONTEXT, module)
        assert new_module is not module, "Expected a new module to be returned"
        assert new_module.is_structurally_equivalent(module), (
            "Expected the module to be unchanged after canonicalization"
        )

    def test_canonicalization_with_non_zero_phase_shift_is_not_eliminated(self):
        """Tests that a phase shift with a non-zero modulo 2pi value is not removed by
        canonicalization."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        constant_phase = ConstantOp(PhaseAttr(pi / 2))
        shifted_frame = PhaseShiftOp(frame, constant_phase)
        module = build_module_from_ops([freq_const, frame, constant_phase, shifted_frame])

        # Apply canonicalization, check it's not removed
        CanonicalizePass().apply(_CONTEXT, module)
        phase_shift_ops = get_operations_with_type(module, PhaseShiftOp)
        assert len(phase_shift_ops) == 1, (
            f"Expected 1 PhaseShiftOp, but found {len(phase_shift_ops)}"
        )

    def test_canonicalization_with_non_constant_phase(self):
        """Tests with a mock operation that returns a PhaseType that cannot be assumed
        constant."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        non_constant_phase = _DummyOp(PhaseType())
        shifted_frame = PhaseShiftOp(frame, non_constant_phase)
        module = build_module_from_ops(
            [freq_const, frame, non_constant_phase, shifted_frame]
        )

        # Apply canonicalization, check it's not removed
        _, new_module = CanonicalizePass().apply_to_clone(_CONTEXT, module)
        assert new_module is not module, "Expected a new module to be returned"
        assert new_module.is_structurally_equivalent(module), (
            "Expected the module to be unchanged after canonicalization"
        )

    def test_canonicalization_on_multiple_phase_shifts(self):
        """Tests that multiple phase shifts with zero modulo 2pi values are all removed."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        constant_phase1 = ConstantOp(PhaseAttr(0.0))
        constant_phase2 = ConstantOp(PhaseAttr(2 * pi))
        shifted_frame1 = PhaseShiftOp(frame, constant_phase1)
        shifted_frame2 = PhaseShiftOp(shifted_frame1, constant_phase2)
        module = build_module_from_ops(
            [
                freq_const,
                frame,
                constant_phase1,
                constant_phase2,
                shifted_frame1,
                shifted_frame2,
            ]
        )

        # Apply canonicalization, check both are removed
        CanonicalizePass().apply(_CONTEXT, module)
        phase_shift_ops = get_operations_with_type(module, PhaseShiftOp)
        assert len(phase_shift_ops) == 0, (
            f"Expected no PhaseShiftOps, but found {len(phase_shift_ops)}"
        )
        assert frame.result.uses.get_length() == 0


class TestWaitElimination:
    """Tests that waits with zero duration are removed."""

    @pytest.mark.parametrize("duration_value", [0, 0.0, 1e-12])
    def test_canonicalization_with_zero_duration_wait_is_eliminated(self, duration_value):
        """Tests that a wait with zero duration is removed by canonicalization."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        zero_duration = ConstantOp(TimeAttr(duration_value))
        wait_op = WaitOp(frame, zero_duration)
        acquire_duration = ConstantOp(TimeAttr(100e-9))
        acquire_op = AcquireOp(wait_op, acquire_duration)
        module = build_module_from_ops(
            [freq_const, frame, zero_duration, wait_op, acquire_duration, acquire_op]
        )

        # Apply canonicalization, check it's removed
        CanonicalizePass().apply(_CONTEXT, module)
        wait_ops = get_operations_with_type(module, WaitOp)
        assert len(wait_ops) == 0, f"Expected no WaitOps, but found {len(wait_ops)}"
        assert acquire_op.frame is frame.result

    def test_canonicalization_with_non_zero_duration_wait_is_not_eliminated(self):
        """Tests that a wait with non-zero duration is not removed by canonicalization."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        non_zero_duration = ConstantOp(TimeAttr(1e-9))
        wait_op = WaitOp(frame, non_zero_duration)
        acquire_duration = ConstantOp(TimeAttr(100e-9))
        acquire_op = AcquireOp(wait_op, acquire_duration)
        module = build_module_from_ops(
            [freq_const, frame, non_zero_duration, wait_op, acquire_duration, acquire_op]
        )

        # Apply canonicalization, check it's not removed
        CanonicalizePass().apply(_CONTEXT, module)
        wait_ops = get_operations_with_type(module, WaitOp)
        assert len(wait_ops) == 1, f"Expected 1 WaitOp, but found {len(wait_ops)}"
        assert acquire_op.frame is wait_op.result

    def test_canonicalization_with_no_wait(self):
        """Tests that a frame with no wait is not affected by canonicalization."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        module = build_module_from_ops([freq_const, frame])

        # Apply canonicalization, check the frame is still there and unchanged
        _, new_module = CanonicalizePass().apply_to_clone(_CONTEXT, module)
        assert new_module is not module, "Expected a new module to be returned"
        assert new_module.is_structurally_equivalent(module), (
            "Expected the module to be unchanged after canonicalization"
        )

    def test_canonicalization_with_non_constant_duration(self):
        """Tests with a mock operation that returns a TimeType that cannot be assumed
        constant."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        non_constant_duration = _DummyOp(TimeType())
        wait_op = WaitOp(frame, non_constant_duration)
        module = build_module_from_ops([freq_const, frame, non_constant_duration, wait_op])

        # Apply canonicalization, check it's not removed
        _, new_module = CanonicalizePass().apply_to_clone(_CONTEXT, module)
        assert new_module is not module, "Expected a new module to be returned"
        assert new_module.is_structurally_equivalent(module), (
            "Expected the module to be unchanged after canonicalization"
        )

    def test_canonicalization_on_multiple_waits(self):
        """Tests that multiple waits with zero durations are all removed."""

        # Set up the IR
        freq = FrequencyAttr(5.5e9)
        freq_const = ConstantOp(freq)
        frame = CreateFrameOp(freq_const, StringAttr("port"))
        zero_duration = ConstantOp(TimeAttr(0.0))
        wait_op1 = WaitOp(frame, zero_duration)
        wait_op2 = WaitOp(wait_op1, zero_duration)
        acquire_duration = ConstantOp(TimeAttr(100e-9))
        acquire_op = AcquireOp(wait_op2, acquire_duration)
        module = build_module_from_ops(
            [
                freq_const,
                frame,
                zero_duration,
                wait_op1,
                wait_op2,
                acquire_duration,
                acquire_op,
            ]
        )

        # Apply canonicalization, check both are removed
        CanonicalizePass().apply(_CONTEXT, module)
        wait_ops = get_operations_with_type(module, WaitOp)
        assert len(wait_ops) == 0, f"Expected no WaitOps, but found {len(wait_ops)}"
        assert acquire_op.frame is frame.result
