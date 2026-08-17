# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the pulse kernel builder."""

import numpy as np
import pytest
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.interpreters.scf import scf

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    BlackmanWaveformOp,
    ConstantOp,
    CreateFrameOp,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    IntegrateOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    ReturnOp,
    RoundedSquareWaveformOp,
    SampledWaveformAttr,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinusoidalWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    SynchronizeOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.ops import KernelOp
from qat.experimental.dialect.results.ir import (
    AddRecordOp,
    CreateRecordOp,
    CreateResultsCollectionOp,
)
from qat.experimental.frontend.importer.pulse.builder import PulseKernelBuilder


def _ops(kernel: KernelOp):
    assert isinstance(kernel, KernelOp)
    return list(kernel.walk())


def _ops_of_type(kernel: KernelOp, op_type):
    return [op for op in _ops(kernel) if isinstance(op, op_type)]


def _kernel_body(kernel: KernelOp):
    return list(kernel.body.block.ops)


def _pulse_constant_value(ssa_value):
    owner = ssa_value.owner
    if isinstance(owner, ConstantOp):
        return owner.value.literal_value
    if isinstance(owner, ArithConstantOp):
        return owner.value.value.data
    raise AssertionError(f"Unexpected constant owner type: {type(owner).__name__}")


def _assert_no_repeat_epilogue(kernel: KernelOp):
    body_ops = _kernel_body(kernel)
    assert isinstance(body_ops[-2], CreateRecordOp)
    assert isinstance(body_ops[-1], ReturnOp)
    assert list(body_ops[-1].arguments) == [body_ops[-2].result]


class TestPulseKernelBuilderOperations:
    """Tests the methods to create operations give the expected results, including constant
    producing operations.

    Sets the tests up with no repeats to just test the operations themselves, not the repeat
    logic. Checks the kernel ends in yielding a record.
    """

    def test_create_frame_gives_correct_ops(self):
        """Does a create frame and then finalises, inspecting the frame has the correct
        properties."""
        kernel = (
            PulseKernelBuilder("test").create_frame("q0/drive", 4.8e9, "port0").finalize()
        )

        [frame_op] = _ops_of_type(kernel, CreateFrameOp)
        assert frame_op.port.data == "port0"
        assert _pulse_constant_value(frame_op.frequency) == pytest.approx(4.8e9)
        _assert_no_repeat_epilogue(kernel)

    def test_phase_set_gives_correct_ops(self):
        """Does a phase set and then finalises, inspecting the phase set has the correct
        properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/drive", 4.8e9, "port0")
            .phase_set("q0/drive", 0.25)
            .finalize()
        )

        [frame_op] = _ops_of_type(kernel, CreateFrameOp)
        [phase_set_op] = _ops_of_type(kernel, PhaseSetOp)
        assert phase_set_op.frame is frame_op.result
        assert _pulse_constant_value(phase_set_op.phase) == pytest.approx(0.25)
        _assert_no_repeat_epilogue(kernel)

    def test_phase_shift_gives_correct_ops(self):
        """Does a phase shift and then finalises, inspecting the phase shift has the correct
        properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/drive", 4.8e9, "port0")
            .phase_shift("q0/drive", 0.5)
            .finalize()
        )

        [frame_op] = _ops_of_type(kernel, CreateFrameOp)
        [phase_shift_op] = _ops_of_type(kernel, PhaseShiftOp)
        assert phase_shift_op.frame is frame_op.result
        assert _pulse_constant_value(phase_shift_op.phase) == pytest.approx(0.5)
        _assert_no_repeat_epilogue(kernel)

    def test_wait_gives_correct_ops(self):
        """Does a wait and then finalises, inspecting the wait has the correct
        properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/drive", 4.8e9, "port0")
            .wait("q0/drive", 120e-9)
            .finalize()
        )

        [frame_op] = _ops_of_type(kernel, CreateFrameOp)
        [wait_op] = _ops_of_type(kernel, WaitOp)
        assert wait_op.frame is frame_op.result
        assert _pulse_constant_value(wait_op.duration) == pytest.approx(120e-9)
        _assert_no_repeat_epilogue(kernel)

    def test_synchronize_gives_correct_ops(self):
        """Does a synchronize and then finalises, inspecting the synchronize has the correct
        properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/drive", 4.8e9, "port0")
            .create_frame("q1/drive", 5.2e9, "port1")
            .synchronize("q0/drive", "q1/drive")
            .finalize()
        )

        frame_ops = _ops_of_type(kernel, CreateFrameOp)
        [sync_op] = _ops_of_type(kernel, SynchronizeOp)
        assert list(sync_op.frames) == [frame_ops[0].result, frame_ops[1].result]
        assert len(sync_op.result) == 2
        _assert_no_repeat_epilogue(kernel)

    def test_acquire_without_weights_gives_correct_ops(self):
        """Does an acquire without weights and then finalises, inspecting the acquire has
        the correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/measure", 8.8e9, "port0")
            .acquire("q0/measure", "m0", 1e-6, integrate=False)
            .finalize()
        )

        [frame_op] = _ops_of_type(kernel, CreateFrameOp)
        [acquire_op] = _ops_of_type(kernel, AcquireOp)
        assert acquire_op.frame is frame_op.result
        assert _pulse_constant_value(acquire_op.duration) == pytest.approx(1e-6)
        assert acquire_op.weights is None
        assert acquire_op.label is not None
        assert acquire_op.label.data == "m0"
        record_op = _ops_of_type(kernel, CreateRecordOp)[0]
        assert list(record_op.values) == [acquire_op.acquisition_result]
        _assert_no_repeat_epilogue(kernel)

    def test_acquire_with_weights_gives_correct_ops(self):
        """Does an acquire with weights and then finalises, inspecting the acquire has the
        correct properties."""
        weights = [0.1, 0.2, 0.3]
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/measure", 8.8e9, "port0")
            .acquire("q0/measure", "m0", 1e-6, weights=weights, integrate=False)
            .finalize()
        )

        [acquire_op] = _ops_of_type(kernel, AcquireOp)
        assert list(acquire_op.weights.weights.data) == pytest.approx(weights)
        assert acquire_op.label is not None
        assert acquire_op.label.data == "m0"
        record_op = _ops_of_type(kernel, CreateRecordOp)[0]
        assert list(record_op.values) == [acquire_op.acquisition_result]
        _assert_no_repeat_epilogue(kernel)

    def test_acquire_with_integrate_gives_correct_ops(self):
        """Does an acquire with integrate and then finalises, inspecting the acquire has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/measure", 8.8e9, "port0")
            .acquire("q0/measure", "m0", 1e-6, integrate=True)
            .finalize()
        )

        [acquire_op] = _ops_of_type(kernel, AcquireOp)
        [integrate_op] = _ops_of_type(kernel, IntegrateOp)
        assert integrate_op.acquisition is acquire_op.acquisition_result
        record_op = _ops_of_type(kernel, CreateRecordOp)[0]
        assert list(record_op.values) == [integrate_op.result]
        _assert_no_repeat_epilogue(kernel)

    def test_pulse_with_square_waveform_gives_correct_ops(self):
        """Does a pulse with square waveform and then finalises, inspecting the pulse has
        the correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/drive", 4.8e9, "port0")
            .create_square_waveform("wf", 0.4, 80e-9)
            .pulse("q0/drive", "wf")
            .finalize()
        )

        [frame_op] = _ops_of_type(kernel, CreateFrameOp)
        [waveform_op] = _ops_of_type(kernel, SquareWaveformOp)
        [pulse_op] = _ops_of_type(kernel, PulseOp)
        assert pulse_op.frame is frame_op.result
        assert pulse_op.waveform is waveform_op.result
        _assert_no_repeat_epilogue(kernel)

    def test_pulse_with_custom_waveform_gives_correct_ops(self):
        """Custom waveforms should be consumable by pulse operations."""
        kernel = (
            PulseKernelBuilder("test")
            .create_frame("q0/drive", 4.8e9, "port0")
            .create_custom_waveform("wf", [0.1, -0.1j, 0.2 + 0.3j], 12e-9)
            .pulse("q0/drive", "wf")
            .finalize()
        )

        [pulse_op] = _ops_of_type(kernel, PulseOp)
        assert isinstance(pulse_op.waveform.owner, ConstantOp)
        assert isinstance(pulse_op.waveform.owner.value, SampledWaveformAttr)
        _assert_no_repeat_epilogue(kernel)


class TestPulseKernelWaveformOperations:
    """Tests the methods to create waveforms."""

    @staticmethod
    def _assert_waveform_op(
        kernel: KernelOp, op_type, expected_values, prop_assertion=None
    ):
        ops = _ops_of_type(kernel, op_type)
        assert len(ops) == 1
        op = ops[0]
        operand_values = [_pulse_constant_value(operand) for operand in op.operands]
        assert len(operand_values) == len(expected_values)
        for actual, expected in zip(operand_values, expected_values, strict=True):
            assert actual == pytest.approx(expected)
        if prop_assertion is not None:
            prop_assertion(op)
        _assert_no_repeat_epilogue(kernel)
        return op

    @staticmethod
    def _assert_zero_at_edges(op):
        assert bool(op.regularize.value.data)

    def test_square_waveform_gives_correct_ops(self):
        """Does a square waveform and then finalises, inspecting the waveform has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test").create_square_waveform("wf", 0.5, 80e-9).finalize()
        )
        self._assert_waveform_op(kernel, SquareWaveformOp, [80e-9, 0.5])

    def test_gaussian_waveform_gives_correct_ops(self):
        """Does a gaussian waveform and then finalises, inspecting the waveform has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_gaussian_waveform("wf", 0.5, 80e-9, 0.4)
            .finalize()
        )
        self._assert_waveform_op(kernel, GaussianWaveformOp, [80e-9, 0.5, 0.4])

    def test_gaussian_square_waveform_gives_correct_ops(self):
        """Does a gaussian square waveform and then finalises, inspecting the waveform has
        the correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_gaussian_square_waveform("wf", 0.5, 80e-9, 0.3, 0.5, True)
            .finalize()
        )
        self._assert_waveform_op(
            kernel,
            GaussianSquareWaveformOp,
            [80e-9, 0.5, 0.3, 0.5],
            self._assert_zero_at_edges,
        )

    def test_gaussian_square_waveform_with_drag_coefficients_gives_correct_ops(self):
        """Gaussian-square waveforms should allow one first-order DRAG coefficient."""
        kernel = (
            PulseKernelBuilder("test")
            .create_gaussian_square_waveform("wf", 0.5, 80e-9, 0.3, 0.5, True, 0.1)
            .finalize()
        )
        self._assert_waveform_op(
            kernel,
            GaussianSquareWaveformOp,
            [80e-9, 0.5, 0.3, 0.5, 0.1],
            self._assert_zero_at_edges,
        )

    def test_soft_square_waveform_gives_correct_ops(self):
        """Does a soft square waveform and then finalises, inspecting the waveform has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_soft_square_waveform("wf", 0.5, 80e-9, 0.75, 0.2)
            .finalize()
        )
        self._assert_waveform_op(kernel, SoftSquareWaveformOp, [80e-9, 0.5, 0.75, 0.2])

    def test_regularized_soft_square_waveform_gives_correct_ops(self):
        """Does a regularized soft square waveform and then finalises, inspecting the
        waveform has the correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_soft_square_waveform("wf", 0.5, 80e-9, 0.75, 0.2, True)
            .finalize()
        )
        self._assert_waveform_op(
            kernel,
            SoftSquareWaveformOp,
            [80e-9, 0.5, 0.75, 0.2],
            self._assert_zero_at_edges,
        )

    def test_regularized_gaussian_waveform_gives_correct_ops(self):
        """Does a regularized gaussian waveform and then finalises, inspecting the waveform
        has the correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_gaussian_waveform("wf", 0.5, 80e-9, 0.4, True)
            .finalize()
        )
        self._assert_waveform_op(
            kernel,
            GaussianWaveformOp,
            [80e-9, 0.5, 0.4],
            self._assert_zero_at_edges,
        )

    def test_gaussian_waveform_with_drag_coefficients_gives_correct_ops(self):
        """Gaussian waveforms should accept variadic DRAG coefficients."""
        kernel = (
            PulseKernelBuilder("test")
            .create_gaussian_waveform("wf", 0.5, 80e-9, 0.4, False, 0.1, 0.2)
            .finalize()
        )
        self._assert_waveform_op(kernel, GaussianWaveformOp, [80e-9, 0.5, 0.4, 0.1, 0.2])

    def test_blackman_waveform_gives_correct_ops(self):
        """Does a Blackman waveform and then finalises, inspecting the waveform has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test").create_blackman_waveform("wf", 0.5, 80e-9).finalize()
        )
        self._assert_waveform_op(kernel, BlackmanWaveformOp, [80e-9, 0.5])

    def test_setup_hold_waveform_gives_correct_ops(self):
        """Does a setup hold waveform and then finalises, inspecting the waveform has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_setup_hold_waveform("wf", 0.5, 80e-9, 0.25, 0.2)
            .finalize()
        )
        self._assert_waveform_op(kernel, SetupHoldWaveformOp, [80e-9, 0.5, 0.25, 0.2])

    def test_rounded_square_waveform_gives_correct_ops(self):
        """Does a rounded square waveform and then finalises, inspecting the waveform has
        the correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_rounded_square_waveform("wf", 0.5, 80e-9, 0.5, 0.2)
            .finalize()
        )
        self._assert_waveform_op(kernel, RoundedSquareWaveformOp, [80e-9, 0.5, 0.5, 0.2])

    def test_sech_waveform_gives_correct_ops(self):
        """Does a sech waveform and then finalises, inspecting the waveform has the correct
        properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_sech_waveform("wf", 0.5, 80e-9, 0.3)
            .finalize()
        )
        self._assert_waveform_op(kernel, SechWaveformOp, [80e-9, 0.5, 0.3])

    def test_sinusoidal_waveform_gives_correct_ops(self):
        """Does a sinusoidal waveform and then finalises, inspecting the waveform has the
        correct properties."""
        kernel = (
            PulseKernelBuilder("test")
            .create_sinusoidal_waveform("wf", 0.5, 80e-9, 3.0, 0.25)
            .finalize()
        )
        self._assert_waveform_op(kernel, SinusoidalWaveformOp, [80e-9, 0.5, 3.0, 0.25])

    def test_custom_waveform_gives_correct_attribute_payload(self):
        """Custom waveforms should be emitted as a constant sampled waveform."""
        samples = [0.0, 0.25 + 0.5j, -0.75]
        duration = 12e-9
        kernel = (
            PulseKernelBuilder("test")
            .create_custom_waveform("wf", samples, duration)
            .finalize()
        )

        constant_ops = _ops_of_type(kernel, ConstantOp)
        sampled_constants = [
            op for op in constant_ops if isinstance(op.value, SampledWaveformAttr)
        ]
        assert len(sampled_constants) == 1

        attr = sampled_constants[0].value
        assert isinstance(attr, SampledWaveformAttr)
        assert np.allclose(attr.samples.data, np.asarray(samples, dtype=np.complex128))
        assert attr.width.literal_value == pytest.approx(duration)
        assert attr.sample_time.literal_value == pytest.approx(duration / len(samples))
        _assert_no_repeat_epilogue(kernel)


class TestPulseKernelBuilderWithRepeats:
    """Tests that when repeats is used, the correct SCF operations are added, including
    results collections."""

    def test_repeats_adds_for_loop_and_results_collection(self):
        """Does a kernel with repeats and then finalises, inspecting the kernel has the
        correct properties."""
        kernel = PulseKernelBuilder("test", shots=4).finalize()

        body_ops = _kernel_body(kernel)
        assert len(body_ops) == 6
        assert isinstance(body_ops[0], ArithConstantOp)
        assert isinstance(body_ops[1], ArithConstantOp)
        assert isinstance(body_ops[2], ArithConstantOp)
        assert isinstance(body_ops[3], CreateResultsCollectionOp)
        assert isinstance(body_ops[4], scf.ForOp)
        assert isinstance(body_ops[5], ReturnOp)

        assert body_ops[0].value.value.data == 0
        assert body_ops[1].value.value.data == 4
        assert body_ops[2].value.value.data == 1

        loop = body_ops[4]
        loop_body_ops = list(loop.body.block.ops)
        assert isinstance(loop_body_ops[-3], CreateRecordOp)
        assert isinstance(loop_body_ops[-2], AddRecordOp)
        assert isinstance(loop_body_ops[-1], scf.YieldOp)
        assert loop_body_ops[-2].collection is loop.body.block.args[1]
        assert loop_body_ops[-2].record is loop_body_ops[-3].result
        assert list(body_ops[5].arguments) == [loop.results[0]]


class TestPulseKernelBuilderErrors:
    """Tests that when errors are raised, they are raised correctly."""

    def test_operation_on_unknown_frame_raises_key_error(self):
        """Does an operation on a frame that has not been created, and checks the correct
        error is raised."""
        builder = PulseKernelBuilder("test")
        with pytest.raises(KeyError, match="Frame 'missing' not found"):
            builder.phase_set("missing", 0.25)

    def test_operation_with_unknown_waveform_raises_key_error(self):
        """Does an operation with a waveform that has not been created, and checks the
        correct error is raised."""
        builder = PulseKernelBuilder("test").create_frame("q0/drive", 4.8e9, "port0")
        with pytest.raises(KeyError, match="Waveform 'missing' not found"):
            builder.pulse("q0/drive", "missing")

    def test_sampled_waveform_with_empty_samples_raises_value_error(self):
        """Attempts to create a sampled waveform with an empty list of samples, and checks
        the correct error is raised."""
        builder = PulseKernelBuilder("test")
        with pytest.raises(
            ValueError, match="Samples list cannot be empty for the sampled waveform."
        ):
            builder.create_custom_waveform("wf", [], 12e-9)
