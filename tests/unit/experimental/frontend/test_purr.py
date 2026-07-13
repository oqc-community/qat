# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :class:`PurrImporter` (Purr -> Pulse dialect importer)."""

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import StringAttr
from xdsl.interpreters.scf import scf

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    BlackmanWaveformOp,
    ConstantOp,
    CosWaveformOp,
    CreateFrameOp,
    DragGaussianWaveformOp,
    ExtraSoftSquareWaveformOp,
    FrequencyAttr,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    GaussianZeroEdgeWaveformOp,
    IntegrateOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    RoundedSquareWaveformOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinWaveformOp,
    SofterGaussianWaveformOp,
    SofterSquareWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.frontend.importer.purr import PurrImporter
from qat.ir.instruction_basetypes import AcquireMode
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    CustomPulse,
    Delay,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    PostProcessing,
    Pulse,
    Repeat,
    Sweep,
    Synchronize,
    Variable,
)


@pytest.fixture
def hw():
    return get_default_echo_hardware()


@pytest.fixture
def builder(hw):
    return QuantumInstructionBuilder(hw)


def _ops(importer: PurrImporter):
    """Return the translated ops from inside the module's ``main`` function body."""
    [main] = list(importer.module.body.block.ops)
    assert isinstance(main, func.FuncOp)
    return list(main.body.block.ops)


def _ops_of_type(importer: PurrImporter, op_type):
    return [op for op in _ops(importer) if isinstance(op, op_type)]


class TestPurrImporterPhase:
    def test_phase_shift_emits_phase_shift_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 1.3))
        imp = PurrImporter()
        imp.build(builder)
        phase_ops = _ops_of_type(imp, PhaseShiftOp)
        assert len(phase_ops) == 1
        phase_const = phase_ops[0].phase.owner
        assert isinstance(phase_const, ConstantOp)
        assert phase_const.value.value.data == pytest.approx(1.3)

    def test_phase_set_emits_phase_set_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseSet(ch, 0.75))
        imp = PurrImporter()
        imp.build(builder)
        phase_set_ops = _ops_of_type(imp, PhaseSetOp)
        assert len(phase_set_ops) == 1
        assert phase_set_ops[0].phase.owner.value.value.data == pytest.approx(0.75)

    def test_phase_reset_emits_phase_set_to_zero(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseReset(ch))
        imp = PurrImporter()
        imp.build(builder)
        phase_set_ops = _ops_of_type(imp, PhaseSetOp)
        assert len(phase_set_ops) == 1
        assert phase_set_ops[0].phase.owner.value.value.data == pytest.approx(0.0)

    def test_phase_reset_on_multiple_channels(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(PhaseReset([ch0, ch1]))
        imp = PurrImporter()
        imp.build(builder)
        assert len(_ops_of_type(imp, PhaseSetOp)) == 2


class TestPurrImporterDelayAndSync:
    def test_delay_emits_wait_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Delay(ch, 320e-9))
        imp = PurrImporter()
        imp.build(builder)
        wait_ops = _ops_of_type(imp, WaitOp)
        assert len(wait_ops) == 1
        assert wait_ops[0].duration.owner.value.value.data == pytest.approx(320e-9)

    def test_synchronize_single_target_emits_no_sync_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Synchronize(ch))
        imp = PurrImporter()
        imp.build(builder)
        assert _ops_of_type(imp, SynchronizeOp) == []

    def test_synchronize_multi_targets_emits_sync_op(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(Synchronize([ch0, ch1]))
        imp = PurrImporter()
        imp.build(builder)
        sync_ops = _ops_of_type(imp, SynchronizeOp)
        assert len(sync_ops) == 1
        assert len(sync_ops[0].frames) == 2


class TestPurrImporterFrameTracking:
    def test_first_use_creates_frame(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.5))
        imp = PurrImporter()
        imp.build(builder)

        create_frames = _ops_of_type(imp, CreateFrameOp)
        assert len(create_frames) == 1
        assert create_frames[0].physical_channel.data == ch.physical_channel.full_id()

    def test_frame_reused_across_instructions(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.5))
        builder.add(PhaseShift(ch, 0.25))
        imp = PurrImporter()
        imp.build(builder)
        # Only one frame creation; later PhaseShifts consume the latest result.
        assert len(_ops_of_type(imp, CreateFrameOp)) == 1
        assert len(_ops_of_type(imp, PhaseShiftOp)) == 2

    def test_distinct_channels_create_distinct_frames(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(PhaseShift(ch0, 0.1))
        builder.add(PhaseShift(ch1, 0.1))
        imp = PurrImporter()
        imp.build(builder)
        assert len(_ops_of_type(imp, CreateFrameOp)) == 2

    def test_get_frame_key_uses_full_id(self, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        assert PurrImporter.get_frame_key(ch) == "purr_frame_" + ch.full_id()

    def test_chain_of_phase_shifts_threads_through_frame_results(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        builder.add(PhaseShift(ch, 0.2))
        builder.add(PhaseShift(ch, 0.3))
        imp = PurrImporter()
        imp.build(builder)
        create_frame_ops = _ops_of_type(imp, CreateFrameOp)
        assert len(create_frame_ops) == 1
        shifts = _ops_of_type(imp, PhaseShiftOp)
        assert shifts[0].frame is create_frame_ops[0].result
        assert shifts[1].frame is shifts[0].result
        assert shifts[2].frame is shifts[1].result

    def test_synchronize_threads_through_all_frames(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(PhaseShift(ch0, 0.1))
        builder.add(PhaseShift(ch1, 0.2))
        builder.add(Synchronize([ch0, ch1]))
        builder.add(PhaseShift(ch0, 0.3))
        builder.add(PhaseShift(ch1, 0.4))
        imp = PurrImporter()
        imp.build(builder)
        [sync] = _ops_of_type(imp, SynchronizeOp)
        shifts = _ops_of_type(imp, PhaseShiftOp)

        # Check the first set of shift's frames are threaded to the sync
        first_shifts = shifts[:2]
        assert {first_shifts[0].result, first_shifts[1].result} == set(sync.frames)
        # Check the second set of shift's frames are threaded from the sync
        last_shifts = shifts[2:]
        assert {last_shifts[0].frame, last_shifts[1].frame} == set(sync.results)

    def test_mixed_op_chain_threads_frame(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        builder.add(Delay(ch, 100e-9))
        builder.add(Pulse(ch, PulseShapeType.SQUARE, width=80e-9, amp=0.4))
        imp = PurrImporter()
        imp.build(builder)
        [shift] = _ops_of_type(imp, PhaseShiftOp)
        [wait] = _ops_of_type(imp, WaitOp)
        [pulse] = _ops_of_type(imp, PulseOp)
        assert wait.frame is shift.result
        assert pulse.frame is wait.result


class TestPurrImporterAcquire:
    def test_acquire(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6))
        imp = PurrImporter()
        imp.build(builder)
        acq_ops = _ops_of_type(imp, AcquireOp)
        assert len(acq_ops) == 1
        assert acq_ops[0].duration.owner.value.value.data == pytest.approx(1e-6)
        # No waveform constructed when no filter is given.
        assert not any(isinstance(op, SquareWaveformOp) for op in _ops(imp))

    def test_acquire_with_filter_emits_waveform(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        filt = Pulse(ch, PulseShapeType.SQUARE, width=1e-6, amp=0.3)
        builder.add(Acquire(ch, time=1e-6, filter=filt))
        imp = PurrImporter()
        imp.build(builder)
        assert len(_ops_of_type(imp, AcquireOp)) == 1
        assert len(_ops_of_type(imp, SquareWaveformOp)) == 1

    def test_acquire_with_integrator_emits_integrate_op(self, builder, hw):
        """When the INTEGRATOR mode is used, an IntegrateOp is emitted consuming the
        acquisition result."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, mode=AcquireMode.INTEGRATOR))
        imp = PurrImporter()
        imp.build(builder)
        acq_ops = _ops_of_type(imp, AcquireOp)
        assert len(acq_ops) == 1
        int_ops = _ops_of_type(imp, IntegrateOp)
        assert len(int_ops) == 1
        assert int_ops[0].acquisition is acq_ops[0].acquisition_result

    def test_acquire_with_scope_raises_not_implemented_error(self, builder, hw):
        """When the SCOPE mode is used, a NotImplementedError is raised."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, mode=AcquireMode.SCOPE))
        imp = PurrImporter()
        with pytest.raises(NotImplementedError, match="Scope mode is not yet supported"):
            imp.build(builder)


class TestPurrImporterUnsupportedInstructions:
    def test_custom_pulse_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(CustomPulse(ch, np.zeros(8)))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Not yet supported"):
            imp.build(builder)

    def test_post_processing_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        acq = Acquire(ch, time=1e-6)
        from qat.ir.instruction_basetypes import PostProcessType

        builder.add(PostProcessing(acq, PostProcessType.MEAN))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Not yet supported"):
            imp.build(builder)

    def test_sweep_raises(self, builder):
        builder.add(Sweep())
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Not yet implemented"):
            imp.build(builder)

    def test_end_sweep_raises(self, builder):
        builder.add(EndSweep())
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Not yet implemented"):
            imp.build(builder)


WAVEFORM_CASES = [
    (
        {"shape": PulseShapeType.SQUARE, "width": 80e-9, "amp": 0.5},
        SquareWaveformOp,
    ),
    (
        {"shape": PulseShapeType.GAUSSIAN, "width": 80e-9, "amp": 0.5, "rise": 16e-9},
        GaussianWaveformOp,
    ),
    (
        {"shape": PulseShapeType.SOFT_SQUARE, "width": 80e-9, "amp": 0.5, "rise": 1e-9},
        SoftSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SOFTER_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "rise": 1e-9,
        },
        SofterSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.EXTRA_SOFT_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "rise": 1e-9,
        },
        ExtraSoftSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.GAUSSIAN_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "square_width": 40e-9,
            "zero_at_edges": 1,
        },
        GaussianSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SOFTER_GAUSSIAN,
            "width": 80e-9,
            "amp": 0.5,
            "rise": 16e-9,
        },
        SofterGaussianWaveformOp,
    ),
    (
        {"shape": PulseShapeType.BLACKMAN, "width": 80e-9, "amp": 0.5},
        BlackmanWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SETUP_HOLD,
            "width": 80e-9,
            "amp": 0.5,
            "amp_setup": 0.25,
            "rise": 16e-9,
        },
        SetupHoldWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.ROUNDED_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "rise": 1e-9,
            "std_dev": 8e-9,
        },
        RoundedSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.GAUSSIAN_DRAG,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "beta": 0.1,
            "zero_at_edges": 0,
        },
        DragGaussianWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.GAUSSIAN_ZERO_EDGE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "zero_at_edges": 1,
        },
        GaussianZeroEdgeWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SECH,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
        },
        SechWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.COS,
            "width": 80e-9,
            "amp": 0.5,
            "frequency": 5e9,
            "internal_phase": 0.5,
        },
        CosWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SIN,
            "width": 80e-9,
            "amp": 0.5,
            "frequency": 5e9,
            "internal_phase": 0.5,
        },
        SinWaveformOp,
    ),
]


class TestPurrImporterWaveformTranslation:
    @pytest.mark.parametrize(
        "pulse_kwargs,expected_op_type",
        WAVEFORM_CASES,
        ids=[case[0]["shape"].name for case in WAVEFORM_CASES],
    )
    def test_each_shape_translates_to_matching_op(
        self, builder, hw, pulse_kwargs, expected_op_type
    ):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Pulse(ch, **pulse_kwargs))
        imp = PurrImporter()
        imp.build(builder)
        assert len(_ops_of_type(imp, expected_op_type)) == 1
        # Every pulse instruction also emits a PulseOp consuming the waveform.
        pulse_ops = _ops_of_type(imp, PulseOp)
        assert len(pulse_ops) == 1
        assert isinstance(pulse_ops[0].waveform.owner, expected_op_type)


class TestPurrImporterDeviceUpdate:
    def test_assigning_frequency_creates_new_frame(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        builder.add(DeviceUpdate(ch, "frequency", 6e9))
        builder.add(PhaseShift(ch, 0.2))
        imp = PurrImporter()
        imp.build(builder)
        # Two CreateFrame ops: the original and the device-update reissue.
        create_frame_ops = _ops_of_type(imp, CreateFrameOp)
        assert len(create_frame_ops) == 2
        # Second create uses the freshly emitted frequency constant.
        new_freq = create_frame_ops[1].frequency.owner
        assert isinstance(new_freq, ConstantOp)
        assert new_freq.value.value.data == pytest.approx(6e9)
        # Subsequent phase shift threads through the new frame.
        shifts = _ops_of_type(imp, PhaseShiftOp)
        assert shifts[1].frame is create_frame_ops[1].result

    def test_variable_frequency_uses_source_variable(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(DeviceUpdate(ch, "frequency", Variable("freq_var")))
        imp = PurrImporter()
        # Pre-bind the source variable to a constant SSA value (inside main).
        const_freq = ConstantOp(FrequencyAttr(7e9))
        imp._current_block.add_op(const_freq)
        imp._current_environment_variables["freq_var"] = const_freq.result
        imp.build(builder)
        creates = _ops_of_type(imp, CreateFrameOp)
        # The new frame's frequency operand is the pre-bound SSA value.
        assert creates[0].frequency is const_freq.result

    def test_unsupported_attribute_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(DeviceUpdate(ch, "scale", 1.0))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported pulse channel attribute"):
            imp.build(builder)

    def test_unsupported_device_raises(self, builder, hw):
        qubit = hw.get_qubit(0)
        builder.add(DeviceUpdate(qubit, "frequency", 1e9))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported device"):
            imp.build(builder)


class TestPurrImporterRepeat:
    def test_single_repeat_opens_and_closes_scf_for(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Repeat(100))
        builder.add(PhaseShift(ch, 0.1))
        builder.add(EndRepeat())
        imp = PurrImporter()
        imp.build(builder)
        for_ops = [op for op in _ops(imp) if isinstance(op, scf.ForOp)]
        assert len(for_ops) == 1
        # The PhaseShift lives inside the loop body.
        body_ops = list(for_ops[0].body.block.ops)
        assert any(isinstance(op, PhaseShiftOp) for op in body_ops)

    def test_sequential_repeats(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Repeat(10))
        builder.add(PhaseShift(ch, 0.1))
        builder.add(EndRepeat())
        builder.add(Repeat(20))
        builder.add(PhaseShift(ch, 0.2))
        builder.add(EndRepeat())
        imp = PurrImporter()
        imp.build(builder)
        for_ops = [op for op in _ops(imp) if isinstance(op, scf.ForOp)]
        assert len(for_ops) == 2

    def test_nested_repeats(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Repeat(10))
        builder.add(Repeat(5))
        builder.add(PhaseShift(ch, 0.1))
        builder.add(EndRepeat())
        builder.add(EndRepeat())
        imp = PurrImporter()
        imp.build(builder)
        outer_for_ops = [op for op in _ops(imp) if isinstance(op, scf.ForOp)]
        assert len(outer_for_ops) == 1
        inner_for_ops = [
            op for op in outer_for_ops[0].body.block.ops if isinstance(op, scf.ForOp)
        ]
        assert len(inner_for_ops) == 1
        # The PhaseShift lives inside the inner loop.
        inner_body_ops = list(inner_for_ops[0].body.block.ops)
        assert any(isinstance(op, PhaseShiftOp) for op in inner_body_ops)


class TestPurrImporterConstHelpers:
    @pytest.mark.parametrize(
        "value, expected_arith_type",
        [(1.5, "f64"), (3, "i32")],
    )
    def test_get_const_without_attr(self, value, expected_arith_type):
        imp = PurrImporter()
        ssa = imp._get_const_or_var_ssa(value)
        owner = ssa.owner
        assert isinstance(owner, ArithConstantOp)
        assert str(owner.result.type) == expected_arith_type

    @pytest.mark.parametrize(
        "attr_cls",
        [TimeAttr, FrequencyAttr, PhaseAttr, AmplitudeAttr],
    )
    def test_literal_const_with_pulse_attr(self, attr_cls):
        imp = PurrImporter()
        ssa = imp._get_const_or_var_ssa(0.5, attr_cls)
        owner = ssa.owner
        assert isinstance(owner, ConstantOp)
        assert isinstance(owner.value, attr_cls)

    def test_variable_resolves_via_source_variable(self):
        imp = PurrImporter()
        const = ConstantOp(TimeAttr(1e-7))
        imp._current_block.add_op(const)
        imp._current_environment_variables["t_var"] = const.result
        ssa = imp._get_const_or_var_ssa(Variable("t_var"), TimeAttr)
        assert ssa is const.result

    def test_invalid_value_type_raises(self):
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported value"):
            imp._get_const_or_var_ssa("not a number")

    def test_invalid_attr_class_raises(self):
        imp = PurrImporter()
        # StringAttr is a real xDSL attribute but not one of the
        # supported Pulse-dialect numeric attribute classes.
        with pytest.raises(ValueError, match="Unsupported type"):
            imp._get_const_or_var_ssa(0.5, StringAttr)

    def test_int_literal_emits_i32_arith_constant(self):
        imp = PurrImporter()
        ssa = imp._get_const_or_var_ssa(42)
        owner = ssa.owner
        assert isinstance(owner, ArithConstantOp)
        assert str(owner.result.type) == "i32"
        assert owner.value.value.data == 42


class TestPurrImporterModuleStructure:
    """End-to-end checks that the produced module is well-formed."""

    def test_empty_builder_produces_main_with_return_only(self, builder):
        imp = PurrImporter()
        module = imp.build(builder)
        [main] = list(module.body.block.ops)
        assert isinstance(main, func.FuncOp)
        assert main.sym_name.data == "main"
        body_ops = list(main.body.block.ops)
        assert len(body_ops) == 1
        assert isinstance(body_ops[0], func.ReturnOp)

    def test_build_terminates_main_with_func_return(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        imp = PurrImporter()
        imp.build(builder)
        assert isinstance(_ops(imp)[-1], func.ReturnOp)

    def test_unknown_instruction_raises(self, builder):
        # A plain object is not a registered QuantumInstruction subtype,
        # so the singledispatch base method should fire.
        imp = PurrImporter()
        with pytest.raises(ValueError, match="not a supported instruction"):
            imp.translate(object())

    def test_unsupported_pulse_shape_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        pulse = Pulse(ch, PulseShapeType.SQUARE, width=80e-9, amp=0.4)
        # Force an unsupported shape to hit the ``_waveform_to_op`` fallback.
        pulse.shape = "not_a_real_shape"
        builder.add(pulse)
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported shape"):
            imp.build(builder)


class TestPurrImporterPulseWithVariables:
    def test_pulse_width_variable_resolves_via_environment(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(
            Pulse(
                ch,
                PulseShapeType.SQUARE,
                width=Variable("w_var"),
                amp=0.4,
            )
        )
        imp = PurrImporter()
        # Pre-bind the variable to an SSA value inside main.
        w_const = ConstantOp(TimeAttr(80e-9))
        imp._current_block.add_op(w_const)
        imp._current_environment_variables["w_var"] = w_const.result
        imp.build(builder)
        [sq] = _ops_of_type(imp, SquareWaveformOp)
        # The waveform's width operand is the pre-bound SSA value -- no
        # new constant was synthesised for it.
        assert sq.width is w_const.result


class TestPurrImporterRepeatThreading:
    """Verify SSA values flow through ``scf.for`` iter-args/results correctly."""

    def test_phase_shift_inside_repeat_threads_iter_arg(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        # PhaseShift outside the loop seeds the env with an outer frame
        # SSA value; the loop must capture it as an iter-arg.
        builder.add(PhaseShift(ch, 0.1))
        builder.add(Repeat(8))
        builder.add(PhaseShift(ch, 0.2))
        builder.add(EndRepeat())
        imp = PurrImporter()
        imp.build(builder)

        outer_shifts = [op for op in _ops(imp) if isinstance(op, PhaseShiftOp)]
        assert len(outer_shifts) == 1
        [for_op] = [op for op in _ops(imp) if isinstance(op, scf.ForOp)]
        # iter-args at enter time == the latest frame SSA value, which
        # is the result of the outer phase shift.
        assert for_op.iter_args[0] is outer_shifts[0].result
        # The body block has one block argument per iter-arg (plus the
        # induction variable).
        assert len(for_op.body.block.args) == 2
        # The yielded value matches the inner phase shift's result.
        inner_shift = next(
            op for op in for_op.body.block.ops if isinstance(op, PhaseShiftOp)
        )
        yield_op = for_op.body.block.last_op
        assert isinstance(yield_op, scf.YieldOp)
        assert list(yield_op.arguments) == [inner_shift.result]
