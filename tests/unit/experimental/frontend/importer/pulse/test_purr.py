# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :class:`PurrImporter` (Purr -> Pulse dialect importer)."""

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, TupleType
from xdsl.interpreters.scf import scf
from xdsl.ir import Operation

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    BlackmanWaveformOp,
    CallKernelOp,
    ConstantOp,
    CreateFrameOp,
    DiscriminateOp,
    EqualiseOp,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    IntegrateOp,
    KernelOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    RoundedSquareWaveformOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinusoidalWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    SynchronizeOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.attributes import (
    RealThresholdPolicyAttr,
    SampledWaveformAttr,
)
from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    MapOp,
    PostSelectOp,
    RecordType,
)
from qat.experimental.frontend.importer.pulse.post_processing import PostSelectionBuilder
from qat.experimental.frontend.importer.pulse.purr import PurrImporter
from qat.experimental.system_data.pulse.post_processing import (
    PostProcessing as PostSelectionData,
)
from qat.ir.instruction_basetypes import AcquireMode
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    CustomPulse,
    Delay,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    Pulse,
    QuantumInstruction,
    Repeat,
    Return,
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


def _ops(module: ModuleOp):
    """Return all operations in a built module, including nested region ops."""

    return list(module.walk())


def _ops_of_type(module: ModuleOp, op_type):
    return [op for op in _ops(module) if isinstance(op, op_type)]


def _record_create_ops(module: Operation) -> list[CreateOp]:
    return [
        op
        for op in _ops_of_type(module, CreateOp)
        if isinstance(op.result.type, RecordType)
    ]


def _record_keys(create_record: CreateOp) -> tuple[str, ...]:
    schema = create_record.result.type.schema
    return tuple(field.key.data for field in schema.fields.data)


def _has_parent_of_type(op: Operation, parent_type: type[Operation]):
    """Recursively walks up the parent chain to see if the operation is contained in a
    function."""

    parent = op.parent_op()
    while parent is not None:
        if isinstance(parent, parent_type):
            return True
        parent = parent.parent_op()
    return False


def _has_function_parent(op: Operation):
    return _has_parent_of_type(op, func.FuncOp)


def _has_kernel_parent(op: Operation):
    return _has_parent_of_type(op, KernelOp)


class TestPurrImporterPhase:
    def test_phase_shift_emits_phase_shift_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 1.3))
        imp = PurrImporter()
        module = imp.build(builder)
        phase_ops = _ops_of_type(module, PhaseShiftOp)
        assert len(phase_ops) == 1
        phase_const = phase_ops[0].phase.owner
        assert isinstance(phase_const, ConstantOp)
        assert phase_const.value.value.data == pytest.approx(1.3)

    def test_phase_set_emits_phase_set_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseSet(ch, 0.75))
        imp = PurrImporter()
        module = imp.build(builder)
        phase_set_ops = _ops_of_type(module, PhaseSetOp)
        assert len(phase_set_ops) == 1
        assert phase_set_ops[0].phase.owner.value.value.data == pytest.approx(0.75)

    def test_phase_reset_emits_phase_set_to_zero(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseReset(ch))
        imp = PurrImporter()
        module = imp.build(builder)
        phase_set_ops = _ops_of_type(module, PhaseSetOp)
        assert len(phase_set_ops) == 1
        assert phase_set_ops[0].phase.owner.value.value.data == pytest.approx(0.0)

    def test_phase_reset_on_multiple_channels(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(PhaseReset([ch0, ch1]))
        imp = PurrImporter()
        module = imp.build(builder)
        assert len(_ops_of_type(module, PhaseSetOp)) == 2


class TestPurrImporterDelayAndSync:
    def test_delay_emits_wait_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Delay(ch, 320e-9))
        imp = PurrImporter()
        module = imp.build(builder)
        wait_ops = _ops_of_type(module, WaitOp)
        assert len(wait_ops) == 1
        assert wait_ops[0].duration.owner.value.value.data == pytest.approx(320e-9)

    def test_synchronize_single_target_emits_no_sync_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Synchronize(ch))
        imp = PurrImporter()
        module = imp.build(builder)
        assert _ops_of_type(module, SynchronizeOp) == []

    def test_synchronize_multi_targets_emits_sync_op(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(Synchronize([ch0, ch1]))
        imp = PurrImporter()
        module = imp.build(builder)
        sync_ops = _ops_of_type(module, SynchronizeOp)
        assert len(sync_ops) == 1
        assert len(sync_ops[0].frames) == 2


class TestPurrImporterFrameTracking:
    def test_first_use_creates_frame(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.5))
        imp = PurrImporter()
        module = imp.build(builder)

        create_frames = _ops_of_type(module, CreateFrameOp)
        assert len(create_frames) == 1
        assert create_frames[0].port.data == ch.physical_channel.full_id()

    def test_frame_reused_across_instructions(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.5))
        builder.add(PhaseShift(ch, 0.25))
        imp = PurrImporter()
        module = imp.build(builder)
        # Only one frame creation; later PhaseShifts consume the latest result.
        assert len(_ops_of_type(module, CreateFrameOp)) == 1
        assert len(_ops_of_type(module, PhaseShiftOp)) == 2

    def test_distinct_channels_create_distinct_frames(self, builder, hw):
        ch0 = hw.get_qubit(0).get_drive_channel()
        ch1 = hw.get_qubit(1).get_drive_channel()
        builder.add(PhaseShift(ch0, 0.1))
        builder.add(PhaseShift(ch1, 0.1))
        imp = PurrImporter()
        module = imp.build(builder)
        assert len(_ops_of_type(module, CreateFrameOp)) == 2

    def test_get_frame_key_uses_partial_id(self, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        assert PurrImporter._frame_key(ch) == ch.partial_id()

    def test_chain_of_phase_shifts_threads_through_frame_results(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        builder.add(PhaseShift(ch, 0.2))
        builder.add(PhaseShift(ch, 0.3))
        imp = PurrImporter()
        module = imp.build(builder)
        create_frame_ops = _ops_of_type(module, CreateFrameOp)
        assert len(create_frame_ops) == 1
        shifts = _ops_of_type(module, PhaseShiftOp)
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
        module = imp.build(builder)
        [sync] = _ops_of_type(module, SynchronizeOp)
        shifts = _ops_of_type(module, PhaseShiftOp)

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
        module = imp.build(builder)
        [shift] = _ops_of_type(module, PhaseShiftOp)
        [wait] = _ops_of_type(module, WaitOp)
        [pulse] = _ops_of_type(module, PulseOp)
        assert wait.frame is shift.result
        assert pulse.frame is wait.result


class TestPurrImporterAcquire:
    def test_acquire(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6))
        imp = PurrImporter()
        module = imp.build(builder)
        acq_ops = _ops_of_type(module, AcquireOp)
        assert len(acq_ops) == 1
        assert acq_ops[0].duration.owner.value.value.data == pytest.approx(1e-6)
        # No waveform constructed when no filter is given.
        assert not any(isinstance(op, SquareWaveformOp) for op in _ops(module))

    def test_acquire_with_filter_sets_weights(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        weights_arr = np.array([0.1] * 1000)
        filt = CustomPulse(ch, weights_arr)
        builder.add(Acquire(ch, time=1e-6, filter=filt))
        imp = PurrImporter()
        module = imp.build(builder)
        acquire_ops = _ops_of_type(module, AcquireOp)
        assert len(acquire_ops) == 1
        assert np.allclose(acquire_ops[0].weights.weights.data, weights_arr)

    def test_acquire_with_integrator_emits_integrate_op(self, builder, hw):
        """When the INTEGRATOR mode is used, an IntegrateOp is emitted consuming the
        acquisition result."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, mode=AcquireMode.INTEGRATOR))
        imp = PurrImporter()
        module = imp.build(builder)
        acq_ops = _ops_of_type(module, AcquireOp)
        assert len(acq_ops) == 1
        int_ops = _ops_of_type(module, IntegrateOp)
        assert len(int_ops) == 1
        assert int_ops[0].acquisition is acq_ops[0].acquisition_result

    def test_acquire_with_scope_raises_not_implemented_error(self, builder, hw):
        """When the SCOPE mode is used, a NotImplementedError is raised."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, mode=AcquireMode.SCOPE))
        imp = PurrImporter()
        with pytest.raises(NotImplementedError, match="Scope mode is not yet supported"):
            imp.build(builder)


class TestPurrImporterPostProcessing:
    """Tests that post-processing is applied to the acquisition, but is hoisted outside of
    the kernel and that the correct post-processing operations are emitted."""

    def test_acquire_with_linear_map_creates_ssa_chain(self, builder, hw):
        """Creates instructions with an acquisition followed by LINEAR_MAP_COMPLEX_TO_REAL
        post-processing, checking the processing chain is as expected."""

        a = 0.254 + 0.1j
        b = 0.454 - 0.2j

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable="measurement"
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                args=[a, b],
            )
        )
        imp = PurrImporter()
        module = imp.build(builder)
        acq_ops = _ops_of_type(module, AcquireOp)
        assert len(acq_ops) == 1
        int_ops = _ops_of_type(module, IntegrateOp)
        assert len(int_ops) == 1
        equalise_ops = _ops_of_type(module, EqualiseOp)
        assert len(equalise_ops) == 1

        assert acq_ops[0].acquisition_result is int_ops[0].acquisition
        assert isinstance(equalise_ops[0].value.owner, ExtractOp)

        map_ops = _ops_of_type(module, MapOp)
        assert len(map_ops) == 1
        map_body_ops = list(map_ops[0].body.block.ops)
        [extract_op] = [op for op in map_body_ops if isinstance(op, ExtractOp)]
        [equalise_op] = [op for op in map_body_ops if isinstance(op, EqualiseOp)]
        assert map_body_ops.index(extract_op) < map_body_ops.index(equalise_op)

        affine = equalise_ops[0].affine_transform
        assert affine.linear_coefficient.data == 0.5 * a
        assert affine.conjugate_coefficient.data == 0.5 * np.conj(a)
        assert affine.translation.data == np.real(b) + 0j

        # The linear matrix should map any complex number onto a real number
        linear = affine.linear_matrix
        assert linear[0, 0] == np.real(a)
        assert linear[0, 1] == -np.imag(a)
        assert linear[1, 1] == linear[1, 0] == 0.0

        # The translation should be a real translation
        translation = affine.translation_vector
        assert translation[0] == np.real(b)
        assert translation[1] == 0.0

        assert _has_function_parent(equalise_ops[0])

    def test_acquire_with_discriminate_ssa_chain(self, builder, hw):
        """Creates instructions with an acquisition followed by DISCRIMINATE post-
        processing, checking the processing chain is as expected."""

        threshold = 0.5

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable="measurement"
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.DISCRIMINATE,
                args=[threshold],
            )
        )
        imp = PurrImporter()
        module = imp.build(builder)
        acq_ops = _ops_of_type(module, AcquireOp)
        assert len(acq_ops) == 1
        int_ops = _ops_of_type(module, IntegrateOp)
        assert len(int_ops) == 1
        discriminate_ops = _ops_of_type(module, DiscriminateOp)
        assert len(discriminate_ops) == 1

        assert acq_ops[0].acquisition_result is int_ops[0].acquisition
        assert discriminate_ops[0].value.owner.name == "results.extract"

        map_ops = _ops_of_type(module, MapOp)
        assert len(map_ops) == 1
        map_body_ops = list(map_ops[0].body.block.ops)
        [extract_op] = [op for op in map_body_ops if isinstance(op, ExtractOp)]
        [discriminate_op] = [op for op in map_body_ops if isinstance(op, DiscriminateOp)]
        assert map_body_ops.index(extract_op) < map_body_ops.index(discriminate_op)

        policy = discriminate_ops[0].policy
        assert isinstance(policy, RealThresholdPolicyAttr)
        assert policy.threshold.data == threshold
        assert discriminate_ops[0].result.type.state_range == (0, 1)

    def test_post_processing_without_prior_acquire_raises(self, builder, hw):
        """Post-processing should fail clearly when no prior acquire result exists."""

        ch = hw.get_qubit(0).get_acquire_channel()
        acquire = Acquire(ch, time=1e-6, output_variable="measurement")
        builder.add(
            PostProcessing(
                acquire,
                process=PostProcessType.DISCRIMINATE,
                args=[0.5],
            )
        )
        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="no prior acquisition found in the environment",
        ):
            imp.build(builder)

    def test_linear_map_complex_to_real_with_missing_args_raises(self, builder, hw):
        """LINEAR_MAP_COMPLEX_TO_REAL requires exactly two arguments."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable="measurement"
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                args=[],
            )
        )

        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="LINEAR_MAP_COMPLEX_TO_REAL expects 2 arguments",
        ):
            imp.build(builder)

    def test_discriminate_with_missing_args_raises(self, builder, hw):
        """DISCRIMINATE requires exactly one threshold argument."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable="measurement"
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.DISCRIMINATE,
                args=[],
            )
        )

        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="DISCRIMINATE expects 1 argument",
        ):
            imp.build(builder)

    def test_linear_map_complex_to_real_with_too_many_args_raises(self, builder, hw):
        """LINEAR_MAP_COMPLEX_TO_REAL rejects more than two arguments."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable="measurement"
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                args=[0.1 + 0.2j, 0.3 + 0.4j, 0.5 + 0.6j],
            )
        )

        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="LINEAR_MAP_COMPLEX_TO_REAL expects 2 arguments",
        ):
            imp.build(builder)

    def test_discriminate_with_too_many_args_raises(self, builder, hw):
        """DISCRIMINATE rejects more than one argument."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable="measurement"
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.DISCRIMINATE,
                args=[0.5, 1.0],
            )
        )

        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="DISCRIMINATE expects 1 argument",
        ):
            imp.build(builder)

    def test_post_processing_on_non_integrated_acquire_raises(self, builder, hw):
        """Post-processing (EqualiseOp/DiscriminateOp) requires IQResultType operand.

        This test verifies the defensive check that ensures integrated acquisitions are
        used. It builds a RAW acquire so the importer stores a non-integrated
        AcquisitionType and then verifies the importer raises a clear error.
        """

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(ch, time=1e-6, mode=AcquireMode.RAW, output_variable="measurement")
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.DISCRIMINATE,
                args=[0.5],
            )
        )

        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="Post-processing expects an IQResultType.*Ensure the acquire has mode INTEGRATOR",
        ):
            imp.build(builder)


class TestPurrImporterReturns:
    """Tests return instructions with the PurrImporter.

    Tests that only those values are used in the returned record in the map operation, and
    if none is given, it returns all.
    """

    def test_no_return_returns_all(self, builder, hw):
        """Tests that no return instruction returns all variables in the record."""
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        imp = PurrImporter()
        module = imp.build(builder)
        map_ops = _ops_of_type(module, MapOp)
        assert len(map_ops) == 1

        [create_record] = _record_create_ops(map_ops[0])
        assert _record_keys(create_record) == (
            "measurement1",
            "measurement2",
        )
        assert isinstance(create_record.values[0].owner, ExtractOp)
        assert isinstance(create_record.values[1].owner, ExtractOp)
        assert create_record.values[0].owner.key.data == "measurement1"
        assert create_record.values[1].owner.key.data == "measurement2"

    def test_return_only_specified_variables(self, builder, hw):
        """Tests that only the specified variables are returned in the record."""
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement3"))
        builder.add(Return(["measurement1", "measurement3"]))
        imp = PurrImporter()
        module = imp.build(builder)
        map_ops = _ops_of_type(module, MapOp)
        assert len(map_ops) == 1

        [create_record] = _record_create_ops(map_ops[0])
        assert _record_keys(create_record) == (
            "measurement1",
            "measurement3",
        )
        assert isinstance(create_record.values[0].owner, ExtractOp)
        assert isinstance(create_record.values[1].owner, ExtractOp)
        assert create_record.values[0].owner.key.data == "measurement1"
        assert create_record.values[1].owner.key.data == "measurement3"

    def test_return_with_unknown_variable_raises(self, builder, hw):
        """Tests that a return instruction with an unknown variable raises a ValueError."""
        builder.add(Return(["unknown_var"]))
        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="Return variables must be a subset of the post-processing results.",
        ):
            imp.build(builder)


class TestPurrImporterAssign:
    """Tests assign instructions with the PurrImporter, which is used to move values into a
    list or to a new identifier."""

    def test_assign_with_list_of_variables(self, builder, hw):
        """Tests that an assign instruction with a list of variables creates a record with
        the correct keys and values."""
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        builder.add(Assign("my_list", ["measurement1", Variable("measurement2")]))
        imp = PurrImporter()
        module = imp.build(builder)
        map_ops = _ops_of_type(module, MapOp)
        assert len(map_ops) == 1

        [create_record] = _record_create_ops(map_ops[0])
        keys = _record_keys(create_record)
        assert "my_list" in keys
        my_list_index = keys.index("my_list")
        my_list_value = create_record.values[my_list_index]
        assert isinstance(my_list_value.owner, CreateOp)
        assert isinstance(my_list_value.owner.result.type, TupleType)
        assert len(my_list_value.owner.values) == 2
        assert isinstance(my_list_value.owner.values[0].owner, ExtractOp)
        assert my_list_value.owner.values[0].owner.key.data == "measurement1"
        assert isinstance(my_list_value.owner.values[1].owner, ExtractOp)
        assert my_list_value.owner.values[1].owner.key.data == "measurement2"

    def test_assign_with_scalar_string_aliases_value(self, builder, hw):
        """Scalar string assign should alias an existing SSA value under a new key."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        builder.add(Assign("measurement_alias", "measurement2"))

        imp = PurrImporter()
        module = imp.build(builder)
        [map_op] = _ops_of_type(module, MapOp)
        [create_record] = _record_create_ops(map_op)

        keys = _record_keys(create_record)
        assert "measurement2" in keys
        assert "measurement_alias" in keys

        source_idx = keys.index("measurement2")
        alias_idx = keys.index("measurement_alias")
        assert create_record.values[alias_idx] is create_record.values[source_idx]
        assert isinstance(create_record.values[alias_idx].owner, ExtractOp)
        assert create_record.values[alias_idx].owner.key.data == "measurement2"

    def test_assign_with_scalar_variable_aliases_value(self, builder, hw):
        """Scalar variable assign should alias an existing SSA value under a new key."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        builder.add(Assign("measurement_alias", Variable("measurement2")))

        imp = PurrImporter()
        module = imp.build(builder)
        [map_op] = _ops_of_type(module, MapOp)
        [create_record] = _record_create_ops(map_op)

        keys = _record_keys(create_record)
        assert "measurement2" in keys
        assert "measurement_alias" in keys

        source_idx = keys.index("measurement2")
        alias_idx = keys.index("measurement_alias")
        assert create_record.values[alias_idx] is create_record.values[source_idx]
        assert isinstance(create_record.values[alias_idx].owner, ExtractOp)
        assert create_record.values[alias_idx].owner.key.data == "measurement2"

    def test_assign_list_with_non_variable_raises(self, builder, hw):
        """Tests that an assign instruction with a list containing a non-variable raises a
        ValueError."""
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        builder.add(Assign("my_list", ["measurement1", 5]))
        imp = PurrImporter()
        with pytest.raises(
            ValueError, match="Cannot assign value 5 in assign instruction."
        ):
            imp.build(builder)

    def test_assign_with_unknown_variable_raises(self, builder, hw):
        """Tests that an assign instruction with an unknown variable raises a ValueError."""
        builder.add(Assign("my_var", ["unknown_var"]))
        imp = PurrImporter()
        with pytest.raises(
            ValueError,
            match="Assign value unknown_var not found in post-processing results.",
        ):
            imp.build(builder)


class TestPurrImporterUnsupportedInstructions:
    @pytest.mark.parametrize(
        "type_", [PostProcessType.MEAN, PostProcessType.DOWN_CONVERT, PostProcessType.MUL]
    )
    def test_post_processing_raises(self, builder, hw, type_):
        ch = hw.get_qubit(0).get_acquire_channel()
        acq = Acquire(ch, time=1e-6, mode=AcquireMode.INTEGRATOR)
        builder.add(acq)

        builder.add(PostProcessing(acq, process=type_))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported post-processing type"):
            imp.build(builder)

    def test_sweep_raises(self, builder):
        builder.add(Sweep())
        imp = PurrImporter()
        with pytest.raises(NotImplementedError, match="Sweep instructions"):
            imp.build(builder)

    def test_end_sweep_raises(self, builder):
        builder.add(EndSweep())
        imp = PurrImporter()
        with pytest.raises(ValueError, match="not a supported instruction"):
            imp.build(builder)

    def test_acquire_with_non_custom_pulse_weights(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        filter_ = Pulse(ch, PulseShapeType.SQUARE, width=1e-6, amp=0.5)
        builder.add(Acquire(ch, time=1e-6, filter=filter_))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Acquire filter must be a CustomPulse"):
            imp.build(builder)

    def test_variable_raises_value_error(self, builder):
        builder.add(Variable("my_var"))
        imp = PurrImporter()
        with pytest.raises(
            ValueError, match="Standalone variable instructions are not supported"
        ):
            imp.build(builder)

    def test_instruction_with_variable_operand_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(PhaseShift(ch, Variable("measurement")))
        imp = PurrImporter()
        with pytest.raises(
            NotImplementedError, match="Variable resolution is not yet supported."
        ):
            imp.build(builder)

    def test_instruction_with_non_numeric_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(PhaseShift(ch, "not_a_number"))
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported value type"):
            imp.build(builder)


WAVEFORM_CASES = [
    (
        {"shape": PulseShapeType.SQUARE, "width": 80e-9, "amp": 0.5},
        SquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.GAUSSIAN,
            "width": 80e-9,
            "amp": 0.5,
            "rise": 16e-9,
            "drag": 0.15,
        },
        GaussianWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SOFT_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "rise": 1e-9,
            "drag": 0.2,
        },
        SoftSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SOFTER_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "rise": 1e-9,
            "drag": 0.1,
        },
        SoftSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.EXTRA_SOFT_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "rise": 1e-9,
            "drag": 0.05,
        },
        SoftSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.GAUSSIAN_SQUARE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "square_width": 40e-9,
            "zero_at_edges": 1,
            "drag": 0.12,
        },
        GaussianSquareWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SOFTER_GAUSSIAN,
            "width": 80e-9,
            "amp": 0.5,
            "rise": 16e-9,
            "drag": 0.08,
        },
        GaussianWaveformOp,
    ),
    (
        {"shape": PulseShapeType.BLACKMAN, "width": 80e-9, "amp": 0.5, "drag": 0.18},
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
            "drag": 0.22,
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
        GaussianWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.GAUSSIAN_ZERO_EDGE,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "zero_at_edges": 1,
            "drag": 0.11,
        },
        GaussianWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SECH,
            "width": 80e-9,
            "amp": 0.5,
            "std_dev": 8e-9,
            "drag": 0.09,
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
            "drag": 0.14,
        },
        SinusoidalWaveformOp,
    ),
    (
        {
            "shape": PulseShapeType.SIN,
            "width": 80e-9,
            "amp": 0.5,
            "frequency": 5e9,
            "internal_phase": 0.5,
            "drag": 0.19,
        },
        SinusoidalWaveformOp,
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
        module = imp.build(builder)
        assert len(_ops_of_type(module, expected_op_type)) == 1
        # Every pulse instruction also emits a PulseOp consuming the waveform.
        pulse_ops = _ops_of_type(module, PulseOp)
        assert len(pulse_ops) == 1
        assert isinstance(pulse_ops[0].waveform.owner, expected_op_type)

        # Verify DRAG coefficient if present in kwargs
        expected_drag = pulse_kwargs.get("drag")
        if expected_drag is not None:
            waveform_op = pulse_ops[0].waveform.owner
            assert len(waveform_op.drag_coefficients) == 1
            assert waveform_op.drag_coefficients[0].owner.value.value.data == pytest.approx(
                expected_drag
            )

    def test_custom_pulse_emits_pulse_op(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(CustomPulse(ch, np.zeros(8)))
        imp = PurrImporter()
        module = imp.build(builder)

        [pulse_op] = _ops_of_type(module, PulseOp)
        assert isinstance(pulse_op.waveform.owner, ConstantOp)
        assert isinstance(pulse_op.waveform.owner.value, SampledWaveformAttr)

    def test_custom_pulse_uses_samples_and_duration(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        samples = [0.25 + 0.5j, 0.5, -0.75j, -0.125]
        builder.add(CustomPulse(ch, samples))

        imp = PurrImporter()
        module = imp.build(builder)

        [pulse_op] = _ops_of_type(module, PulseOp)
        waveform_owner = pulse_op.waveform.owner
        assert isinstance(waveform_owner, ConstantOp)
        assert isinstance(waveform_owner.value, SampledWaveformAttr)

        sampled_attr = waveform_owner.value
        assert np.allclose(
            sampled_attr.samples.data, np.asarray(samples, dtype=np.complex128)
        )
        assert sampled_attr.width.literal_value == pytest.approx(
            ch.sample_time * len(samples)
        )
        assert sampled_attr.sample_time.literal_value == pytest.approx(ch.sample_time)

    def test_gaussian_drag_maps_beta_to_drag_coefficient(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(
            Pulse(
                ch,
                shape=PulseShapeType.GAUSSIAN_DRAG,
                width=80e-9,
                amp=0.5,
                std_dev=8e-9,
                beta=0.1,
                zero_at_edges=0,
            )
        )

        module = PurrImporter().build(builder)
        [waveform_op] = _ops_of_type(module, GaussianWaveformOp)
        assert len(waveform_op.drag_coefficients) == 1
        assert waveform_op.drag_coefficients[0].owner.value.value.data == pytest.approx(0.1)


class TestPurrImporterDeviceUpdate:
    def test_assigning_frequency_changes_frequency_value(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        builder.add(DeviceUpdate(ch, "frequency", 6e9))
        builder.add(PhaseShift(ch, 0.2))
        imp = PurrImporter()
        module = imp.build(builder)
        # Device assign changes the frequency of the pulse channel in PuRR, check for
        # correspondence here
        create_frame_ops = _ops_of_type(module, CreateFrameOp)
        assert len(create_frame_ops) == 1
        # Second create uses the freshly emitted frequency constant.
        new_freq = create_frame_ops[0].frequency.owner
        assert isinstance(new_freq, ConstantOp)
        assert new_freq.value.value.data == pytest.approx(6e9)
        # Subsequent phase shift threads through the new frame.
        shifts = _ops_of_type(module, PhaseShiftOp)
        assert shifts[0].frame is create_frame_ops[0].result

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

    def test_device_update_with_variable_frequency_raises_not_implemented_error(
        self, builder, hw
    ):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(DeviceUpdate(ch, "frequency", Variable("my_var")))
        imp = PurrImporter()
        with pytest.raises(
            NotImplementedError,
            match="Variable resolution is not yet supported in the device update.",
        ):
            imp.build(builder)

    def test_multiple_device_updates_on_same_channel_raises_value_error(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(DeviceUpdate(ch, "frequency", 6e9))
        builder.add(DeviceUpdate(ch, "frequency", 7e9))
        imp = PurrImporter()
        with pytest.raises(
            ValueError, match="Multiple frequency updates for pulse channel"
        ):
            imp.build(builder)

    def test_translate_unsupported_instruction_raises_value_error(self, builder, hw):
        class UnsupportedQuantumInstruction(QuantumInstruction):
            pass

        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(UnsupportedQuantumInstruction(ch))

        imp = PurrImporter()
        with pytest.raises(ValueError, match="not a supported instruction"):
            imp.build(builder)


class TestPurrImporterRepeat:
    def test_single_repeat_opens_and_closes_scf_for(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Repeat(100))
        builder.add(PhaseShift(ch, 0.1))
        builder.add(EndRepeat())
        imp = PurrImporter()
        module = imp.build(builder)
        for_ops = [op for op in _ops(module) if isinstance(op, scf.ForOp)]
        assert len(for_ops) == 1
        # The PhaseShift lives inside the loop body.
        body_ops = list(for_ops[0].body.block.ops)
        assert any(isinstance(op, PhaseShiftOp) for op in body_ops)

    def test_multiple_repeats_raises_value_error(self, builder):
        builder.add(Repeat(10))
        builder.add(Repeat(20))
        builder.add(EndRepeat())
        builder.add(EndRepeat())
        imp = PurrImporter()
        with pytest.raises(
            ValueError, match="Multiple repeat instructions are not supported."
        ):
            imp.build(builder)


class TestPurrImporterModuleStructure:
    """End-to-end checks that the produced module is well-formed."""

    def test_main_calls_kernel_then_maps_then_returns(self, builder, hw):
        """Checks that main wires kernel execution through a results map and return."""

        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))

        imp = PurrImporter()
        module = imp.build(builder)

        main = next(
            op
            for op in module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == "main"
        )
        body_ops = list(main.body.block.ops)

        assert len(body_ops) == 3
        call_op, map_op, return_op = body_ops
        assert isinstance(call_op, CallKernelOp)
        assert isinstance(map_op, MapOp)
        assert isinstance(return_op, func.ReturnOp)

        assert map_op.value is call_op.result[0]
        assert main.function_type.outputs.data[0] == map_op.result.type
        assert tuple(return_op.operands) == (map_op.result,)

    def test_main_signature_tracks_map_output_schema(self, builder, hw):
        """Main return type should follow the mapped collection, not the kernel output."""

        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement1"))
        builder.add(Acquire(ch, time=1e-6, output_variable="measurement2"))
        builder.add(Return(["measurement1"]))

        imp = PurrImporter()
        module = imp.build(builder)

        main = next(
            op
            for op in module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == "main"
        )
        call_op, map_op, _return_op = list(main.body.block.ops)

        assert isinstance(call_op, CallKernelOp)
        assert isinstance(map_op, MapOp)
        assert call_op.result[0].type != map_op.result.type
        assert main.function_type.outputs.data[0] == map_op.result.type

    def test_quantum_ops_are_nested_in_kernel_loop(self, builder, hw):
        """Checks that repeated quantum ops are emitted in the kernel loop, not in main."""

        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(Repeat(5))
        builder.add(PhaseShift(ch, 0.1))
        builder.add(EndRepeat())

        imp = PurrImporter()
        module = imp.build(builder)

        [for_op] = [op for op in _ops(module) if isinstance(op, scf.ForOp)]
        loop_body_ops = list(for_op.body.block.ops)
        phase_ops_in_loop = [op for op in loop_body_ops if isinstance(op, PhaseShiftOp)]

        assert len(phase_ops_in_loop) == 1
        assert _has_kernel_parent(phase_ops_in_loop[0])
        assert not _has_function_parent(phase_ops_in_loop[0])

        main = next(
            op
            for op in module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == "main"
        )
        assert _ops_of_type(main, PhaseShiftOp) == []
        assert len(_ops_of_type(main, CallKernelOp)) == 1
        assert len(_ops_of_type(main, MapOp)) == 1
        assert len(_ops_of_type(main, func.ReturnOp)) == 1

    def test_empty_builder_produces_main_with_return_only(self, builder):
        imp = PurrImporter()
        module = imp.build(builder)
        main = next(
            op
            for op in module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == "main"
        )
        assert isinstance(main, func.FuncOp)
        assert main.sym_name.data == "main"
        body_ops = list(main.body.block.ops)
        assert len(body_ops) == 3
        assert isinstance(body_ops[-1], func.ReturnOp)

    def test_build_terminates_main_with_func_return(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        builder.add(PhaseShift(ch, 0.1))
        imp = PurrImporter()
        module = imp.build(builder)
        main = next(
            op
            for op in module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == "main"
        )
        assert isinstance(list(main.body.block.ops)[-1], func.ReturnOp)

    def test_unknown_instruction_raises(self, builder):
        # A plain object is not a registered QuantumInstruction subtype,
        # so the singledispatch base method should fire.
        imp = PurrImporter()
        with pytest.raises(ValueError, match="not a supported instruction"):
            imp.translate(object(), None)

    def test_unsupported_pulse_shape_raises(self, builder, hw):
        ch = hw.get_qubit(0).get_drive_channel()
        pulse = Pulse(ch, PulseShapeType.SQUARE, width=80e-9, amp=0.4)
        # Force an unsupported shape to hit the ``_waveform_to_op`` fallback.
        pulse.shape = "not_a_real_shape"
        builder.add(pulse)
        imp = PurrImporter()
        with pytest.raises(ValueError, match="Unsupported shape"):
            imp.build(builder)


class TestPurrImporterPostSelection:
    """Tests that post-selection is correctly applied when a PostSelectionBuilder is
    provided to the importer."""

    def _make_builder_with_acquire(
        self, hw, output_variable: str = "meas0"
    ) -> QuantumInstructionBuilder:
        """Returns a builder with a single INTEGRATOR acquire on qubit 0."""
        builder = QuantumInstructionBuilder(hw)
        ch = hw.get_qubit(0).get_acquire_channel()
        builder.add(
            Acquire(
                ch, time=1e-6, mode=AcquireMode.INTEGRATOR, output_variable=output_variable
            )
        )
        builder.add(
            PostProcessing(
                builder.instructions[-1],
                process=PostProcessType.DISCRIMINATE,
                args=[0.0],
            )
        )
        return builder

    def _post_selection_builder(
        self,
        hw,
        enabled: bool = True,
        disallowed_states: frozenset[int] = frozenset({-1}),
    ) -> PostSelectionBuilder:
        ch = hw.get_qubit(0).get_acquire_channel()
        pp = PostSelectionData(
            channel_to_disallowed_states={ch.partial_id(): set(disallowed_states)},
            known_channel_ids=frozenset({ch.partial_id()}),
        )
        return PostSelectionBuilder(pp, enabled=enabled)

    def _main_ops(self, module: ModuleOp) -> list:
        main = next(
            op
            for op in module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == "main"
        )
        return list(main.body.block.ops)

    def test_no_post_selection_builder_emits_no_post_select_op(self, hw):
        builder = self._make_builder_with_acquire(hw)
        module = PurrImporter().build(builder)
        assert not any(isinstance(op, PostSelectOp) for op in _ops(module))

    def test_disabled_builder_emits_no_post_select_op(self, hw):
        builder = self._make_builder_with_acquire(hw)
        psb = self._post_selection_builder(hw, enabled=False)
        module = PurrImporter(post_selection_builder=psb).build(builder)
        assert not any(isinstance(op, PostSelectOp) for op in _ops(module))

    def test_no_disallowed_states_emits_no_post_select_op(self, hw):
        builder = self._make_builder_with_acquire(hw)
        psb = self._post_selection_builder(hw, disallowed_states=frozenset())
        module = PurrImporter(post_selection_builder=psb).build(builder)
        assert not any(isinstance(op, PostSelectOp) for op in _ops(module))

    def test_enabled_with_disallowed_states_emits_post_select_op(self, hw):
        builder = self._make_builder_with_acquire(hw)
        psb = self._post_selection_builder(hw, enabled=True)
        module = PurrImporter(post_selection_builder=psb).build(builder)
        assert any(isinstance(op, PostSelectOp) for op in _ops(module))

    def test_post_select_op_appears_after_map_op_in_main(self, hw):
        builder = self._make_builder_with_acquire(hw)
        psb = self._post_selection_builder(hw, enabled=True)
        module = PurrImporter(post_selection_builder=psb).build(builder)
        main_ops = self._main_ops(module)
        op_types = [type(op) for op in main_ops]
        assert MapOp in op_types
        assert PostSelectOp in op_types
        assert op_types.index(MapOp) < op_types.index(PostSelectOp)

    def test_post_select_op_predicate_key_matches_output_variable(self, hw):
        builder = self._make_builder_with_acquire(hw, output_variable="meas0")
        psb = self._post_selection_builder(hw, enabled=True)
        module = PurrImporter(post_selection_builder=psb).build(builder)
        post_select_ops = _ops_of_type(module, PostSelectOp)
        assert len(post_select_ops) == 1
        predicate_keys = [p.key.data for p in post_select_ops[0].predicates.data]
        assert "meas0" in predicate_keys

    def test_post_select_op_predicate_disallowed_states_match(self, hw):
        builder = self._make_builder_with_acquire(hw)
        psb = self._post_selection_builder(hw, disallowed_states=frozenset({-1, -2}))
        module = PurrImporter(post_selection_builder=psb).build(builder)
        post_select_ops = _ops_of_type(module, PostSelectOp)
        assert len(post_select_ops) == 1
        predicate = post_select_ops[0].predicates.data[0]
        assert {s.data for s in predicate.disallowed_values.data} == {-1, -2}

    def test_return_op_uses_post_select_result_when_present(self, hw):
        builder = self._make_builder_with_acquire(hw)
        psb = self._post_selection_builder(hw, enabled=True)
        module = PurrImporter(post_selection_builder=psb).build(builder)
        main_ops = self._main_ops(module)
        return_op = next(op for op in main_ops if isinstance(op, func.ReturnOp))
        post_select_op = next(op for op in main_ops if isinstance(op, PostSelectOp))
        assert return_op.operands[0] is post_select_op.result

    def test_unrelated_channel_produces_no_post_select_op(self, hw):
        """A PostSelectionBuilder whose disallowed states are for a different channel than
        the one acquired should not emit a PostSelectOp.

        A warning is expected because the acquired channel ID is not known to the post-
        processing data.
        """
        builder = self._make_builder_with_acquire(hw)
        pp = PostSelectionData(
            channel_to_disallowed_states={"some_other_channel_id": {-1}},
            known_channel_ids=frozenset({"some_other_channel_id"}),
        )
        psb = PostSelectionBuilder(pp, enabled=True)
        with pytest.warns(UserWarning, match="Unmatched channels"):
            module = PurrImporter(post_selection_builder=psb).build(builder)
        assert not any(isinstance(op, PostSelectOp) for op in _ops(module))
