# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random
from enum import Enum

import numpy as np
import pytest

from qat.backend.passes.validation import PydReturnSanitisationValidation
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import PhaseReset as PydPhaseReset
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.instructions import Return as PydReturn
from qat.ir.measure import Acquire as PydAcquire
from qat.ir.measure import AcquireMode
from qat.ir.measure import MeasureBlock as PydMeasureBlock
from qat.ir.measure import PostProcessing as PydPostProcessing
from qat.ir.measure import PostProcessType, ProcessAxis
from qat.ir.waveforms import GaussianWaveform, SquareWaveform
from qat.middleend.passes.analysis import ActiveChannelResults
from qat.middleend.passes.legacy.transform import LoopCount, RepeatTranslation
from qat.middleend.passes.transform import (
    AcquireSanitisation,
    EndOfTaskResetSanitisation,
    EvaluatePulses,
    InactivePulseChannelSanitisation,
    InitialPhaseResetSanitisation,
    InstructionGranularitySanitisation,
    InstructionLengthSanitisation,
    MeasurePhaseResetSanitisation,
    PhaseOptimisation,
    PydPhaseOptimisation,
    PydPostProcessingSanitisation,
    PydReturnSanitisation,
    ResetsToDelays,
    SynchronizeTask,
)
from qat.model.loaders.converted import EchoModelLoader as PydEchoModelLoader
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    CustomPulse,
    Delay,
    EndRepeat,
    GreaterThan,
    Instruction,
    Jump,
    Label,
    MeasurePulse,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    Plus,
    Pulse,
    PulseShapeType,
    Reset,
    Synchronize,
    Variable,
)
from qat.utils.hardware_model import generate_hw_model

from tests.unit.utils.pulses import pulse_attributes


class TestPhaseOptimisation:
    hw = EchoModelLoader(qubit_count=4).load()

    def test_merged_identical_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        phase_reset = PhaseReset(self.hw.qubits)
        builder.add(phase_reset)
        builder.add(phase_reset)
        builder.delay(phase_reset.quantum_targets, 1e-3)  # to stop them being deleted
        assert len(builder.instructions) == 3

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())
        for inst in builder.instructions[:-1]:
            assert isinstance(inst, PhaseSet)
        channels = set([inst.channel for inst in builder.instructions[:-1]])
        assert len(channels) == len(builder.instructions) - 1

    def test_empty_constructor(self):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.add(PhaseShift(qubit.get_drive_channel(), phase))
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        if pulse_enabled:
            assert len(builder_optimised.instructions) == len(hw.qubits)
        else:
            assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.phase_shift(qubit, phase)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
        else:
            assert len(phase_shifts) == 0

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        qubits = hw.qubits
        for qubit in qubits:
            builder.phase_shift(qubit, phase)

        random.shuffle(hw.qubits)
        for qubit in qubits:
            builder.phase_shift(qubit, phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = EchoModelLoader(2).load()
        builder = hw.create_builder()
        qubits = list(hw.qubits)

        for qubit in qubits:
            builder.phase_shift(qubit, 0.5)
            builder.reset(qubit)

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shifts) == 0

    def test_reset_and_shift_become_set(self):
        hw = EchoModelLoader(2).load()
        ir = hw.create_builder()
        chan = hw.qubits[0].get_drive_channel()

        ir.add(PhaseReset(chan))
        ir.add(PhaseShift(chan, np.pi / 4))
        ir.pulse(chan, shape=PulseShapeType.SQUARE, width=80e-9)

        ir = PhaseOptimisation().run(ir, ResultManager(), MetricsManager())

        assert len(ir.instructions) == 2
        assert isinstance(ir.instructions[0], PhaseSet)
        assert np.isclose(ir.instructions[0].phase, np.pi / 4)

    def test_phaseset_does_not_commute_through_delay(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        chan = self.hw.qubits[0].get_drive_channel()
        builder.add(PhaseReset(chan))
        builder.delay(chan, 1e-3)
        builder.add(PhaseSet(chan, np.pi / 2))
        builder.delay(chan, 1e-3)
        builder.add(PhaseShift(chan, np.pi / 4))
        builder.delay(chan, 1e-3)

        builder = PhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Delay,
            Delay,
        ]

    def test_phaseset_does_not_commute_through_sync(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        chan1 = self.hw.qubits[0].get_drive_channel()
        chan2 = self.hw.qubits[0].get_measure_channel()
        builder.add(PhaseReset(chan1))
        builder.add(PhaseReset(chan2))
        builder.delay(chan1, 1e-3)
        builder.synchronize([chan1, chan2])
        builder.add(PhaseSet(chan1, np.pi / 2))
        builder.add(PhaseSet(chan2, np.pi / 2))
        builder.delay(chan2, 1e-3)
        builder.synchronize([chan1, chan2])
        builder.add(PhaseShift(chan1, np.pi / 4))
        builder.add(PhaseShift(chan2, np.pi / 4))

        builder = PhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Synchronize,
            PhaseSet,
            Delay,
            PhaseSet,
            Synchronize,
        ]
        assert [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, PhaseSet)
        ] == [chan1, chan2, chan2, chan1]


class TestPydPhaseOptimisation:
    hw = PydEchoModelLoader(8).load()

    def test_empty_constructor(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.add(PydPhaseShift(targets=qubit.drive_pulse_channel.uuid, phase=phase))
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        if pulse_enabled:
            assert builder_optimised.number_of_instructions == self.hw.number_of_qubits
        else:
            assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.phase_shift(target=qubit, theta=phase)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PydPhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == self.hw.number_of_qubits
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubits = list(self.hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase)

        random.shuffle(qubits)
        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=SquareWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PydPhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == self.hw.number_of_qubits
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubits = list(self.hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=0.5)
            builder.reset(qubit)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PydPhaseShift)
        ]
        assert len(phase_shifts) == 0

    def test_merged_identical_phase_resets(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        targets = [pulse_ch.uuid for pulse_ch in qubit.all_pulse_channels]

        phase_reset = PydPhaseReset(targets=targets)
        builder.add(phase_reset)
        builder.add(phase_reset)
        assert builder.number_of_instructions == 2

        PydPhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )
        # The two phase resets should be merged to one.
        assert builder.number_of_instructions == 1
        # assert set(builder.instructions[0].quantum_targets) == set(phase_reset.quantum_targets)

    def test_merged_phase_resets(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)

        targets_q1 = set(
            [pulse_ch.uuid for pulse_ch in self.hw.qubit_with_index(0).all_pulse_channels]
        )
        targets_q2 = set(
            [pulse_ch.uuid for pulse_ch in self.hw.qubit_with_index(1).all_pulse_channels]
        )
        builder.add(PydPhaseReset(targets=targets_q1))
        builder.add(PydPhaseReset(targets=targets_q2))

        PydPhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )
        # The two phase resets should be merged to one, and the targets of both phase resets should be merged.
        assert builder.number_of_instructions == 1
        merged_targets = targets_q1 | targets_q2

        assert builder.instructions[0].targets == merged_targets


class TestPydPostProcessingSanitisation:
    hw = PydEchoModelLoader(32).load()

    def test_meas_acq_with_pp(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

    @pytest.mark.parametrize(
        "acq_mode,pp_type,pp_axes",
        [
            (AcquireMode.SCOPE, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]),
            (AcquireMode.INTEGRATOR, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]),
            (AcquireMode.INTEGRATOR, PostProcessType.MEAN, [ProcessAxis.TIME]),
        ],
    )
    def test_invalid_acq_pp(self, acq_mode, pp_type, pp_axes):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.measure(targets=qubit, mode=acq_mode, output_variable="test")
        builder.post_processing(
            target=qubit, process_type=pp_type, axes=pp_axes, output_variable="test"
        )
        assert isinstance(builder._ir.tail, PydPostProcessing)

        # Pass should remove the invalid post-processing instruction from the IR.
        assert not PydPostProcessingSanitisation()._valid_pp(acq_mode, builder._ir.tail)

        PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())
        assert not isinstance(builder._ir.tail, PydPostProcessing)

    def test_invalid_raw_acq(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.measure(targets=qubit, mode=AcquireMode.RAW, output_variable="test")

        with pytest.raises(ValueError):
            PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

    def test_mid_circuit_measurement_two_diff_post_processing(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(2)

        # Mid-circuit measurement with some manual (different) post-processing options.
        builder.measure(targets=qubit, mode=AcquireMode.SCOPE)
        assert isinstance(builder._ir.tail, PydMeasureBlock)
        builder.post_processing(
            target=qubit,
            output_variable=builder._ir.tail.output_variables[0],
            process_type=PostProcessType.DOWN_CONVERT,
        )
        builder.X(target=qubit)
        builder.measure(targets=qubit, mode=AcquireMode.INTEGRATOR)
        assert isinstance(builder._ir.tail, PydMeasureBlock)
        builder.post_processing(
            target=qubit,
            output_variable=builder._ir.tail.output_variables[0],
            process_type=PostProcessType.MEAN,
        )

        PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

        # Make sure no instructions get discarded in the post-processing sanitisation for a mid-circuit measurement.
        pp = [instr for instr in builder if isinstance(instr, PydPostProcessing)]
        assert len(pp) == 2
        assert pp[0].output_variable != pp[1].output_variable


class TestPydReturnSanitisation:
    hw = generate_hw_model(8)

    def test_empty_builder(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            PydReturnSanitisationValidation().run(builder, res_mgr)

        PydReturnSanitisation().run(builder, res_mgr)
        PydReturnSanitisationValidation().run(builder, res_mgr)

        return_instr: PydReturn = builder._ir.tail
        assert len(return_instr.variables) == 0

    def test_single_return(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.returns(variables=["test"])
        ref_nr_instructions = builder.number_of_instructions

        res_mgr = ResultManager()
        PydReturnSanitisationValidation().run(builder, res_mgr)
        PydReturnSanitisation().run(builder, res_mgr)

        assert builder.number_of_instructions == ref_nr_instructions
        assert builder.instructions[0].variables == ["test"]

    def test_multiple_returns_squashed(self):
        q0 = self.hw.qubit_with_index(0)
        q1 = self.hw.qubit_with_index(1)

        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.measure_single_shot_z(target=q0, output_variable="out_q0")
        builder.measure_single_shot_z(target=q1, output_variable="out_q1")

        output_vars = [
            instr.output_variable for instr in builder if isinstance(instr, PydAcquire)
        ]
        assert len(output_vars) == 2

        builder.returns(variables=[output_vars[0]])
        builder.returns(variables=[output_vars[1]])

        res_mgr = ResultManager()
        # Two returns in a single IR should raise an error.
        with pytest.raises(ValueError):
            PydReturnSanitisationValidation().run(builder, res_mgr)

        # Compress the two returns to a single return and validate.
        PydReturnSanitisation().run(builder, res_mgr)
        PydReturnSanitisationValidation().run(builder, res_mgr)

        return_instr = builder._ir.tail
        assert isinstance(return_instr, PydReturn)
        for var in return_instr.variables:
            assert var in output_vars


class TestAcquireSanitisation:

    def test_acquire_with_no_delay_is_unchanged(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        acquire_chan = model.qubits[0].get_acquire_channel()
        acquire_block_time = acquire_chan.physical_channel.block_time
        builder = model.create_builder()

        # Make some instructions to test
        builder.acquire(acquire_chan, time=acquire_block_time * 10, delay=0.0)

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Acquire)

    def test_acquire_with_delay_is_decomposed(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        acquire_chan = model.qubits[0].get_acquire_channel()
        acquire_block_time = acquire_chan.physical_channel.block_time
        builder = model.create_builder()

        # Make some instructions to test
        builder.acquire(
            acquire_chan, time=acquire_block_time * 10, delay=acquire_block_time
        )

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 2
        assert isinstance(builder.instructions[0], Delay)
        assert isinstance(builder.instructions[1], Acquire)

    def test_acquire_with_delay_two_chans_is_decomposed(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        for qubit in (0, 1):
            acquire_chan = model.qubits[qubit].get_acquire_channel()
            acquire_block_time = acquire_chan.physical_channel.block_time

            # Make some instructions to test
            builder.acquire(
                acquire_chan, time=acquire_block_time * 10, delay=acquire_block_time
            )

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 4
        assert isinstance(builder.instructions[0], Delay)
        assert isinstance(builder.instructions[1], Acquire)
        assert isinstance(builder.instructions[2], Delay)
        assert isinstance(builder.instructions[3], Acquire)


class TestInstructionGranularitySanitisation:

    def test_instructions_with_correct_timings_are_unchanged(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        drive_chan = model.qubits[0].get_drive_channel()
        acquire_chan = model.qubits[0].get_acquire_channel()
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = 8e-8
        delay_time = np.random.randint(1, 100) * clock_cycle
        builder.delay(drive_chan, delay_time)
        pulse_time = np.random.randint(1, 100) * clock_cycle
        builder.pulse(
            quantum_target=drive_chan, shape=PulseShapeType.SQUARE, width=pulse_time
        )
        acquire_time = np.random.randint(1, 100) * clock_cycle
        builder.acquire(acquire_chan, time=acquire_time, delay=0.0)

        ir = InstructionGranularitySanitisation(clock_cycle).run(builder)

        # compare in units of ns to ensure np.isclose works fine
        assert np.isclose(ir.instructions[0].duration * 1e9, delay_time * 1e9)
        assert np.isclose(ir.instructions[1].duration * 1e9, pulse_time * 1e9)
        assert np.isclose(ir.instructions[2].duration * 1e9, acquire_time * 1e9)

    def test_instructions_are_rounded_up(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        drive_chan = model.qubits[0].get_drive_channel()
        acquire_chan = model.qubits[0].get_acquire_channel()
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = 8e-9
        delay_time = np.random.randint(1, 100) * clock_cycle
        builder.delay(drive_chan, delay_time + np.random.rand() * clock_cycle)
        pulse_time = np.random.randint(1, 100) * clock_cycle
        builder.pulse(
            quantum_target=drive_chan,
            shape=PulseShapeType.SQUARE,
            width=pulse_time + np.random.rand() * clock_cycle,
        )
        acquire_time = np.random.randint(1, 100) * clock_cycle
        builder.acquire(
            acquire_chan,
            time=acquire_time + np.random.rand() * clock_cycle,
            delay=0.0,
        )

        ir = InstructionGranularitySanitisation(clock_cycle).run(builder)

        # compare in units of ns to ensure np.isclose works fine
        assert np.isclose(
            ir.instructions[0].duration * 1e9, (delay_time + clock_cycle) * 1e9
        )
        assert np.isclose(
            ir.instructions[1].duration * 1e9, (pulse_time + clock_cycle) * 1e9
        )
        assert np.isclose(
            ir.instructions[2].duration * 1e9, (acquire_time + clock_cycle) * 1e9
        )

    def test_custom_pulses_with_correct_length_are_unchanged(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        drive_chan = model.qubits[0].get_drive_channel()
        sample_time = drive_chan.physical_channel.sample_time
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = 8e-9
        supersampling = int(np.round(clock_cycle / sample_time, 0))
        num_samples = np.random.randint(1, 100) * supersampling
        samples = [1.0] * num_samples
        builder.add(CustomPulse(drive_chan, samples))

        ir = InstructionGranularitySanitisation().run(builder)
        assert ir.instructions[0].samples == samples

    def test_custom_pulses_with_invalid_length_are_padded(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        drive_chan = model.qubits[0].get_drive_channel()
        sample_time = drive_chan.physical_channel.sample_time
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = 8e-9
        supersampling = int(np.round(clock_cycle / sample_time, 0))
        num_samples = np.random.randint(1, 100) * supersampling

        samples = [1.0] * (num_samples + np.random.randint(1, supersampling - 1))
        builder.add(CustomPulse(drive_chan, samples))

        ir = InstructionGranularitySanitisation(clock_cycle).run(builder)
        assert len(ir.instructions[0].samples) == num_samples + supersampling


class TestPhaseResetSanitisation:
    hw = EchoModelLoader(qubit_count=4).load()

    @pytest.mark.parametrize("reset_qubits", [False, True])
    def test_phase_reset_shot(self, reset_qubits):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        if reset_qubits:
            builder.add(PhaseReset(self.hw.qubits))

        active_targets = {}
        for qubit in self.hw.qubits:
            builder.X(target=qubit)
            drive_pulse_ch = qubit.get_drive_channel()
            active_targets[drive_pulse_ch] = qubit

        n_instr_before = len(builder.instructions)

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActiveChannelResults(target_map=active_targets))
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        # One `PhaseReset` instruction with possibly multiple targets gets added to the IR,
        # even if there is already a phase reset present.
        assert len(builder.instructions) == n_instr_before + 1
        assert isinstance(builder.instructions[0], PhaseReset)
        assert builder.instructions[0].quantum_targets == list(active_targets.keys())

    def test_phase_reset_shot_no_active_pulse_channels(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        for qubit in self.hw.qubits:
            builder.X(target=qubit)

        n_instr_before = len(builder.instructions)

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActiveChannelResults())
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        # No phase reset added since there are no active targets.
        assert len(builder.instructions) == n_instr_before
        assert not isinstance(builder.instructions[0], PhaseReset)


class TestMeasurePhaseResetSanitisation:

    hw = EchoModelLoader(qubit_count=4).load()

    def test_measure_phase_reset(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        for qubit in self.hw.qubits:
            builder.X(target=qubit)
            builder.measure(qubit)

        n_instr_before = len(builder.instructions)

        ir = MeasurePhaseResetSanitisation().run(builder)

        # A phase reset should be added for each measure instruction.
        assert len(ir.instructions) == n_instr_before + len(self.hw.qubits)

        ref_measure_pulse_channels = set(
            [qubit.get_measure_channel() for qubit in self.hw.qubits]
        )
        measure_pulse_channels = set()
        for i, instr in enumerate(ir.instructions):
            if isinstance(instr, MeasurePulse):
                assert isinstance(ir.instructions[i - 1], PhaseReset)
                measure_pulse_channels.update(ir.instructions[i - 1].quantum_targets)

        assert measure_pulse_channels == ref_measure_pulse_channels


class TestInactivePulseChannelSanitisation:

    def test_syncs_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        builder.X(model.qubits[0])
        builder.synchronize(model.qubits[0])
        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.quantum_targets) > 1

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.quantum_targets) == 1
        assert next(iter(sync_inst.quantum_targets)) == model.qubits[0].get_drive_channel()

    def test_delays_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.delay(qubit.get_measure_channel(), 80e-9)

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_phase_shifts_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.phase_shift(qubit.get_measure_channel(), np.pi)

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_Z_is_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.Z(qubit, np.pi)

        num_phase_shifts_before = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)

        num_phase_shifts_after = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        assert num_phase_shifts_after < num_phase_shifts_before
        assert num_phase_shifts_after == 1


@pytest.mark.parametrize("max_duration", [1e-03, 5.5e-06, 2])
class TestInstructionLengthSanitisation:
    hw = EchoModelLoader(8).load()

    def test_delay_not_sanitised(self, max_duration):
        builder = QuantumInstructionBuilder(self.hw)
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        valid_duration = max_duration / 2
        builder.add(Delay(pulse_ch, valid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(max_duration).run(builder)
        assert len(builder.instructions) == 1
        assert builder.instructions[0].duration == valid_duration

    def test_delay_sanitised_zero_remainder(self, max_duration):
        builder = QuantumInstructionBuilder(self.hw)
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        invalid_duration = max_duration * 2
        builder.add(Delay(pulse_ch, invalid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(max_duration).run(builder)
        assert len(builder.instructions) == 2
        for instr in builder.instructions:
            assert instr.duration == max_duration

    def test_delay_sanitised(self, max_duration):
        builder = QuantumInstructionBuilder(self.hw)
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        remainder = random.uniform(1e-06, 2e-06)
        invalid_duration = max_duration * 2 + remainder
        builder.add(Delay(pulse_ch, invalid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(max_duration).run(builder)
        assert len(builder.instructions) == 3
        for instr in builder.instructions[:-1]:
            assert instr.duration == max_duration
        assert math.isclose(builder.instructions[-1].duration, remainder)

    def test_square_pulse_not_sanitised(self, max_duration):
        builder = QuantumInstructionBuilder(self.hw)
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        valid_width = max_duration / 2
        builder.add(Pulse(pulse_ch, shape=PulseShapeType.SQUARE, width=valid_width))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(max_duration).run(builder)
        assert len(builder.instructions) == 1
        assert builder.instructions[0].width == valid_width

    def test_square_pulse_sanitised(self, max_duration):
        builder = QuantumInstructionBuilder(self.hw)
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        remainder = random.uniform(1e-06, 2e-06)
        invalid_width = max_duration * 2 + remainder
        builder.add(Pulse(pulse_ch, shape=PulseShapeType.SQUARE, width=invalid_width))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(max_duration).run(builder)
        assert len(builder.instructions) == 3

        for instr in builder.instructions[:-1]:
            assert instr.width == max_duration
        assert math.isclose(builder.instructions[-1].width, remainder)


class TestSynchronizeTask:

    def test_synchronize_task_adds_sync(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]
        drive_chan = qubit.get_drive_channel()
        measure_chan = qubit.get_measure_channel()

        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(target_map={drive_chan: qubit, measure_chan: qubit})
        )

        builder = model.create_builder()
        builder.pulse(drive_chan, shape=PulseShapeType.SQUARE, width=800e-9)
        builder.pulse(measure_chan, shape=PulseShapeType.SQUARE, width=400e-9)

        ir = SynchronizeTask().run(builder, res_mgr)
        assert isinstance(ir.instructions[-1], Synchronize)
        assert set(ir.instructions[-1].quantum_targets) == set([drive_chan, measure_chan])

    def test_synchronize_task_not_adding_if_inactive(self):
        model = EchoModelLoader().load()

        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={}))

        builder = model.create_builder()

        ir = SynchronizeTask().run(builder, res_mgr)
        assert len(ir.instructions) == 0


class TestEndOfTaskResetSanitisation:

    @pytest.mark.parametrize("reset_q1", [False, True])
    @pytest.mark.parametrize("reset_q2", [False, True])
    def test_resets_added(self, reset_q1, reset_q2):
        model = EchoModelLoader().load()
        qubits = model.qubits[0:2]

        builder = model.create_builder()
        builder.had(qubits[0])
        builder.cnot(qubits[0], qubits[1])
        builder.measure(qubits[0])
        builder.measure(qubits[1])
        if reset_q1:
            builder.reset(qubits[0])
        if reset_q2:
            builder.reset(qubits[1])

        res = ActiveChannelResults()
        for qubit in qubits:
            res.target_map[qubit.get_drive_channel()] = qubit
            res.target_map[qubit.get_measure_channel()] = qubit
            res.target_map[qubit.get_acquire_channel()] = qubit
        res.target_map[qubits[0].get_cross_resonance_channel(qubits[1])] = qubits[0]
        res.target_map[qubits[1].get_cross_resonance_cancellation_channel(qubits[0])] = (
            qubits[1]
        )

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        assert before + (not reset_q1) + (not reset_q2) == len(builder.instructions)

        reset_channels = [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, Reset)
        ]
        assert len(reset_channels) == 2
        assert set(reset_channels) == set([qubit.get_drive_channel() for qubit in qubits])

    def test_mid_circuit_reset_is_ignored(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.had(qubit)
        builder.reset(qubit)
        builder.had(qubit)

        res = ActiveChannelResults()
        res.target_map[qubit.get_drive_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        assert before + 1 == len(builder.instructions)

        reset_channels = [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, Reset)
        ]
        assert len(reset_channels) == 2
        assert set(reset_channels) == set([qubit.get_drive_channel()])

    def test_inactive_instruction_after_reset_ignored(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.had(qubit)
        builder.reset(qubit)
        builder.phase_shift(qubit.get_drive_channel(), np.pi)

        res = ActiveChannelResults()
        res.target_map[qubit.get_drive_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        assert before == len(builder.instructions)

        reset_channels = [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, Reset)
        ]
        assert len(reset_channels) == 1
        assert reset_channels[0] == qubit.get_drive_channel()

    def test_reset_with_no_drive_channel(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.pulse(qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=80e-9)

        res = ActiveChannelResults()
        res.target_map[qubit.get_measure_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        reset_instrs = [instr for instr in builder.instructions if isinstance(instr, Reset)]
        assert len(reset_instrs) == 1
        assert reset_instrs[0].quantum_targets[0] == qubit.get_measure_channel()


@pytest.mark.parametrize("passive_reset_time", [3.2e-06, 1e-03, 5.0])
class TestResetsToDelays:
    @pytest.mark.parametrize("add_reset", [True, False])
    def test_qubit_reset(self, passive_reset_time: float, add_reset: bool):

        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.had(qubit)
        if add_reset:
            builder.reset(qubit)

        res = ActiveChannelResults()
        res.target_map[qubit.get_drive_channel()] = qubit
        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = ResetsToDelays(passive_reset_time).run(builder, res_mgr)
        assert before == len(builder.instructions)

        delays = []
        for instr in builder.instructions:
            assert not isinstance(instr, Reset)

            if isinstance(instr, Delay):
                delays.append(instr)

        if add_reset:
            assert len(delays) == 1
            assert len(delays[0].quantum_targets) == 1
            assert delays[0].time == passive_reset_time

    def test_pulse_channel_reset(self, passive_reset_time: float):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]
        pulse_channels = qubit.get_all_channels()

        builder = model.create_builder()
        builder.reset(pulse_channels)

        res = ActiveChannelResults()
        for pulse_ch in pulse_channels:
            res.target_map[pulse_ch] = qubit
        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = ResetsToDelays(passive_reset_time).run(builder, res_mgr)

        delays = []
        for instr in builder.instructions:
            assert not isinstance(instr, Reset)

            if isinstance(instr, Delay):
                delays.append(instr)

        # All pulse channels belong to a single qubit so only 1 reset -> delay.
        assert len(delays) == 1
        # All pulse channels in the qubit are active channels.
        assert len(delays[0].quantum_targets) == len(pulse_channels)
        assert delays[0].time == passive_reset_time

    @pytest.mark.parametrize("reset_chan", ["acquire", "measure"])
    def test_reset_with_no_drive_channel(self, passive_reset_time, reset_chan):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        if reset_chan == "measure":
            reset_chan = qubit.get_measure_channel()
        else:
            reset_chan = qubit.get_acquire_channel()

        builder = model.create_builder()
        builder.pulse(qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.acquire(qubit.get_acquire_channel(), time=80e-9, delay=0.0)
        builder.add(Reset(reset_chan))

        res = ActiveChannelResults()
        res.target_map[qubit.get_measure_channel()] = qubit
        res.target_map[qubit.get_acquire_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = ResetsToDelays(passive_reset_time).run(builder, res_mgr)
        reset_instrs = [instr for instr in builder.instructions if isinstance(instr, Reset)]
        assert len(reset_instrs) == 0
        delay_instrs = [instr for instr in builder.instructions if isinstance(instr, Delay)]
        assert len(delay_instrs) == 1
        assert set(delay_instrs[0].quantum_targets) == set(
            [qubit.get_acquire_channel(), qubit.get_measure_channel()]
        )


class MockPulseShapeType(Enum):
    HEXAGON = "hexagon"
    SPAGHETTI = "spaghetti"


def mock_evaluate(data, t, phase_offset):
    if isinstance(data, Pulse) and data.shape in MockPulseShapeType:
        if data.shape == MockPulseShapeType.HEXAGON:
            return np.linspace(0, 1, len(t))
        elif data.shape == MockPulseShapeType.SPAGHETTI:
            return np.linspace(-1, 0, len(t))
    else:
        return evaluate_shape(data, t, phase_offset)


class TestEvaluatePulses:

    @pytest.mark.parametrize("scale", [True, False])
    def test_pulse_not_lowered(self, scale: bool):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        builder.pulse(
            chan,
            shape=PulseShapeType.SQUARE,
            width=400e-9,
            amp=1.0,
            ignore_channel_scale=scale,
        )

        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        pulse = builder.instructions[0]
        assert isinstance(pulse, Pulse)
        assert pulse.channel == chan
        assert pulse.ignore_channel_scale
        assert pulse.amp == 1.0 * (1.0 if scale else 0.05)

    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("attributes", pulse_attributes)
    def test_non_square_pulses_are_lowered(self, scale: bool, attributes: dict[str]):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        builder.pulse(chan, width=400e-9, amp=1.0, ignore_channel_scale=scale, **attributes)

        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        pulse = builder.instructions[0]
        assert isinstance(pulse, CustomPulse)
        assert pulse.ignore_channel_scale
        assert len(pulse.samples) == int(np.ceil(400e-9 / chan.sample_time - 1e-10))
        assert pulse.channel == chan
        # the pulse is never sampled at the peak, so amp is actually < 1.0
        assert np.max(np.abs(builder.instructions[0].samples)) >= 0.99 * (
            1.0 if scale else 0.05
        )

    @pytest.mark.parametrize("scale", [True, False])
    def test_custom_pulse(self, scale):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        builder.add(CustomPulse(chan, np.ones(80), ignore_channel_scale=scale))
        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        pulse = builder.instructions[0]
        assert isinstance(pulse, CustomPulse)
        assert pulse.ignore_channel_scale
        assert len(pulse.samples) == 80
        assert np.allclose(pulse.samples, 1.0 if scale else 0.05)

    @pytest.mark.parametrize("scale", [True, False])
    def test_acquire_is_lowered(self, scale):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_acquire_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        pulse = Pulse(
            chan,
            shape=PulseShapeType.SQUARE,
            width=400e-9,
            amp=1.0,
            ignore_channel_scale=scale,
        )
        builder.acquire(chan, time=400e-9, filter=pulse, delay=0.0)

        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        acquire = builder.instructions[0]
        assert isinstance(acquire, Acquire)
        pulse = acquire.filter
        assert isinstance(pulse, CustomPulse)
        assert pulse.ignore_channel_scale
        assert len(pulse.samples) == int(np.ceil(400e-9 / chan.sample_time - 1e-10))
        assert pulse.channel == chan
        assert np.allclose(pulse.samples, (1.0 if scale else 0.05))

    @pytest.mark.parametrize("scale", [True, False])
    def test_acquire_is_not_lowered(self, scale):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_acquire_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        pulse = Pulse(
            chan,
            shape=PulseShapeType.SQUARE,
            width=400e-9,
            amp=1.0,
            ignore_channel_scale=scale,
        )
        builder.acquire(chan, time=400e-9, filter=pulse, delay=0.0)

        EvaluatePulses(acquire_ignored_shapes=[PulseShapeType.SQUARE]).run(builder)
        assert len(builder.instructions) == 1
        acquire = builder.instructions[0]
        assert isinstance(acquire, Acquire)
        pulse = acquire.filter
        assert isinstance(pulse, Pulse)
        assert pulse.ignore_channel_scale
        assert pulse.shape == PulseShapeType.SQUARE

    def test_evaluate_injection_with_square(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        builder = model.create_builder()
        builder.pulse(chan, shape=PulseShapeType.SQUARE, width=400e-9, amp=1.0)
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.HEXAGON,
            width=400e-9,
            amp=1.0,
        )
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.SPAGHETTI,
            width=400e-9,
            amp=1.0,
        )

        EvaluatePulses(eval_function=mock_evaluate).run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Pulse)
        assert builder.instructions[0].shape == PulseShapeType.SQUARE
        assert isinstance(builder.instructions[1], CustomPulse)
        assert isinstance(builder.instructions[2], CustomPulse)

    def test_evaluate_injection_with_custom_ignore(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        builder = model.create_builder()
        builder.pulse(chan, shape=PulseShapeType.SQUARE, width=400e-9, amp=1.0)
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.HEXAGON,
            width=400e-9,
            amp=1.0,
        )
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.SPAGHETTI,
            width=400e-9,
            amp=1.0,
        )

        EvaluatePulses(
            ignored_shapes=[MockPulseShapeType.HEXAGON], eval_function=mock_evaluate
        ).run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], CustomPulse)
        assert isinstance(builder.instructions[1], Pulse)
        assert builder.instructions[1].shape == MockPulseShapeType.HEXAGON
        assert isinstance(builder.instructions[2], CustomPulse)

    def test_evaluate_injection_with_two_custom_ignores(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        builder = model.create_builder()
        builder.pulse(chan, shape=PulseShapeType.SQUARE, width=400e-9, amp=1.0)
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.HEXAGON,
            width=400e-9,
            amp=1.0,
        )
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.SPAGHETTI,
            width=400e-9,
            amp=1.0,
        )

        EvaluatePulses(
            ignored_shapes=[PulseShapeType.SQUARE, MockPulseShapeType.HEXAGON],
            eval_function=mock_evaluate,
        ).run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Pulse)
        assert builder.instructions[0].shape == PulseShapeType.SQUARE
        assert isinstance(builder.instructions[1], Pulse)
        assert builder.instructions[1].shape == MockPulseShapeType.HEXAGON
        assert isinstance(builder.instructions[2], CustomPulse)

    @pytest.mark.parametrize("shape1", list(PulseShapeType) + list(MockPulseShapeType))
    @pytest.mark.parametrize("shape2", list(PulseShapeType) + list(MockPulseShapeType))
    def test_hashing_for_different_shapes(self, shape1, shape2):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        pulse1 = Pulse(chan, shape1, width=400e-9)
        pulse2 = Pulse(chan, shape2, width=400e-9)
        pass_ = EvaluatePulses()
        if shape1 == shape2:
            assert pass_.hash_pulse(pulse1) == pass_.hash_pulse(pulse2)
        else:
            assert pass_.hash_pulse(pulse1) != pass_.hash_pulse(pulse2)

    def test_hashing_for_different_param(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        pulse1 = Pulse(chan, PulseShapeType.SQUARE, width=254e-9)
        pulse2 = Pulse(chan, PulseShapeType.SQUARE, width=454e-9)
        pass_ = EvaluatePulses()
        assert pass_.hash_pulse(pulse1) != pass_.hash_pulse(pulse2)

    def test_hashing_for_different_channels(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan2 = model.qubits[0].get_measure_channel()
        pulse1 = Pulse(chan, PulseShapeType.SQUARE, width=400e-9)
        pulse2 = Pulse(chan2, PulseShapeType.SQUARE, width=400e-9)
        pass_ = EvaluatePulses()
        assert pass_.hash_pulse(pulse1) != pass_.hash_pulse(pulse2)


@pytest.mark.parametrize("explicit_close", [False, True])
class TestRepeatTranslation:
    hw = EchoModelLoader(1).load()

    @staticmethod
    def _check_loop_start(ir: InstructionBuilder, indices: list[int]):
        for index in indices:
            # Create Variable
            assert isinstance(var := ir.instructions[index], Variable)
            assert var.var_type == LoopCount
            name_base = var.name.removesuffix("_count")

            # Assign with 0 to start
            assert isinstance(assign := ir.instructions[index + 1], Assign)
            assert name_base in assign.name
            assert assign.value == 0

            # Create Label
            assert isinstance(label := ir.instructions[index + 2], Label)
            assert name_base == label.name

    @staticmethod
    def _check_loop_close(ir: InstructionBuilder, indices: list[int], repeats: list[int]):
        for index, repeat in zip(indices, repeats):
            # Increment LoopCount Variable with 1
            assert isinstance(assign := ir.instructions[index], Assign)
            name_base = assign.name.removesuffix("_count")
            assert isinstance(plus := assign.value, Plus)
            assert isinstance(var := plus.left, Variable)
            assert name_base in var.name
            assert var.var_type == LoopCount
            assert plus.right == 1

            # Conditional Jump
            assert isinstance(jump := ir.instructions[index + 1], Jump)
            assert name_base == jump.target
            assert isinstance(condition := jump.condition, GreaterThan)
            assert condition.right == var
            assert condition.left == repeat

    def test_single_repeat(self, explicit_close):
        builder = self.hw.create_builder()
        builder.repeat(1000)

        if explicit_close:
            builder.add(EndRepeat())

        ir = RepeatTranslation(self.hw).run(builder)

        assert len(ir.existing_names) == 1

        self._check_loop_start(ir, [0])
        self._check_loop_close(ir, [3], [1000])

    def test_multiple_repeat(self, explicit_close):
        builder = self.hw.create_builder()
        builder.repeat(1000).repeat(200)

        if explicit_close:
            builder.add([EndRepeat(), EndRepeat()])

        ir = RepeatTranslation(self.hw).run(builder)

        assert len(ir.existing_names) == 2

        self._check_loop_start(ir, [0, 3])
        self._check_loop_close(ir, [8, 6], [1000, 200])

    @pytest.mark.parametrize("first", [0, 2])
    @pytest.mark.parametrize("second", [0, 3])
    @pytest.mark.parametrize("third", [0, 5])
    @pytest.mark.parametrize("fourth", [0, 1])
    @pytest.mark.parametrize("fifth", [0, 4])
    def test_with_other_instructions(
        self, explicit_close, first, second, third, fourth, fifth
    ):
        """Test all possible combinations of having additional instructions, before,
        between, and after the repeats, with and without explicitly closing scopes.

        [<first>, Repeat_a, <second>, Repeat_b, <third>, <close_b>, <fourth>, <close_a>, <fifth>]
        """
        builder = self.hw.create_builder()
        start_indices = [0, 3]
        close_indices = [8, 6]

        if first > 0:
            builder.add([Instruction()] * first)
            start_indices = [i + first for i in start_indices]
            close_indices = [i + first for i in close_indices]

        builder.repeat(1000)

        if second > 0:
            builder.add([Instruction()] * second)
            start_indices[1] += second
            close_indices = [i + second for i in close_indices]

        builder.repeat(300)

        if third > 0:
            builder.add([Instruction()] * third)
            close_indices = [i + third for i in close_indices]

        if explicit_close:
            builder.add(EndRepeat())

        if fourth > 0:
            builder.add([Instruction()] * fourth)
            if not explicit_close:
                close_indices[1] += fourth
            close_indices[0] += fourth

        if explicit_close:
            builder.add(EndRepeat())

        if fifth > 0:
            builder.add([Instruction()] * fifth)
            if not explicit_close:
                close_indices = [i + fifth for i in close_indices]

        ir = RepeatTranslation(self.hw).run(builder)

        self._check_loop_start(ir, start_indices)
        self._check_loop_close(ir, close_indices, [1000, 300])
