# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random

import numpy as np
import pytest

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.measure import AcquireMode
from qat.ir.measure import MeasureBlock as PydMeasureBlock
from qat.ir.measure import PostProcessing as PydPostProcessing
from qat.ir.measure import PostProcessType, ProcessAxis
from qat.ir.waveforms import GaussianWaveform, SquareWaveform
from qat.middleend.passes.analysis import ActivePulseChannelAnalysis
from qat.middleend.passes.transform import (
    AcquireSanitisation,
    InactivePulseChannelSanitisation,
    InstructionGranularitySanitisation,
    PhaseOptimisation,
    PydPhaseOptimisation,
    PydPostProcessingSanitisation,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.instructions import (
    Acquire,
    CustomPulse,
    Delay,
    PhaseShift,
    Pulse,
    PulseShapeType,
    Synchronize,
)
from qat.utils.hardware_model import generate_hw_model


class TestPhaseOptimisation:
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


class TestPydPhaseOptimisation:
    def test_empty_constructor(self):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        for qubit in hw.qubits.values():
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
            assert builder_optimised.number_of_instructions == hw.number_of_qubits
        else:
            assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        for qubit in hw.qubits.values():
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
            assert len(phase_shifts) == hw.number_of_qubits
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        qubits = list(hw.qubits.values())

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
            assert len(phase_shifts) == hw.number_of_qubits
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = generate_hw_model(2)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        qubits = list(hw.qubits.values())

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


class TestPydPostProcessingSanitisation:
    hw = generate_hw_model(32)

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

    def test_only_first_acquire_is_decomposed(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        acquire_chan = model.qubits[0].get_acquire_channel()
        acquire_block_time = acquire_chan.physical_channel.block_time
        builder = model.create_builder()

        # Make some instructions to test
        builder.acquire(
            acquire_chan, time=acquire_block_time * 10, delay=acquire_block_time
        )
        builder.acquire(
            acquire_chan, time=acquire_block_time * 10, delay=acquire_block_time
        )

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Delay)
        assert isinstance(builder.instructions[1], Acquire)
        assert isinstance(builder.instructions[2], Acquire)
        assert builder.instructions[2].delay == 0.0


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

        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis().run(builder, res_mgr)
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

        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis().run(builder, res_mgr)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_phase_shifts_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.phase_shift(qubit.get_measure_channel(), np.pi)

        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis().run(builder, res_mgr)
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

        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis().run(builder, res_mgr)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)

        num_phase_shifts_after = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        assert num_phase_shifts_after < num_phase_shifts_before
        assert num_phase_shifts_after == 1
