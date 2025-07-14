# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import math
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import (
    Assign,
    Delay,
    EndRepeat,
    GreaterThan,
    Instruction,
    Jump,
    Label,
    LessThan,
    LoopCount,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    Plus,
    QuantumInstruction,
    Repeat,
    Reset,
    Return,
    Synchronize,
    Variable,
)
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    MeasureBlock,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
)
from qat.ir.waveforms import GaussianWaveform, Pulse, SampledWaveform, SquareWaveform
from qat.middleend.passes.analysis import ActivePulseChannelResults
from qat.middleend.passes.transform import (
    BatchedShots,
    EndOfTaskResetSanitisation,
    EvaluateWaveforms,
    FreqShiftSanitisation,
    InactivePulseChannelSanitisation,
    InitialPhaseResetSanitisation,
    InstructionGranularitySanitisation,
    InstructionLengthSanitisation,
    LowerSyncsToDelays,
    MeasurePhaseResetSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatTranslation,
    ResetsToDelays,
    ReturnSanitisation,
    ScopeSanitisation,
    SquashDelaysOptimisation,
)
from qat.middleend.passes.validation import ReturnSanitisationValidation
from qat.model.loaders.converted import EchoModelLoader as PydEchoModelLoader
from qat.model.target_data import (
    AbstractTargetData,
    QubitDescription,
    ResonatorDescription,
    TargetData,
)

from tests.unit.utils.pulses import test_waveforms


@pytest.mark.parametrize("explicit_close", [False, True])
class TestRepeatTranslation:
    hw = PydEchoModelLoader(1).load()

    @staticmethod
    def _check_loop_start(ir: QuantumInstructionBuilder, indices: list[int]):
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
    def _check_loop_close(
        ir: QuantumInstructionBuilder, indices: list[int], repeats: list[int]
    ):
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
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        builder.repeat(1000)

        if explicit_close:
            builder.add(EndRepeat())

        ir = RepeatTranslation(TargetData.default()).run(builder)

        # assert len(ir.existing_names) == 1

        self._check_loop_start(ir, [0])
        self._check_loop_close(ir, [3], [1000])

    def test_multiple_repeat(self, explicit_close):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        builder.repeat(1000).repeat(200)

        if explicit_close:
            builder.add(*[EndRepeat(), EndRepeat()])

        ir = RepeatTranslation(TargetData.default()).run(builder)

        # assert len(ir.existing_names) == 2

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
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        start_indices = [0, 3]
        close_indices = [8, 6]

        if first > 0:
            builder.add(*[Instruction()] * first)
            start_indices = [i + first for i in start_indices]
            close_indices = [i + first for i in close_indices]

        builder.repeat(1000)

        if second > 0:
            builder.add(*[Instruction()] * second)
            start_indices[1] += second
            close_indices = [i + second for i in close_indices]

        builder.repeat(300)

        if third > 0:
            builder.add(*[Instruction()] * third)
            close_indices = [i + third for i in close_indices]

        if explicit_close:
            builder.add(EndRepeat())

        if fourth > 0:
            builder.add(*[Instruction()] * fourth)
            if not explicit_close:
                close_indices[1] += fourth
            close_indices[0] += fourth

        if explicit_close:
            builder.add(EndRepeat())

        if fifth > 0:
            builder.add(*[Instruction()] * fifth)
            if not explicit_close:
                close_indices = [i + fifth for i in close_indices]

        ir = RepeatTranslation(TargetData.default()).run(builder)

        self._check_loop_start(ir, start_indices)
        self._check_loop_close(ir, close_indices, [1000, 300])


class TestPydPhaseOptimisation:
    hw = PydEchoModelLoader(8).load()

    def test_merged_identical_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        target = qubit.drive_pulse_channel.uuid

        phase_reset = PhaseReset(target=target)
        builder.add(phase_reset)
        builder.add(phase_reset)
        builder.add(Delay(target=target, duration=1e-3))
        assert builder.number_of_instructions == 3

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())
        # The two phase resets should be merged to one.
        assert builder.number_of_instructions == 2

        for inst in builder.instructions[:-1]:
            assert isinstance(inst, PhaseSet)
        channels = set([inst.target for inst in builder.instructions[:-1]])
        assert len(channels) == builder.number_of_instructions - 1

    def test_empty_constructor(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.add(PhaseShift(target=qubit.drive_pulse_channel.uuid, phase=phase))
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        if pulse_enabled:
            assert builder_optimised.number_of_instructions == self.hw.number_of_qubits
        else:
            assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.phase_shift(target=qubit, theta=phase)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PhaseShift)
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
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubits = list(self.hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase)

        random.shuffle(qubits)
        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=SquareWaveform(width=80e-9),
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PhaseShift)
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
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubits = list(self.hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=0.5)
            builder.reset(qubit)

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shifts) == 0

    def test_reset_and_shift_become_set(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        target = self.hw.qubit_with_index(0).drive_pulse_channel.uuid

        builder.add(PhaseReset(target=target))
        builder.add(PhaseShift(target=target, phase=np.pi / 4))
        builder.add(Pulse(target=target, waveform=SquareWaveform(width=80e-9)))

        builder = PhaseOptimisation().run(builder, ResultManager(), MetricsManager())

        assert len(builder.instructions) == 2
        assert isinstance(builder.instructions[0], PhaseSet)
        assert np.isclose(builder.instructions[0].phase, np.pi / 4)

    def test_phaseset_does_not_commute_through_delay(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        target = qubit.drive_pulse_channel.uuid

        builder.add(PhaseReset(target=target))
        builder.add(Delay(target=target, duration=1e-3))
        builder.add(PhaseSet(target=target, phase=np.pi / 2))
        builder.add(Delay(target=target, duration=1e-3))
        builder.add(PhaseShift(target=target, phase=np.pi / 4))
        builder.add(Delay(target=target, duration=1e-3))

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Delay,
            Delay,
        ]

    def test_phaseset_does_not_commute_through_sync(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        target1 = qubit.drive_pulse_channel.uuid
        target2 = qubit.measure_pulse_channel.uuid

        builder.add(PhaseReset(target=target1))
        builder.add(PhaseReset(target=target2))
        builder.add(Delay(target=target1, duration=1e-3))
        builder.add(Synchronize(targets=[target1, target2]))
        builder.add(PhaseSet(target=target1, phase=np.pi / 2))
        builder.add(PhaseSet(target=target2, phase=np.pi / 2))
        builder.add(Delay(target=target2, duration=1e-3))
        builder.add(Synchronize(targets=[target1, target2]))
        builder.add(PhaseShift(target=target1, phase=np.pi / 4))
        builder.add(PhaseShift(target=target2, phase=np.pi / 4))

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
            inst.target for inst in builder.instructions if isinstance(inst, PhaseSet)
        ] == [target1, target2, target2, target1]


class TestPydPostProcessingSanitisation:
    hw = PydEchoModelLoader(32).load()

    def test_meas_acq_with_pp(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure(targets=qubit, mode=AcquireMode.SCOPE, output_variable="test")
            builder.post_processing(
                target=qubit, process_type=PostProcessType.MEAN, output_variable="test"
            )
        n_instr_before = builder.number_of_instructions

        met_mgr = MetricsManager()
        PostProcessingSanitisation().run(builder, ResultManager(), met_mgr)

        assert builder.number_of_instructions == met_mgr.get_metric(
            MetricsType.OptimizedInstructionCount
        )
        assert builder.number_of_instructions == n_instr_before

    @pytest.mark.parametrize(
        "acq_mode,pp_type,pp_axes",
        [
            (AcquireMode.SCOPE, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]),
            (AcquireMode.INTEGRATOR, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]),
            (AcquireMode.INTEGRATOR, PostProcessType.MEAN, [ProcessAxis.TIME]),
        ],
    )
    def test_invalid_acq_pp(self, acq_mode, pp_type, pp_axes):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.measure(targets=qubit, mode=acq_mode, output_variable="test")
        builder.post_processing(
            target=qubit, process_type=pp_type, axes=pp_axes, output_variable="test"
        )
        assert isinstance(builder._ir.tail, PostProcessing)

        # Pass should remove the invalid post-processing instruction from the IR.
        assert not PostProcessingSanitisation()._valid_pp(acq_mode, builder._ir.tail)

        PostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())
        assert not isinstance(builder._ir.tail, PostProcessing)

    def test_invalid_raw_acq(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.measure(targets=qubit, mode=AcquireMode.RAW, output_variable="test")

        with pytest.raises(ValueError):
            PostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

    def test_mid_circuit_measurement_two_diff_post_processing(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(2)

        # Mid-circuit measurement with some manual (different) post-processing options.
        builder.measure(targets=qubit, mode=AcquireMode.SCOPE)
        assert isinstance(builder._ir.tail, MeasureBlock)
        builder.post_processing(
            target=qubit,
            output_variable=builder._ir.tail.output_variables[0],
            process_type=PostProcessType.DOWN_CONVERT,
        )
        builder.X(target=qubit)
        builder.measure(targets=qubit, mode=AcquireMode.INTEGRATOR)
        assert isinstance(builder._ir.tail, MeasureBlock)
        builder.post_processing(
            target=qubit,
            output_variable=builder._ir.tail.output_variables[0],
            process_type=PostProcessType.MEAN,
        )

        PostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

        # Make sure no instructions get discarded in the post-processing sanitisation for a mid-circuit measurement.
        pp = [instr for instr in builder if isinstance(instr, PostProcessing)]
        assert len(pp) == 2
        assert pp[0].output_variable != pp[1].output_variable


class TestMeasurePhaseResetSanitisation:
    hw = PydEchoModelLoader(qubit_count=4).load()

    def test_measure_phase_reset(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        for qubit in self.hw.qubits.values():
            builder.X(target=qubit)
            builder.measure(qubit)

        n_instr_before = builder.number_of_instructions

        ir = MeasurePhaseResetSanitisation(self.hw).run(builder)

        # A phase reset should be added for each measure instruction.
        assert ir.number_of_instructions == n_instr_before + self.hw.number_of_qubits

        ref_measure_pulse_channels = set(
            [qubit.measure_pulse_channel.uuid for qubit in self.hw.qubits.values()]
        )
        measure_pulse_channels = set()

        for i, instr in enumerate(ir.instructions):
            if isinstance(instr, MeasureBlock):
                assert isinstance(ir.instructions[i - 1], PhaseReset)
                measure_pulse_channels.update(ir.instructions[i - 1].targets)

        assert measure_pulse_channels == ref_measure_pulse_channels

    def test_measure_phase_reset_with_subset_of_qubits(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        number_of_measured_qubits = 2
        ref_measure_pulse_channels = set()

        for i in range(number_of_measured_qubits):
            qubit = self.hw.qubit_with_index(0)
            builder.X(target=qubit)
            builder.measure(qubit)
            ref_measure_pulse_channels.add(qubit.measure_pulse_channel.uuid)

        n_instr_before = builder.number_of_instructions
        ir = MeasurePhaseResetSanitisation(self.hw).run(builder)

        # A phase reset should be added for each measure instruction.
        assert ir.number_of_instructions == n_instr_before + number_of_measured_qubits

        measure_pulse_channels = set()
        for i, instr in enumerate(ir.instructions):
            if isinstance(instr, MeasureBlock):
                assert isinstance(ir.instructions[i - 1], PhaseReset)
                measure_pulse_channels.update(ir.instructions[i - 1].targets)

        assert measure_pulse_channels == ref_measure_pulse_channels


class TestInactivePulseChannelSanitisation:
    hw = PydEchoModelLoader(10).load()

    def test_sync_on_one_qubit_with_one_pulse_channel_is_removed(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        builder.X(target=qubit)
        builder.synchronize(targets=qubit)

        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.targets) > 1

        res = ActivePulseChannelResults(target_map={qubit.drive_pulse_channel.uuid: qubit})
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        sync_inst = [inst for inst in builder.instructions if isinstance(inst, Synchronize)]
        assert len(sync_inst) == 0

    def test_sync_on_one_qubit_with_multiple_pulses_is_sanitised(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        builder.X(target=qubit)
        builder.synchronize(targets=qubit)

        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.targets) > 1

        res = ActivePulseChannelResults(
            target_map={
                qubit.drive_pulse_channel.uuid: qubit,
                qubit.measure_pulse_channel.uuid: qubit,
                qubit.acquire_pulse_channel.uuid: qubit,
            }
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        sync_inst = [inst for inst in builder.instructions if isinstance(inst, Synchronize)]
        assert len(sync_inst) == 1

    def test_sync_on_multiple_qubits_is_sanitised(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit1 = self.hw.qubit_with_index(0)
        qubit2 = self.hw.qubit_with_index(1)
        builder.X(target=qubit1)
        builder.X(target=qubit2)
        builder.synchronize(targets=[qubit1, qubit2])

        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.targets) > 1

        res = ActivePulseChannelResults(
            target_map={
                qubit1.drive_pulse_channel.uuid: qubit1,
                qubit2.drive_pulse_channel.uuid: qubit2,
            }
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        sync_inst = [inst for inst in builder.instructions if isinstance(inst, Synchronize)]
        assert len(sync_inst) == 1
        assert len(sync_inst[0].targets) == 2
        assert qubit1.drive_pulse_channel.uuid in sync_inst[0].targets
        assert qubit2.drive_pulse_channel.uuid in sync_inst[0].targets

    def test_delays_are_sanitized(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.pulse(
            target=qubit.drive_pulse_channel.uuid, waveform=SquareWaveform(width=80e-9)
        )

        builder.delay(target=qubit.measure_pulse_channel, duration=80e-9)

        res = ActivePulseChannelResults(target_map={qubit.drive_pulse_channel.uuid: qubit})

        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_phase_shifts_are_sanitized(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.pulse(
            target=qubit.drive_pulse_channel.uuid, waveform=SquareWaveform(width=80e-9)
        )

        builder.phase_shift(target=qubit.measure_pulse_channel, theta=np.pi)

        res = ActivePulseChannelResults(target_map={qubit.drive_pulse_channel.uuid: qubit})

        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_Z_is_sanitized(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.pulse(
            target=qubit.drive_pulse_channel.uuid, waveform=SquareWaveform(width=80e-9)
        )

        builder.Z(target=qubit, theta=np.pi)

        num_phase_shifts_before = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        res = ActivePulseChannelResults(target_map={qubit.drive_pulse_channel.uuid: qubit})
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)

        num_phase_shifts_after = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        assert num_phase_shifts_after < num_phase_shifts_before
        assert num_phase_shifts_after == 1


class TestPydInstructionGranularitySanitisation:
    hw = PydEchoModelLoader(10).load()
    target_data = TargetData(
        max_shots=1000,
        default_shots=10,
        QUBIT_DATA=QubitDescription.random(),
        RESONATOR_DATA=ResonatorDescription.random(),
    )
    qubit = hw.qubits[0]
    drive_chan = qubit.drive_pulse_channel
    acquire_chan = qubit.acquire_pulse_channel

    def test_instructions_with_correct_timings_are_unchanged(self):
        # Make some instructions to test
        # TODO: These three builder calls use three different definitions of the `target`.
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        clock_cycle = self.target_data.clock_cycle
        delay_time = np.random.randint(1, 100) * clock_cycle
        builder.delay(self.drive_chan, delay_time)
        pulse_time = np.random.randint(1, 100) * clock_cycle
        builder.pulse(
            target=self.drive_chan.uuid, waveform=SquareWaveform(width=pulse_time)
        )
        acquire_time = np.random.randint(1, 100) * clock_cycle
        builder.acquire(self.qubit, duration=acquire_time, delay=0.0)

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)

        # compare in units of ns to ensure np.isclose works fine
        assert np.isclose(ir.instructions[0].duration * 1e9, delay_time * 1e9)
        assert np.isclose(ir.instructions[1].duration * 1e9, pulse_time * 1e9)
        assert np.isclose(ir.instructions[2].duration * 1e9, acquire_time * 1e9)

    def test_instructions_are_rounded_down(self):
        # Mock up some channels and a builder
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        # Make some instructions to test
        clock_cycle = self.target_data.clock_cycle
        delay_time = np.random.randint(1, 100) * clock_cycle
        builder.delay(self.drive_chan, delay_time + np.random.rand() * clock_cycle)
        pulse_time = np.random.randint(1, 100) * clock_cycle
        builder.pulse(
            target=self.drive_chan.uuid,
            waveform=SquareWaveform(
                width=pulse_time + np.random.rand() * clock_cycle,
            ),
        )
        acquire_time = np.random.randint(1, 100) * clock_cycle
        builder.acquire(
            self.qubit,
            duration=acquire_time + np.random.rand() * clock_cycle,
            delay=0.0,
        )

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)

        # compare in units of ns to ensure np.isclose works fine
        assert np.isclose(ir.instructions[0].duration * 1e9, delay_time * 1e9)
        assert np.isclose(ir.instructions[1].duration * 1e9, pulse_time * 1e9)
        assert np.isclose(ir.instructions[2].duration * 1e9, acquire_time * 1e9)

    def test_custom_pulses_with_correct_length_are_unchanged(self):
        sample_time = self.qubit.physical_channel.sample_time
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        # Make some instructions to test
        clock_cycle = self.target_data.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))
        num_samples = np.random.randint(1, 100) * supersampling
        samples = [1.0 + 0.0j] * num_samples
        builder.add(
            Pulse(target=self.drive_chan.uuid, waveform=SampledWaveform(samples=samples))
        )

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)
        assert ir.instructions[0].waveform.samples.tolist() == samples

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_custom_pulses_with_invalid_length_are_padded(self, seed):
        # Mock up some channels and a builder
        sample_time = self.target_data.QUBIT_DATA.sample_time
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        # Make some instructions to test
        clock_cycle = self.target_data.QUBIT_DATA.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))
        num_samples = np.random.randint(1, 100) * supersampling

        samples = [1.0 + 0.0j] * (num_samples + np.random.randint(1, supersampling - 1))
        builder.add(
            Pulse(target=self.drive_chan.uuid, waveform=SampledWaveform(samples=samples))
        )
        assert len(builder.instructions[0].waveform.samples) == len(samples)

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)
        n = num_samples + supersampling

        assert len(ir.instructions[0].waveform.samples) == n

    @pytest.mark.skip("Skipped untile SampledWaveform duration is resolved")
    def test_acquires_with_too_large_custom_pulse_filters_are_sanitised(self):
        # TODO: Review for COMPILER-642 changes
        # Mock up some channels and a builder
        sample_time = self.target_data.RESONATOR_DATA.sample_time

        # Make some instructions to test
        clock_cycle = self.target_data.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))

        # Make the times
        num_clock_cycles = np.random.randint(1, 100)
        num_samples = num_clock_cycles * supersampling + 3
        acquire_time = num_samples * sample_time

        # Create the builder
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        filter = Pulse(
            target=self.acquire_chan.uuid,
            waveform=SampledWaveform(samples=np.random.rand(num_samples)),
        )
        acquire = Acquire(
            target=self.acquire_chan, time=acquire_time, delay=0.0, filter=filter
        )
        builder.add(acquire)

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)
        assert isinstance(ir.instructions[0], Acquire)
        assert np.isclose(ir.instructions[0].duration, num_clock_cycles * clock_cycle)
        assert isinstance(ir.instructions[0].filter, Pulse)
        assert (
            len(ir.instructions[0].filter.waveform.samples)
            == num_clock_cycles * supersampling
        )
        assert np.isclose(ir.instructions[0].duration, ir.instructions[0].filter.duration)
        assert np.allclose(
            ir.instructions[0].filter.waveform.samples,
            filter.waveform.samples[0 : len(ir.instructions[0].filter.waveform.samples)],
        )

    @pytest.mark.skip("Skipped untile SampledWaveform duration is resolved")
    def test_acquires_with_too_small_custom_pulse_filters_are_sanitised(self):
        # TODO: Review for COMPILER-642 changes
        # Mock up some channels and a builder
        sample_time = self.target_data.RESONATOR_DATA.sample_time

        # Make some instructions to test
        clock_cycle = self.target_data.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))

        # Make the times
        num_clock_cycles = np.random.randint(1, 100)
        num_samples = num_clock_cycles * supersampling + 3
        acquire_time = num_samples * sample_time

        # Create the builder
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        filter = Pulse(
            target=self.acquire_chan.uuid,
            waveform=SampledWaveform(samples=np.random.rand(num_samples)),
        )
        acquire = Acquire(
            target=self.acquire_chan, time=acquire_time, delay=0.0, filter=filter
        )
        builder.add(acquire)
        filter.waveform.samples = filter.waveform.samples[: num_samples - 5]
        samples = deepcopy(filter.waveform.samples)

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)
        assert isinstance(ir.instructions[0], Acquire)
        assert np.isclose(ir.instructions[0].duration, num_clock_cycles * clock_cycle)
        assert isinstance(ir.instructions[0].filter, Pulse)
        assert (
            len(ir.instructions[0].filter.waveform.samples)
            == num_clock_cycles * supersampling
        )
        assert np.isclose(ir.instructions[0].duration, ir.instructions[0].filter.duration)
        assert np.allclose(ir.instructions[0].filter.waveform.samples[-2:], [0.0, 0.0])
        assert np.allclose(ir.instructions[0].filter.waveform.samples[:-2], samples)

    def test_acuqires_with_square_filters_are_sanitised(self):
        # Mock up some channels and a builder
        sample_time = self.target_data.RESONATOR_DATA.sample_time

        # Make some instructions to test
        clock_cycle = self.target_data.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))

        # Make the times
        num_clock_cycles = np.random.randint(1, 100)
        num_samples = num_clock_cycles * supersampling + 3
        acquire_time = num_samples * sample_time

        # Create the builder
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        filter = Pulse(
            target=self.acquire_chan.uuid, waveform=SquareWaveform(width=acquire_time)
        )
        acquire = Acquire(
            target=self.acquire_chan.uuid, duration=acquire_time, delay=0.0, filter=filter
        )
        builder.add(acquire)

        ir = InstructionGranularitySanitisation(self.hw, self.target_data).run(builder)
        assert isinstance(ir.instructions[0], Acquire)
        assert np.isclose(ir.instructions[0].duration, num_clock_cycles * clock_cycle)
        assert isinstance(ir.instructions[0].filter, Pulse)
        assert np.isclose(ir.instructions[0].duration, ir.instructions[0].filter.duration)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
class TestInstructionLengthSanitisation:
    hw = PydEchoModelLoader(8).load()

    def test_delay_not_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubit_with_index(0).acquire_pulse_channel

        valid_duration = target_data.QUBIT_DATA.pulse_duration_max / 2
        builder.add(Delay(target=pulse_ch.uuid, duration=valid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 1
        assert builder.instructions[0].duration == valid_duration

    def test_delay_sanitised_zero_remainder(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubit_with_index(0).acquire_pulse_channel

        invalid_duration = target_data.QUBIT_DATA.pulse_duration_max * 2
        builder.add(Delay(target=pulse_ch.uuid, duration=invalid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 2
        for instr in builder.instructions:
            assert instr.duration == target_data.QUBIT_DATA.pulse_duration_max

    def test_delay_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubit_with_index(0).acquire_pulse_channel

        remainder = random.uniform(1e-06, 2e-06)
        invalid_duration = target_data.QUBIT_DATA.pulse_duration_max * 2 + remainder
        builder.add(Delay(target=pulse_ch.uuid, duration=invalid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 3
        for instr in builder.instructions[:-1]:
            assert instr.duration == target_data.QUBIT_DATA.pulse_duration_max
        assert math.isclose(builder.instructions[-1].duration, remainder)

    def test_square_pulse_not_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubit_with_index(0).acquire_pulse_channel

        valid_width = target_data.QUBIT_DATA.pulse_duration_max / 2
        builder.add(Pulse(target=pulse_ch.uuid, waveform=SquareWaveform(width=valid_width)))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 1
        assert builder.instructions[0].waveform.width == valid_width

    def test_square_pulse_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubit_with_index(0).acquire_pulse_channel

        remainder = random.uniform(1e-06, 2e-06)
        invalid_width = target_data.QUBIT_DATA.pulse_duration_max * 2 + remainder
        builder.add(
            Pulse(target=pulse_ch.uuid, waveform=SquareWaveform(width=invalid_width))
        )
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 3

        for instr in builder.instructions[:-1]:
            assert instr.waveform.width == target_data.QUBIT_DATA.pulse_duration_max
            assert instr.duration == target_data.QUBIT_DATA.pulse_duration_max
        assert math.isclose(builder.instructions[-1].waveform.width, remainder)
        assert math.isclose(builder.instructions[-1].duration, remainder)


class TestPydReturnSanitisation:
    hw = PydEchoModelLoader(8).load()

    def test_empty_builder(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder, res_mgr)

        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)

        return_instr: Return = builder._ir.tail
        assert len(return_instr.variables) == 0

    def test_single_return(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        builder.returns(variables=["test"])
        ref_nr_instructions = builder.number_of_instructions

        res_mgr = ResultManager()
        ReturnSanitisationValidation().run(builder, res_mgr)
        ReturnSanitisation().run(builder, res_mgr)

        assert builder.number_of_instructions == ref_nr_instructions
        assert builder.instructions[0].variables == ["test"]

    def test_multiple_returns_squashed(self):
        q0 = self.hw.qubit_with_index(0)
        q1 = self.hw.qubit_with_index(1)

        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        builder.measure_single_shot_z(target=q0, output_variable="out_q0")
        builder.measure_single_shot_z(target=q1, output_variable="out_q1")

        output_vars = [
            instr.output_variable for instr in builder if isinstance(instr, Acquire)
        ]
        assert len(output_vars) == 2

        builder.returns(variables=[output_vars[0]])
        builder.returns(variables=[output_vars[1]])

        res_mgr = ResultManager()
        # Two returns in a single IR should raise an error.
        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder, res_mgr)

        # Compress the two returns to a single return and validate.
        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)

        return_instr = builder._ir.tail
        assert isinstance(return_instr, Return)
        for var in return_instr.variables:
            assert var in output_vars


class TestPydBatchedShots:
    @pytest.fixture(scope="class")
    def model(self):
        return PydEchoModelLoader().load()

    def test_with_no_repeat(self, model):
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.delay(model.qubit_with_index(0).drive_pulse_channel, 80e-9)
        target_data = AbstractTargetData(max_shots=10000, default_shots=100)
        ir = BatchedShots(target_data).run(builder)
        assert len([inst for inst in ir if isinstance(inst, Repeat)]) == 0
        assert ir.shots is None
        assert ir.compiled_shots is None

    @pytest.mark.parametrize("num_shots", [1, 1000, 999, 10000])
    def test_not_batched_with_possible_amount(self, num_shots, model):
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.repeat(num_shots)
        target_data = AbstractTargetData(max_shots=10000)
        ir = BatchedShots(target_data).run(builder)
        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        assert len(repeats) == 1
        assert repeats[0].repeat_count == num_shots
        assert ir.shots == num_shots
        assert ir.compiled_shots == num_shots

    @pytest.mark.parametrize("num_shots", [1001, 1999, 2000, 4254])
    def test_shots_are_batched(self, num_shots, model):
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.repeat(num_shots)
        target_data = AbstractTargetData(max_shots=1000)
        ir = BatchedShots(target_data).run(builder)
        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        assert len(repeats) == 1
        assert repeats[0].repeat_count <= 1000
        assert ir.compiled_shots == repeats[0].repeat_count
        assert ir.shots == num_shots
        num_batches = np.ceil(ir.shots / ir.compiled_shots)
        assert num_batches * ir.compiled_shots >= num_shots
        assert (num_batches - 1) * ir.compiled_shots < num_shots


@pytest.mark.parametrize("passive_reset_time", [3.2e-06, 1e-03, 5.0])
class TestResetsToDelays:
    @pytest.fixture(scope="class")
    def model(self):
        return PydEchoModelLoader().load()

    @pytest.mark.parametrize("add_reset", [True, False])
    def test_qubit_reset(self, passive_reset_time: float, add_reset: bool, model):
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=100,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        qubit = model.qubit_with_index(0)

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.had(qubit)
        if add_reset:
            builder.reset(qubit)

        res = ActivePulseChannelResults()
        res.target_map[qubit.drive_pulse_channel.uuid] = qubit
        res_mgr = ResultManager()
        res_mgr.add(res)

        before = builder.number_of_instructions
        builder = ResetsToDelays(model, target_data).run(builder, res_mgr)
        assert before == builder.number_of_instructions

        delays = []
        for instr in builder:
            assert not isinstance(instr, Reset)

            if isinstance(instr, Delay):
                delays.append(instr)

        if add_reset:
            assert len(delays) == 1
            assert len(delays[0].targets) == 1
            assert delays[0].duration == passive_reset_time

    def test_pulse_channel_reset(self, passive_reset_time: float, model):
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=100,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        qubit = model.qubit_with_index(0)

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.reset(qubit)

        res = ActivePulseChannelResults()
        res.target_map[qubit.drive_pulse_channel.uuid] = qubit
        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = ResetsToDelays(model, target_data).run(builder, res_mgr)

        delays = []
        for instr in builder:
            assert not isinstance(instr, Reset)

            if isinstance(instr, Delay):
                delays.append(instr)

        # Only active pulse channel is the drive pulse channel.
        assert len(delays) == 1
        assert len(delays[0].targets) == 1
        assert delays[0].target == qubit.drive_pulse_channel.uuid
        assert delays[0].duration == passive_reset_time

    @pytest.mark.parametrize("reset_chan", ["acquire", "measure"])
    def test_reset_with_no_drive_channel(self, passive_reset_time, reset_chan, model):
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=100,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        qubit_idx = 0
        qubit = model.qubit_with_index(qubit_idx)

        if reset_chan == "measure":
            reset_chan = qubit.measure_pulse_channel
        else:
            reset_chan = qubit.acquire_pulse_channel

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.pulse(
            target=qubit.measure_pulse_channel.uuid, waveform=SquareWaveform(width=80e-9)
        )
        builder.acquire(qubit, duration=80e-9, delay=0.0)
        builder.add(Reset(qubit_target=qubit_idx))

        res = ActivePulseChannelResults()
        res.target_map[qubit.measure_pulse_channel.uuid] = qubit
        res.target_map[qubit.acquire_pulse_channel.uuid] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = ResetsToDelays(model, target_data).run(builder, res_mgr)
        reset_instrs = [instr for instr in builder if isinstance(instr, Reset)]
        assert len(reset_instrs) == 0
        delay_instrs = [instr for instr in builder if isinstance(instr, Delay)]
        assert len(delay_instrs) == 2
        delay_targets = {delay.target for delay in delay_instrs}
        assert delay_targets == set(
            [qubit.measure_pulse_channel.uuid, qubit.acquire_pulse_channel.uuid]
        )


class TestSquashDelaysOptimisation:
    @pytest.fixture(scope="class")
    def model(self):
        return PydEchoModelLoader().load()

    @pytest.mark.parametrize("num_delays", [1, 2, 3, 4])
    @pytest.mark.parametrize("with_phase", [True, False])
    def test_multiple_delays_on_one_channel(self, num_delays, with_phase, model):
        delay_times = np.random.rand(num_delays)

        drive_pulse_ch = model.qubit_with_index(0).drive_pulse_channel
        builder = QuantumInstructionBuilder(hardware_model=model)
        for delay in delay_times:
            builder.delay(target=drive_pulse_ch, duration=delay)
            if with_phase:
                builder.phase_shift(target=drive_pulse_ch, theta=np.random.rand())

        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert builder.number_of_instructions == 1 + with_phase * num_delays
        delay_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Delay)
        ]
        assert len(delay_instructions) == 1
        assert np.isclose(delay_instructions[0].duration, sum(delay_times))

    @pytest.mark.parametrize("num_delays", [1, 2, 3, 4])
    @pytest.mark.parametrize("with_phase", [True, False])
    def test_multiple_delays_on_multiple_channels(self, num_delays, with_phase, model):
        pulse_channels = [qubit.drive_pulse_channel for qubit in model.qubits.values()]
        num_pulse_ch = len(pulse_channels)

        builder = QuantumInstructionBuilder(hardware_model=model)
        accumulated_delays = defaultdict(float)
        for _ in range(num_delays):
            random.shuffle(pulse_channels)
            for pulse_ch in pulse_channels:
                delay = np.random.rand()
                accumulated_delays[pulse_ch.uuid] += delay
                builder.delay(pulse_ch, delay)
                if with_phase:
                    builder.phase_shift(pulse_ch, np.random.rand())

        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert (
            builder.number_of_instructions == (1 + with_phase * num_delays) * num_pulse_ch
        )
        delay_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Delay)
        ]
        assert len(delay_instructions) == num_pulse_ch
        for delay in delay_instructions:
            assert delay.duration == accumulated_delays[delay.target]

    def test_optimize_with_pulse(self, model):
        delay_times = np.random.rand(5)

        pulse_ch = model.qubit_with_index(0).drive_pulse_channel

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.delay(pulse_ch, delay_times[0])
        builder.delay(pulse_ch, delay_times[1])
        builder.pulse(target=pulse_ch.uuid, waveform=SquareWaveform(width=80e-9))
        builder.delay(pulse_ch, delay_times[2])
        builder.delay(pulse_ch, delay_times[3])
        builder.delay(pulse_ch, delay_times[4])
        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert builder.number_of_instructions == 3
        assert isinstance(builder.instructions[0], Delay)
        assert np.isclose(builder.instructions[0].duration, np.sum(delay_times[0:2]))
        assert isinstance(builder.instructions[1], Pulse)
        assert isinstance(builder.instructions[2], Delay)
        assert np.isclose(builder.instructions[2].duration, np.sum(delay_times[2:5]))

    def test_delay_with_multiple_channels(self, model):
        delay_times = np.random.rand(2)

        pulse_ch1 = model.qubit_with_index(0).drive_pulse_channel
        pulse_ch2 = model.qubit_with_index(0).measure_pulse_channel

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.add(Delay(target=pulse_ch1.uuid, duration=5))
        builder.add(Delay(target=pulse_ch2.uuid, duration=5))
        builder.delay(pulse_ch1, delay_times[0])
        builder.delay(pulse_ch2, delay_times[1])
        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert builder.number_of_instructions == 2
        assert builder.instructions[0].duration == 5 + delay_times[0]
        assert builder.instructions[1].duration == 5 + delay_times[1]


class TestScopeSanitisation:
    @pytest.fixture(scope="class")
    def model(self):
        return PydEchoModelLoader().load()

    @pytest.mark.parametrize("num_repeats", [1, 2, 3])
    def test_repeats_are_shifted_to_the_beginning(self, num_repeats, model):
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.X(model.qubit_with_index(0))
        for i in range(num_repeats):
            builder.repeat(i + 1)
        builder = ScopeSanitisation().run(builder)

        for i in range(num_repeats):
            assert isinstance(builder.instructions[i], Repeat)
            assert builder.instructions[i].repeat_count == i + 1


class TestEndOfTaskResetSanitisation:
    @pytest.fixture(scope="class")
    def model(self):
        return PydEchoModelLoader().load()

    @pytest.mark.parametrize("reset_q1", [False, True])
    @pytest.mark.parametrize("reset_q2", [False, True])
    def test_resets_added(self, reset_q1, reset_q2, model):
        qubits = [model.qubit_with_index(0), model.qubit_with_index(1)]

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.had(qubits[0])
        builder.cnot(*qubits)
        builder.measure(qubits[0])
        builder.measure(qubits[1])
        if reset_q1:
            builder.reset(qubits[0])
        if reset_q2:
            builder.reset(qubits[1])

        res = ActivePulseChannelResults()
        for qubit in qubits:
            res.target_map[qubit.drive_pulse_channel.uuid] = qubit
            res.target_map[qubit.measure_pulse_channel.uuid] = qubit
            res.target_map[qubit.acquire_pulse_channel.uuid] = qubit
        res.target_map[qubits[0].cross_resonance_pulse_channels[1].uuid] = qubits[0]
        res.target_map[qubits[1].cross_resonance_cancellation_pulse_channels[0].uuid] = (
            qubits[1]
        )

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = builder.number_of_instructions
        builder = EndOfTaskResetSanitisation(model).run(builder, res_mgr)
        assert before + (not reset_q1) + (not reset_q2) == builder.number_of_instructions

        reset_qubits = [
            inst.qubit_target for inst in builder.instructions if isinstance(inst, Reset)
        ]
        assert len(reset_qubits) == 2
        assert set(reset_qubits) == {0, 1}

    def test_mid_circuit_reset_is_ignored(self, model):
        qubit = model.qubit_with_index(0)

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.had(qubit)
        builder.reset(qubit)
        builder.had(qubit)

        res = ActivePulseChannelResults()
        res.target_map[qubit.drive_pulse_channel.uuid] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = builder.number_of_instructions
        builder = EndOfTaskResetSanitisation(model).run(builder, res_mgr)
        assert before + 1 == builder.number_of_instructions

        reset_qubits = [inst.qubit_target for inst in builder if isinstance(inst, Reset)]
        # Only the first qubit is active.
        assert len(reset_qubits) == 2
        assert set(reset_qubits) == {0}

    def test_inactive_instruction_after_reset_ignored(self, model):
        qubit = model.qubit_with_index(0)

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.had(qubit)
        builder.reset(qubit)
        # Phase shift is a 'virtual' (non-pulse based) instruction,
        # so should not trigger a new `Reset` instruction.
        builder.phase_shift(qubit.drive_pulse_channel, np.pi)

        res = ActivePulseChannelResults()
        res.target_map[qubit.drive_pulse_channel.uuid] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = builder.number_of_instructions
        builder = EndOfTaskResetSanitisation(model).run(builder, res_mgr)
        assert before == len(builder.instructions)

        reset_qubits = [inst.qubit_target for inst in builder if isinstance(inst, Reset)]
        assert len(reset_qubits) == 1
        assert reset_qubits == [0]

    def test_reset_with_no_drive_channel(self, model):
        qubit = model.qubit_with_index(0)

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.pulse(
            target=qubit.measure_pulse_channel.uuid, waveform=SquareWaveform(width=80e-9)
        )

        res = ActivePulseChannelResults()
        res.target_map[qubit.measure_pulse_channel.uuid] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = EndOfTaskResetSanitisation(model).run(builder, res_mgr)
        reset_instrs = [instr for instr in builder if isinstance(instr, Reset)]
        assert len(reset_instrs) == 1
        assert reset_instrs[0].qubit_target == 0


class TestFreqShiftSanitisation:
    @pytest.fixture(scope="class")
    def model(self):
        hw = PydEchoModelLoader().load()
        for q_idx in range(2):
            hw.qubit_with_index(q_idx).freq_shift_pulse_channel.active = True
        return hw

    @pytest.fixture(scope="class")
    def model_no_freq_shift(self):
        return PydEchoModelLoader().load()

    @pytest.fixture(scope="class")
    def freq_shift_pulse_channels(self, model):
        """Model with frequency shift channels added."""
        fs_pulse_channels = set()
        for qubit in model.qubits.values():
            if qubit.freq_shift_pulse_channel.active:
                fs_pulse_channels.add(qubit.freq_shift_pulse_channel)
        return fs_pulse_channels

    @pytest.fixture(scope="class")
    def basic_builder(self, model):
        """Creates a builder with only classical instructions."""

        var = Variable(name="repeat_count", var_type=LoopCount)
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.add(var)
        builder.assign("repeat_count", 0)
        builder.assign("repeat_count", Plus(left=var, right=1))
        builder.returns(["test"])
        return builder

    @pytest.fixture(scope="class")
    def builder_x_and_measure(self, basic_builder):
        """Mocks an X gate and a measure for a builder."""
        new_builder = deepcopy(basic_builder)
        qubit = new_builder.hw.qubit_with_index(0)
        drive_pulse_ch = qubit.drive_pulse_channel
        measure_pulse_ch = qubit.measure_pulse_channel
        acquire_pulse_ch = qubit.acquire_pulse_channel

        new_builder.pulse(target=drive_pulse_ch.uuid, waveform=SquareWaveform(width=40e-9))
        new_builder.pulse(target=drive_pulse_ch.uuid, waveform=SquareWaveform(width=40e-9))
        new_builder.delay(measure_pulse_ch, 80e-9)
        new_builder.delay(acquire_pulse_ch, 128e-9)
        new_builder.pulse(
            target=measure_pulse_ch.uuid, waveform=SquareWaveform(width=400e-9)
        )
        new_builder.acquire(qubit, delay=48e-9, duration=352e-9)
        new_builder.delay(drive_pulse_ch, 400e-9)
        return new_builder

    @pytest.fixture(scope="class")
    def builder(self, model, builder_x_and_measure):
        """Creates a basic builder free of control flow."""
        comp_builder = QuantumInstructionBuilder(hardware_model=model)
        comp_builder += builder_x_and_measure
        comp_builder.returns(["test"])
        return comp_builder

    @pytest.fixture(scope="class")
    def repeat_builder(self, model, builder_x_and_measure):
        """Creates a builder with a repeat scope."""

        qubit = model.qubit_with_index(0)
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.add(Repeat(repeat_count=1000))
        builder.phase_shift(qubit, np.pi / 4)
        builder += builder_x_and_measure
        builder.add(EndRepeat())
        builder.returns(["test"])
        return builder

    @pytest.fixture(scope="class")
    def jump_builder(self, model, builder_x_and_measure):
        """Creates a builder with a label and jump to replicate the behaviour of repeats."""

        qubit = model.qubit_with_index(0)
        var = Variable(name="repeat_count", var_type=LoopCount)
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.add(var)
        builder.assign("repeat_count", 0)
        builder.add(Label(name="repeats"))
        builder.phase_shift(qubit, np.pi / 4)
        builder += builder_x_and_measure
        builder.assign("repeat_count", Plus(left=var, right=1))
        builder.jump("repeats", LessThan(left=var, right=1000))
        builder.returns(["test"])
        return builder

    def test_get_freq_shift_channels(self, model, freq_shift_pulse_channels):
        found_pulse_channels = FreqShiftSanitisation.get_active_freq_shift_pulse_channels(
            model
        )
        assert freq_shift_pulse_channels == set(found_pulse_channels.keys())
        assert set(found_pulse_channels.values()) == set(
            [model.qubit_with_index(i) for i in range(2)]
        )

    def test_add_freq_shift_to_block_ignored_for_no_quantum_instructions(
        self, basic_builder, freq_shift_pulse_channels
    ):
        builder = FreqShiftSanitisation.add_freq_shift_to_ir(
            basic_builder, freq_shift_pulse_channels
        )
        assert all([not isinstance(inst, Pulse) for inst in builder])

    def test_add_freq_shift_to_block_ignored_for_no_duration(
        self, model, freq_shift_pulse_channels
    ):
        builder = QuantumInstructionBuilder(hardware_model=model)
        qubit = model.qubit_with_index(0)
        builder.phase_shift(qubit.drive_pulse_channel, np.pi / 4)
        assert builder.number_of_instructions == 1

        builder = FreqShiftSanitisation(model).add_freq_shift_to_ir(
            builder, freq_shift_pulse_channels
        )
        assert builder.number_of_instructions == 1
        assert isinstance(builder._ir.head, PhaseShift)

    def test_active_results_updated(self, model, freq_shift_pulse_channels):
        qubits = [model.qubit_with_index(i) for i in range(2)]
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.delay(qubits[0].drive_pulse_channel, 80e-9)
        res = ActivePulseChannelResults(
            target_map={qubits[0].drive_pulse_channel.uuid: qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = FreqShiftSanitisation(model).run(builder, res_mgr)
        for fq_pulse_ch in freq_shift_pulse_channels:
            assert fq_pulse_ch.uuid in res.target_map
            assert res.target_map[fq_pulse_ch.uuid] in qubits

    @pytest.mark.parametrize(
        "builder_fixture", ["builder", "repeat_builder", "jump_builder"]
    )
    def test_add_freq_shift(
        self, request, builder_fixture, model, freq_shift_pulse_channels
    ):
        """Considers builders that only have a single block with quantum instructions, and
        ensures the correct frequency shift pulses are added."""
        builder = request.getfixturevalue(builder_fixture)
        pass_ = FreqShiftSanitisation(model)
        builder = pass_.run(builder, ResultManager())

        # Index of the first quantum instruction.
        idx = [i for i, inst in enumerate(builder) if isinstance(inst, QuantumInstruction)][
            0
        ]

        for fq_pulse_ch in freq_shift_pulse_channels:
            pulses = []
            for j, inst in enumerate(builder):
                if isinstance(inst, Pulse) and inst.target == fq_pulse_ch.uuid:
                    assert j >= idx
                    pulses.append(inst)

            assert len(pulses) == 1
            assert pulses[0].duration == 480e-9

    def test_no_freq_shift_pulse_channel(self, model_no_freq_shift):
        builder = QuantumInstructionBuilder(hardware_model=model_no_freq_shift)
        for qubit in model_no_freq_shift.qubits.values():
            builder.X(qubit)

        ref_instructions = builder.instructions
        builder = FreqShiftSanitisation(model_no_freq_shift).run(builder, ResultManager())

        # No instructions added since we do not have freq shift pulse channels.
        assert builder.instructions == ref_instructions

    def test_freq_shift_empty_target(self, model_no_freq_shift):
        res_mgr = ResultManager()
        builder = QuantumInstructionBuilder(hardware_model=model_no_freq_shift)
        builder = FreqShiftSanitisation(model_no_freq_shift).run(builder, res_mgr=res_mgr)
        assert builder.number_of_instructions == 0


class TestPhaseResetSanitisation:
    @pytest.fixture(scope="class")
    def model(self):
        return PydEchoModelLoader().load()

    @pytest.mark.parametrize("reset_qubits", [False, True])
    def test_phase_reset_shot(self, model, reset_qubits):
        builder = QuantumInstructionBuilder(hardware_model=model)

        active_targets = {}
        for ind in model.qubits:
            qubit = model.qubit_with_index(ind)
            drive_pulse_ch_id = qubit.drive_pulse_channel.uuid
            if reset_qubits:
                builder.add(PhaseReset(target=drive_pulse_ch_id))
            builder.X(target=qubit)
            active_targets[drive_pulse_ch_id] = qubit

        n_instr_before = builder.number_of_instructions

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActivePulseChannelResults(target_map=active_targets))
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        assert builder.number_of_instructions == n_instr_before + len(active_targets)
        for key, instr in zip(active_targets.keys(), builder.instructions):
            assert isinstance(instr, PhaseReset)
            assert instr.target == key

    def test_phase_reset_shot_leading_non_quantum_instructions(self, model):
        builder = QuantumInstructionBuilder(hardware_model=model)

        active_targets = {}
        for ind in model.qubits:
            qubit = model.qubit_with_index(ind)
            drive_pulse_ch_id = qubit.drive_pulse_channel.uuid
            # Add a non-quantum instruction before the quantum instructions.
            builder.add(Repeat(repeat_count=42))
            builder.X(target=qubit)
            active_targets[drive_pulse_ch_id] = qubit

        n_instr_before = builder.number_of_instructions

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActivePulseChannelResults(target_map=active_targets))
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        assert builder.number_of_instructions == n_instr_before + len(active_targets)
        assert isinstance(builder.instructions[0], Repeat)
        for key, instr in zip(active_targets.keys(), builder.instructions[1:]):
            assert isinstance(instr, PhaseReset)
            assert instr.target == key

    def test_phase_reset_shot_no_active_pulse_channels(self, model):
        builder = QuantumInstructionBuilder(hardware_model=model)

        for ind in model.qubits:
            qubit = model.qubit_with_index(ind)
            builder.X(target=qubit)

        n_instr_before = builder.number_of_instructions

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActivePulseChannelResults())
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        # No phase reset added since there are no active targets.
        assert builder.number_of_instructions == n_instr_before
        assert not isinstance(builder.instructions[0], PhaseReset)


class TestLowerSyncsToDelays:
    @pytest.mark.parametrize(
        "inst",
        [
            Assign(name="test", value=1.0),
            Return(variables=["test"]),
            Repeat(repeat_count=254),
        ],
    )
    def test_process_instruction(self, inst):
        """Test that LowerSyncsToDelays passes on a select number of non-quantum
        instructions."""
        durations = defaultdict(float)
        new_insts = []
        LowerSyncsToDelays().process_instruction(inst, new_insts, durations)
        assert len(durations) == 0
        assert len(new_insts) == 1
        assert new_insts[0] == inst

    @pytest.mark.parametrize(
        "inst",
        [
            Delay(target="test", duration=1e-6),
            Pulse(target="test", waveform=SquareWaveform(width=1e-6)),
            Acquire(target="test", duration=1e-6, delay=0.0),
        ],
    )
    def test_process_quantum_instruction(self, inst):
        """Test that LowerSyncsToDelays does not pass on quantum instructions."""
        durations = defaultdict(float)
        new_insts = []
        LowerSyncsToDelays().process_instruction(inst, new_insts, durations)
        assert len(durations) == 1
        assert "test" in durations
        assert durations["test"] == inst.duration
        assert len(new_insts) == 1
        assert new_insts[0] == inst

    def test_process_sync(self):
        """Tests that a synchronize instruction is converted to delays, and the durations
        dict is updated accordingly. Tests the case when a channel hasn't yet been seen."""
        durations = defaultdict(float)
        durations["test1"] = 80e-9
        durations["test2"] = 120e-9
        new_insts = []
        inst = Synchronize(targets=["test1", "test2", "test3"])
        LowerSyncsToDelays().process_instruction(inst, new_insts, durations)
        assert len(durations) == 3
        assert "test3" in durations
        assert all(np.isclose(val, 120e-9) for val in durations.values())
        assert len(new_insts) == 2
        for inst in new_insts:
            assert isinstance(inst, Delay)
            assert inst.target in ["test1", "test3"]
            if inst.target == "test1":
                assert np.isclose(inst.duration, 40e-9)
            else:
                assert np.isclose(inst.duration, 120e-9)

    def test_sync_with_two_channels(self):
        model = PydEchoModelLoader().load()
        chan1 = model.qubits[0].drive_pulse_channel
        chan2 = model.qubits[1].drive_pulse_channel

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.pulse(target=chan1.uuid, waveform=SquareWaveform(width=120e-9))
        builder.delay(chan1, 48e-9)
        builder.delay(chan2, 72e-9)
        builder.pulse(target=chan2.uuid, waveform=SquareWaveform(width=168e-9))
        builder.synchronize([chan1, chan2])

        LowerSyncsToDelays().run(builder)
        assert len(builder.instructions) == 5
        assert [type(inst) for inst in builder.instructions] == [
            Pulse,
            Delay,
            Delay,
            Pulse,
            Delay,
        ]
        times = [inst.duration for inst in builder.instructions]
        assert times[0] + times[1] + times[4] == times[2] + times[3]


class TestEvaluateWaveforms:
    @pytest.fixture(scope="class")
    def model(self):
        model = PydEchoModelLoader().load()
        model.qubit_with_index(0).drive_pulse_channel.scale = 2.0
        return model

    @pytest.fixture(scope="class")
    def target_data(self):
        return TargetData.default()

    @pytest.fixture(scope="class")
    def pass_(self, model, target_data):
        return EvaluateWaveforms(
            model, target_data=target_data, ignored_shapes=SquareWaveform
        )

    def test_sanitise_ignore_shapes_default(self):
        ignored = EvaluateWaveforms._sanitise_ignored_shapes(None, (SquareWaveform,))
        assert ignored == (SquareWaveform,)

    def test_sanitise_ignore_shapes_single_item(self):
        ignored = EvaluateWaveforms._sanitise_ignored_shapes(
            GaussianWaveform, (SquareWaveform,)
        )
        assert ignored == (GaussianWaveform,)

    def test_extract_pulse_channel_features(self, model, target_data):
        scales, sample_times = EvaluateWaveforms._extract_pulse_channel_features(
            model, target_data
        )
        for qubit in model.qubits.values():
            for pulse_channel in qubit.pulse_channels.all_pulse_channels:
                assert pulse_channel.uuid in scales
                assert pulse_channel.uuid in sample_times
                assert scales[pulse_channel.uuid] == pulse_channel.scale
                assert (
                    sample_times[pulse_channel.uuid] == target_data.QUBIT_DATA.sample_time
                )
            for pulse_channel in qubit.resonator.pulse_channels.all_pulse_channels:
                assert pulse_channel.uuid in scales
                assert pulse_channel.uuid in sample_times
                assert scales[pulse_channel.uuid] == pulse_channel.scale
                assert (
                    sample_times[pulse_channel.uuid]
                    == target_data.RESONATOR_DATA.sample_time
                )

        assert model.qubit_with_index(0).drive_pulse_channel.scale == 2.0

    def test_evaluate_sampled_waveform_no_scale(self, pass_):
        samples = np.random.rand(100)
        wf = SampledWaveform(samples=samples)
        new_wf = pass_.evaluate_waveform(wf, scale=1.0)
        assert new_wf.samples is samples

    def test_evaluate_sampled_waveform_with_scale(self, pass_):
        samples = np.random.rand(100)
        wf = SampledWaveform(samples=samples)
        new_wf = pass_.evaluate_waveform(wf, scale=0.5)
        assert np.allclose(new_wf.samples, samples * 0.5)

    def test_evaluate_waveform_for_ignored_shape(self, pass_):
        wf = SquareWaveform(width=800e-9, amp=0.5)
        new_wf = pass_.evaluate_waveform(
            wf,
            ignored_shapes=(SquareWaveform,),
            scale=1.0,
            target="test",
            waveform_lookup={},
        )
        assert new_wf == wf

    @pytest.mark.parametrize(
        "waveform",
        [wf for wf in test_waveforms if not isinstance(wf, SquareWaveform)],
        ids=lambda wf: f"{wf.__class__.__name__}",
    )
    def test_evaluate_waveform_for_non_ignored_shape(self, model, pass_, waveform):
        target = model.qubit_with_index(0).drive_pulse_channel.uuid
        new_wf = pass_.evaluate_waveform(
            waveform,
            ignored_shapes=(SquareWaveform,),
            scale=1.0,
            target=target,
            waveform_lookup=defaultdict(dict),
        )
        assert isinstance(new_wf, SampledWaveform)

        scaled_wf = pass_.evaluate_waveform(
            waveform,
            ignored_shapes=(SquareWaveform,),
            scale=0.5,
            target=target,
            waveform_lookup=defaultdict(dict),
        )
        assert np.allclose(0.5 * new_wf.samples, scaled_wf.samples)

    def test_evaluate_waveform_is_cached(self, model, pass_):
        target = model.qubit_with_index(0).drive_pulse_channel.uuid
        waveform = GaussianWaveform(width=80e-9, amp=0.5, rise=1 / 3)
        waveform_lookup = defaultdict(dict)

        # First evaluation should not be cached.
        new_wf = pass_.evaluate_waveform(
            waveform,
            ignored_shapes=(SquareWaveform,),
            scale=1.0,
            target=target,
            waveform_lookup=waveform_lookup,
        )
        assert isinstance(new_wf, SampledWaveform)
        assert target in waveform_lookup
        assert waveform in waveform_lookup[target]
        assert new_wf.samples is waveform_lookup[target][waveform]

        # Second evaluation should use the cached waveform.
        cached_wf = pass_.evaluate_waveform(
            waveform,
            ignored_shapes=(SquareWaveform,),
            scale=1.0,
            target=target,
            waveform_lookup=waveform_lookup,
        )
        assert cached_wf.samples is new_wf.samples

    @pytest.mark.parametrize(
        "instruction",
        [
            Delay(target="test", duration=1e-6),
            Assign(name="test", value=1.0),
            PhaseShift(target="test", theta=np.pi / 2),
        ],
    )
    def test_process_instruction_passes_on_instruction(self, pass_, instruction):
        new_instruction = pass_.process_instruction(instruction)
        assert new_instruction == instruction

    @pytest.mark.parametrize("ignore_channel_scale", [True, False])
    @pytest.mark.parametrize(
        "waveform",
        [wf for wf in test_waveforms if not isinstance(wf, SquareWaveform)],
        ids=lambda wf: f"{wf.__class__.__name__}",
    )
    def test_process_instruction_on_pulse(
        self, pass_, model, ignore_channel_scale, waveform
    ):
        qubit = model.qubit_with_index(0)
        pulse = Pulse(
            target=qubit.drive_pulse_channel.uuid,
            waveform=waveform,
            ignore_channel_scale=ignore_channel_scale,
        )
        new_instruction = pass_.process_instruction(
            pulse, waveform_lookup=defaultdict(dict)
        )
        assert isinstance(new_instruction, Pulse)
        assert isinstance(new_instruction.waveform, SampledWaveform)
        assert new_instruction.target == qubit.drive_pulse_channel.uuid

    @pytest.mark.parametrize("ignore_channel_scale", [True, False])
    def test_process_instruction_on_pulse_with_ignored_shape(
        self, pass_, model, ignore_channel_scale
    ):
        qubit = model.qubit_with_index(0)
        waveform = SquareWaveform(width=800e-9, amp=0.5)
        pulse = Pulse(
            target=qubit.drive_pulse_channel.uuid,
            waveform=waveform,
            ignore_channel_scale=ignore_channel_scale,
        )
        new_instruction = pass_.process_instruction(
            pulse, waveform_lookup=defaultdict(dict)
        )
        assert isinstance(new_instruction, Pulse)
        assert new_instruction.target == qubit.drive_pulse_channel.uuid
        assert new_instruction.ignore_channel_scale == True
        assert new_instruction.waveform.amp == (0.5 if ignore_channel_scale else 1.0)
        if ignore_channel_scale:
            assert new_instruction.waveform == waveform

    @pytest.mark.parametrize("ignore_channel_scale", [True, False])
    def test_process_instruction_on_pulse_with_sampled_pulse(
        self, pass_, model, ignore_channel_scale
    ):
        qubit = model.qubit_with_index(0)
        samples = np.random.rand(100)
        waveform = SampledWaveform(samples=samples)
        pulse = Pulse(
            target=qubit.drive_pulse_channel.uuid,
            waveform=waveform,
            ignore_channel_scale=ignore_channel_scale,
        )
        new_instruction = pass_.process_instruction(
            pulse, waveform_lookup=defaultdict(dict)
        )
        assert isinstance(new_instruction, Pulse)
        assert isinstance(new_instruction.waveform, SampledWaveform)
        assert new_instruction.target == qubit.drive_pulse_channel.uuid
        assert new_instruction.ignore_channel_scale == True
        if ignore_channel_scale:
            assert new_instruction.waveform.samples is samples
        else:
            assert np.allclose(new_instruction.waveform.samples, samples * 2.0)

    def test_process_instruction_on_acquire(self, pass_, model):
        """Test that the acquire instruction is processed correctly."""
        qubit = model.qubit_with_index(0)
        target = qubit.acquire_pulse_channel.uuid
        acquire = Acquire(
            target=target,
            duration=1e-6,
            delay=0.0,
            filter=Pulse(target=target, waveform=SquareWaveform(width=1e-6, amp=0.5)),
        )
        new_instruction = pass_.process_instruction(
            acquire, waveform_lookup=defaultdict(dict)
        )
        assert isinstance(new_instruction, Acquire)
        assert isinstance(new_instruction.filter, Pulse)
        assert isinstance(new_instruction.filter.waveform, SampledWaveform)
        assert np.allclose(new_instruction.filter.waveform.samples, 0.5)

    def test_run(self, pass_, model):
        """Test that the EvaluateWaveforms pass runs correctly on a builder."""
        builder = QuantumInstructionBuilder(hardware_model=model)
        qubit = model.qubit_with_index(0)
        target = qubit.drive_pulse_channel
        acquire_target = qubit.acquire_pulse_channel
        builder.phase_shift(target=target, theta=np.pi / 4)
        builder.pulse(
            target=target.uuid, waveform=GaussianWaveform(width=80e-9, amp=0.5, rise=1 / 3)
        )
        builder.pulse(target=target.uuid, waveform=SquareWaveform(width=80e-9, amp=0.5))
        builder.delay(target, 100e-9)
        builder.acquire(
            qubit,
            delay=0.0,
            duration=200e-9,
            filter=Pulse(
                target=acquire_target.uuid, waveform=SquareWaveform(width=200e-9, amp=0.5)
            ),
        )

        waveform_lookup = defaultdict(dict)
        new_builder = pass_.run(builder, waveform_lookup=waveform_lookup)

        assert isinstance(new_builder.instructions[0], PhaseShift)
        assert isinstance(new_builder.instructions[1], Pulse)
        assert isinstance(new_builder.instructions[1].waveform, SampledWaveform)
        assert isinstance(new_builder.instructions[2], Pulse)
        assert isinstance(new_builder.instructions[2].waveform, SquareWaveform)
        assert isinstance(new_builder.instructions[3], Delay)
        assert isinstance(new_builder.instructions[4], Acquire)
        assert isinstance(new_builder.instructions[4].filter, Pulse)
        assert isinstance(new_builder.instructions[4].filter.waveform, SampledWaveform)
