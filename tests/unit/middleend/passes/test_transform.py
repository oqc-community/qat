# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random
from collections import defaultdict

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
    LoopCount,
    PhaseReset,
    PhaseShift,
    Plus,
    Repeat,
    Reset,
    Return,
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
from qat.ir.waveforms import GaussianWaveform, Pulse, SquareWaveform
from qat.middleend.passes.analysis import ActivePulseChannelResults
from qat.middleend.passes.transform import (
    BatchedShots,
    EndOfTaskResetSanitisation,
    InstructionLengthSanitisation,
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
            builder.add(PhaseShift(targets=qubit.drive_pulse_channel.uuid, phase=phase))
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
                    waveform=SquareWaveform(),
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

    def test_merged_identical_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        target = qubit.drive_pulse_channel.uuid

        phase_reset = PhaseReset(targets=target)
        builder.add(phase_reset)
        builder.add(phase_reset)
        assert builder.number_of_instructions == 2

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())
        # The two phase resets should be merged to one.
        assert builder.number_of_instructions == 1
        # assert set(builder.instructions[0].quantum_targets) == set(phase_reset.quantum_targets)


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
        assert not hasattr(ir, "shots")
        assert not hasattr(ir, "compiled_shots")

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
