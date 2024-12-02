import copy
from itertools import chain, product

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing
from pydantic import ValidationError

from qat.ir import instructions as PydanticInstructions
from qat.ir.instruction_list import InstructionList
from qat.ir.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    CrossResonanceCancelPulse,
    CrossResonancePulse,
    CustomPulse,
    Delay,
    DrivePulse,
    EndSweep,
    FrequencyShift,
    Jump,
    Label,
    MeasureBlock,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    Pulse,
    PulseShapeType,
    QuantumInstruction,
    Repeat,
    Reset,
    ResultsProcessing,
    Return,
    SecondStatePulse,
    Sweep,
    Synchronize,
    Variable,
)
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler import instructions as LegacyInstructions
from qat.utils.ir_converter import IRConverter


class TestVariable:

    @pytest.mark.parametrize(
        ["type", "val", "change"],
        [(float, 0.1, 0.2), (str, "clark", "kent"), (float, None, 0.1), (float, 0.1, None)],
    )
    def test_assign(self, type, val, change):
        var = Variable(name="test", var_type=type, value=val)
        assert var.value == val
        var.value = change
        assert var.value == change

    @pytest.mark.parametrize(
        ["type", "val", "change"],
        [
            (float, 0.1, "kent"),
            (str, "clark", 0.2),
            (str, 0.1, "kent"),
            (float, "clark", 0.2),
        ],
    )
    def test_wrong_type(self, type, val, change):
        with pytest.raises(ValueError):
            var = Variable(name="test", var_type=type, value=val)
            var.value = change

    @pytest.mark.parametrize(["val", "change"], [(0.1, "kent"), ("clark", 0.2)])
    def test_no_type(self, val, change):
        var = Variable(name="test", value=val)
        assert var.value == val
        var.value = change
        assert var.value == change

    def test_auto_name(self):
        var = Variable.with_random_name(value=0.2)
        assert isinstance(var.name, str)
        assert var.value == 0.2


class TestRepeat:

    @pytest.mark.parametrize(
        ["repeat_count", "repetition_period"], product([0, 10, 1024], [10, None])
    )
    def test_initiate(self, repeat_count, repetition_period):
        inst = Repeat(repeat_count, repetition_period)
        assert inst.repeat_count == repeat_count
        assert inst.repetition_period == repetition_period


class TestAssign:
    def test_initiate(self):
        inst = Assign("test", 1)
        assert inst.name == "test"
        assert inst.value == 1


class TestReturn:
    def test_single_return(self):
        inst = Return("test")
        assert inst.variables == ["test"]

    @pytest.mark.parametrize("vars", [["test"], ["clark", "kent"]])
    def test_multiple_returns(self, vars):
        inst = Return(vars)
        assert inst.variables == vars

    @pytest.mark.parametrize(
        "vars", [0.4, ["test", 0.4], Variable(name="test", var_type=float, value=0.4)]
    )
    def test_wrong_input(self, vars):
        with pytest.raises(ValidationError):
            Return(vars)


class TestLabel:
    def test_name_gen(self):
        lbl = Label.with_random_name()
        assert isinstance(lbl.name, str)

    @pytest.mark.parametrize("name", [4, 3.14, None])
    def test_validation(self, name):
        with pytest.raises(ValidationError):
            Label(name)


class TestJump:
    @pytest.mark.parametrize("name", ["test", Label.with_random_name()])
    def test_jump(self, name):
        inst = Jump(name)
        assert isinstance(inst.target, str)


# TODO: add tests to make sure sweep validators work as expected...
class TestSweep:
    pass


class TestResults:
    @pytest.mark.parametrize("res_processing", InlineResultsProcessing)
    def test_res_processing(self, res_processing):
        inst = ResultsProcessing("test", res_processing)
        assert inst.variable == "test"
        assert inst.res_processing == res_processing

    @pytest.mark.parametrize("res_processing", ["mean"])
    def test_wrong_results(self, res_processing):
        with pytest.raises(ValidationError):
            ResultsProcessing("test", res_processing)


class TestQuantumInstruction:
    model = get_default_echo_hardware()

    def test_create_single_target(self):
        targets = chain(
            self.model.basebands,
            self.model.physical_channels,
            self.model.pulse_channels,
            self.model.quantum_devices,
        )
        for target in targets:
            inst = QuantumInstruction(quantum_targets=target)
            assert inst.quantum_targets == target

    def test_create_multiple_targets(self):
        targets = [
            list(self.model.basebands),
            list(self.model.physical_channels),
            list(self.model.pulse_channels),
            list(self.model.quantum_devices),
        ]
        # Try with different target types
        targets.append([lst[0] for lst in targets])

        for target_list in targets:
            inst = QuantumInstruction(quantum_targets=target_list)
            for target in inst.quantum_targets:
                assert target in target_list


class TestPhaseShift:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [0.0, -np.pi, 0.235, np.random.rand()]),
    )
    def test_initiation(self, chan, val):
        inst = PhaseShift(chan, val)
        assert inst.channel == chan.full_id()
        assert inst.phase == val

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(
            model.physical_channels.values(), [3, 0.0, -np.pi, 0.235, np.random.rand()]
        ),
    )
    def test_validation_pulse_channel(self, chan, val):
        with pytest.raises(ValueError):
            PhaseShift(chan, val)

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [1j, "pi"]),
    )
    def test_validation_phase(self, chan, val):
        with pytest.raises(ValidationError):
            PhaseShift(chan, val)


class TestFrequencyShift:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [0.0, -np.pi, 0.235, np.random.rand()]),
    )
    def test_initiation(self, chan, val):
        inst = FrequencyShift(chan, val)
        assert inst.channel == chan.full_id()
        assert inst.frequency == val

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(
            model.physical_channels.values(), [3, 0.0, -np.pi, 0.235, np.random.rand()]
        ),
    )
    def test_validation_pulse_channel(self, chan, val):
        with pytest.raises(ValueError):
            FrequencyShift(chan, val)

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [1j, "pi"]),
    )
    def test_validation_phase(self, chan, val):
        with pytest.raises(ValidationError):
            FrequencyShift(chan, val)


class TestDelay:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        ["target", "delay"],
        product(
            [
                model.get_qubit(0),
                list(model.pulse_channels.values())[0],
            ],
            [0.0, 8e-8, 5.6e-7],
        ),
    )
    def test_targets(self, target, delay):
        inst = Delay(target, delay)
        assert inst.quantum_targets == target.full_id()
        assert inst.time == delay
        assert inst.duration == delay

    @pytest.mark.parametrize("delay", [-1, -1e-8, "pi"])
    def test_validation(self, delay):
        with pytest.raises(ValidationError):
            Delay(self.model.get_qubit(0), delay)


# Syncrhonize and PhaseReset are similar in that they just collect unique pulse channels.
# can be tested together


@pytest.mark.parametrize("instruction", [Synchronize, PhaseReset])
class TestSynchronizePhaseReset:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        "targets",
        [
            list(model.pulse_channels.values())[0],
            list(model.pulse_channels.values())[0:2],
        ],
    )
    def test_pulse_channels(self, instruction, targets):
        inst = instruction(targets)
        assert len(inst.quantum_targets) == len(targets) if isinstance(targets, list) else 1

    @pytest.mark.parametrize(
        "targets",
        [0, [0, 1, 2]],
    )
    def test_qubits(self, instruction, targets):
        if not isinstance(targets, list):
            targets = self.model.get_qubit(targets)
        else:
            targets = [self.model.get_qubit(i) for i in targets]
        inst = instruction(targets)
        targets = targets if isinstance(targets, list) else [targets]
        len_targets = sum([len(target.pulse_channels) for target in targets])
        assert len_targets == len(inst.quantum_targets)
        target_ids = [
            chan.full_id() for target in targets for chan in target.pulse_channels.values()
        ]
        for target_id in target_ids:
            assert target_id in inst.quantum_targets

    def test_add_instruction(self, instruction):
        inst1 = instruction(self.model.get_qubit(0))
        inst2 = instruction(self.model.get_qubit(1))
        inst3 = inst1 + inst2
        targets = inst1.quantum_targets
        targets.update(inst2.quantum_targets)
        assert inst3.quantum_targets == targets

    def test_add_target(self, instruction):
        inst = instruction(self.model.get_qubit(0))
        inst = inst + self.model.get_qubit(1)
        inst2 = Synchronize([self.model.get_qubit(0), self.model.get_qubit(1)])
        assert inst.quantum_targets == inst2.quantum_targets

    @pytest.mark.parametrize("target", ["test", 2.54, ["test", 2.54]])
    def test_add_validation(self, instruction, target):
        inst = instruction(self.model.get_qubit(0))
        with pytest.raises(ValueError):
            inst += target

    def test_iadd(self, instruction):
        inst = instruction(self.model.get_qubit(0))
        inst2 = inst + self.model.get_qubit(1)
        inst += self.model.get_qubit(1)
        assert inst.quantum_targets == inst2.quantum_targets


class TestCustomPulse:

    model = get_default_echo_hardware()

    def gaussian_shape(self, points, amp):
        xs = np.linspace(-1, +1, points)
        return amp * np.exp(-0.5 * (xs**2))

    @pytest.mark.parametrize(
        ["points", "amp", "ignore"],
        product([3, 57, 101], [1.0, 1.656e8, 1.5e4 - 2.1e3j], [True, False]),
    )
    def test_custom_pulse(self, points, amp, ignore):
        chan = list(self.model.pulse_channels.values())[0]
        inst = CustomPulse(
            chan,
            self.gaussian_shape(points, amp),
            ignore,
        )
        assert inst.quantum_targets == chan.full_id()
        assert inst.sample_time == chan.physical_channel.sample_time
        assert inst.duration == chan.physical_channel.sample_time * points


class TestPulse:

    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        ["pulse", "duration", "shape"],
        product(
            [
                Pulse,
                MeasurePulse,
                DrivePulse,
                SecondStatePulse,
                CrossResonancePulse,
                CrossResonanceCancelPulse,
            ],
            [0.0, 8e-8, 8e-6],
            PulseShapeType,
        ),
    )
    def test_waveform(self, pulse, duration, shape):
        chan = list(self.model.pulse_channels.values())[0]
        inst = pulse(chan, shape, duration)
        assert inst.width == duration
        assert inst.duration == duration
        assert inst.shape == shape

    def test_invalid_shape(self):
        with pytest.raises(AttributeError):
            chan = list(self.model.pulse_channels.values())[0]
            Pulse(chan, PulseShapeType.CIRCLE, 8e-8)

    @pytest.mark.parametrize("duration", [8e-8j, "pi"])
    def test_invalid_duraiton(self, duration):
        with pytest.raises(ValidationError):
            chan = list(self.model.pulse_channels.values())[0]
            Pulse(chan, PulseShapeType.SQUARE, duration)


class TestAcquire:
    model = get_default_echo_hardware()

    def test_initiate(self):
        chan = self.model.get_qubit(0).get_acquire_channel()
        inst = Acquire(chan)
        assert inst.time == 1e-6
        assert inst.quantum_targets == chan.full_id()

    def test_filter(self):
        chan = self.model.get_qubit(0).get_acquire_channel()
        filter = Pulse(chan, PulseShapeType.GAUSSIAN, 1e-6)
        inst = Acquire(chan, 1e-6, filter=filter)
        assert inst.filter == filter

    @pytest.mark.parametrize("time", [0, 5e-7, 1.01e-6, 2e-6])
    def test_filter_validation(self, time):
        chan = self.model.get_qubit(0).get_acquire_channel()
        filter = Pulse(chan, PulseShapeType.GAUSSIAN, time)
        with pytest.raises(ValidationError):
            inst = Acquire(chan, 1e-6, filter=filter)


class TestPostProcessing:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize("pp", PostProcessType)
    def test_initiate(self, pp):
        chan = self.model.get_qubit(0).get_acquire_channel()
        acquire = Acquire(chan)
        inst = PostProcessing(acquire, pp)
        assert inst.process == pp
        assert inst.acquire == acquire


class TestReset:
    model = get_default_echo_hardware()

    def test_single_qubit(self):
        inst = Reset(self.model.get_qubit(0))
        assert inst.quantum_targets == set(
            [self.model.get_qubit(0).get_drive_channel().full_id()]
        )

    def test_multiple_qubits(self):
        inst = Reset(self.model.qubits[0:2])
        assert inst.quantum_targets == set(
            [
                self.model.get_qubit(0).get_drive_channel().full_id(),
                self.model.get_qubit(1).get_drive_channel().full_id(),
            ]
        )

    def test_wrong_target(self):
        with pytest.raises(ValueError):
            Reset(self.model.get_qubit(0).measure_device)


class TestInstructionBlocks:
    # Tests are extracted and modified from test_instructions.py

    @pytest.mark.parametrize("mode", list(AcquireMode))
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_simple_measure_block(self, num_qubits, mode):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        mb = MeasureBlock.create_block(targets, mode)
        assert isinstance(mb, MeasureBlock)
        assert mb.quantum_targets == [t.full_id() for t in targets]
        assert mb.target_dict[targets[0].full_id()].mode == mode

    @pytest.mark.parametrize("out_vars", [None, "c"])
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_measure_block_with_output_variables(self, num_qubits, out_vars):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        if isinstance(out_vars, str):
            out_vars = [f"{out_vars}[{i}]" for i in range(num_qubits)]

        mb = MeasureBlock.create_block(
            targets, AcquireMode.INTEGRATOR, output_variables=out_vars
        )
        expected = out_vars or [None] * num_qubits
        assert [val.output_variable for val in mb.target_dict.values()] == expected

    def test_add_to_measure_block(self):
        hw = get_default_echo_hardware()
        targets = [hw.qubits[0], hw.qubits[-1]]
        modes = [AcquireMode.INTEGRATOR, AcquireMode.SCOPE]
        out_vars = ["c[0]", "b[1]"]
        mb = MeasureBlock.create_block(
            targets[0],
            modes[0],
            output_variables=out_vars[:1],
        )
        assert mb.quantum_targets == [t.full_id() for t in hw.qubits[:1]]
        mb.add_measurements(targets[1], modes[1], output_variables=out_vars[1])
        assert mb.quantum_targets == [t.full_id() for t in targets]
        assert [val.mode for val in mb.target_dict.values()] == modes
        assert [val.output_variable for val in mb.target_dict.values()] == out_vars

    def test_cannot_add_duplicate_to_measure_block(self):
        hw = get_default_echo_hardware()
        targets = [hw.qubits[0], hw.qubits[-1]]
        out_vars = ["c[0]", "b[1]"]
        mb = MeasureBlock.create_block(
            targets,
            AcquireMode.INTEGRATOR,
            output_variables=out_vars,
        )
        assert mb.quantum_targets == [t.full_id() for t in targets]
        with pytest.raises(ValueError):
            mb.add_measurements(targets[1], AcquireMode.INTEGRATOR)

    def test_measure_block_duration(self):
        hw = get_default_echo_hardware()
        target = hw.qubits[0]
        mb = MeasureBlock.create_block([], AcquireMode.RAW)
        assert mb.duration == 0.0
        mb.add_measurements(target, AcquireMode.INTEGRATOR)
        acq = mb.get_acquires(target)[0]
        assert mb.duration > 0
        assert mb.duration == pytest.approx(acq.delay + acq.duration)
        mb.duration = 1
        assert mb.duration == 1

    def test_get_acquires(self):
        hw = get_default_echo_hardware()
        mb = MeasureBlock.create_block(hw.qubits, AcquireMode.INTEGRATOR)
        acquires = mb.get_acquires(hw.qubits)
        assert all([isinstance(acq, Acquire) for acq in acquires])


class TestInstructionList:

    def all_instructions(self):
        model = get_default_echo_hardware()
        instruction_list = [
            Repeat(1000, 0.0005),
            Assign("test", 42),
            Return(["dave", "steve"]),
            Sweep(),
            EndSweep(),
            Label.with_random_name(),
            Jump("test"),
            PhaseShift(model.get_qubit(0).get_drive_channel(), np.pi),
            FrequencyShift(model.get_qubit(1).get_measure_channel(), 7e8),
            Delay(model.get_qubit(0).get_drive_channel(), 1e-8),
            Synchronize(model.qubits),
            MeasurePulse(
                model.get_qubit(1).get_measure_channel(),
                PulseShapeType.SQUARE,
                Variable(name="length", var_type=float, value=1e-6),
            ),
            DrivePulse(
                model.get_qubit(0).get_drive_channel(), PulseShapeType.GAUSSIAN, 8e-8
            ),
            Acquire(model.get_qubit(0).get_acquire_channel(), 1e-6, AcquireMode.INTEGRATOR),
            Reset(model.get_qubit(1)),
            PhaseReset(model.get_qubit(0)),
            MeasureBlock.create_block(model.get_qubit(1), AcquireMode.RAW),
        ]
        return InstructionList(instruction_list=instruction_list)

    def test_serialize(self):
        instructions = self.all_instructions()
        blob = instructions.serialize()
        new_instructions = instructions.deserialize(blob)
        for i, instruction in enumerate(instructions.instruction_list):
            assert instruction == new_instructions.instruction_list[i]


def make_instruction_list(InstructionTypes):
    hw = get_default_echo_hardware()
    instructions = [
        InstructionTypes.Repeat(1000, 5e-4),
        InstructionTypes.Assign("test", 5),
        InstructionTypes.Return(["test", "dave"]),
        InstructionTypes.Label("test"),
        InstructionTypes.Jump("test"),
        # InstructionTypes.ResultsProcessing("test", PostProcessType.MEAN),
        InstructionTypes.PhaseShift(hw.get_qubit(0).get_drive_channel(), -0.123),
        InstructionTypes.FrequencyShift(hw.get_qubit(0).get_drive_channel(), 1e8),
        InstructionTypes.Delay(hw.get_qubit(0).get_drive_channel(), 8e-8),
        InstructionTypes.Synchronize(hw.qubits[0:3]),
        InstructionTypes.Synchronize(hw.qubits[0]),
        InstructionTypes.Synchronize(hw.qubits[0].get_measure_channel()),
        InstructionTypes.CustomPulse(hw.get_qubit(1).get_drive_channel(), np.ones(101)),
        InstructionTypes.Pulse(
            hw.get_qubit(0).get_drive_channel(), PulseShapeType.SQUARE, 8e-8
        ),
        InstructionTypes.MeasurePulse(
            hw.get_qubit(0).get_measure_channel(), PulseShapeType.GAUSSIAN, 8e-8
        ),
        InstructionTypes.DrivePulse(
            hw.get_qubit(0).get_drive_channel(), PulseShapeType.SECH, 8e-8
        ),
        InstructionTypes.CrossResonancePulse(
            hw.get_qubit(0).get_cross_resonance_channel(hw.get_qubit(1)),
            PulseShapeType.SQUARE,
            8e-8,
        ),
        InstructionTypes.CrossResonanceCancelPulse(
            hw.get_qubit(1).get_cross_resonance_cancellation_channel(hw.get_qubit(0)),
            PulseShapeType.SQUARE,
            8e-8,
        ),
        InstructionTypes.Acquire(hw.get_qubit(0).get_acquire_channel()),
        InstructionTypes.PostProcessing(
            InstructionTypes.Acquire(hw.get_qubit(1).get_acquire_channel()),
            PostProcessType.MEAN,
        ),
        InstructionTypes.Reset(hw.get_qubit(0).get_drive_channel()),
        InstructionTypes.Reset(hw.get_qubit(0)),
        InstructionTypes.Reset(hw.qubits[0:2]),
        InstructionTypes.PhaseReset(hw.get_qubit(0).get_drive_channel()),
        InstructionTypes.PhaseReset(hw.get_qubit(0)),
        InstructionTypes.PhaseReset(hw.qubits[0:2]),
    ]

    return hw, instructions


def eq_instructions(inst1, inst2):
    dict1 = copy.copy(inst1.__dict__)
    dict2 = copy.copy(inst2.__dict__)
    if dict1.keys() != dict2.keys():
        return False
    for attr in dict1.keys():
        if type(dict1[attr]) != type(dict2[attr]):
            return False
        if isinstance(dict1[attr], list):
            for itm in dict1[attr]:
                if not itm in dict2[attr]:
                    return False
    return True


hw, legacy_instructions = make_instruction_list(LegacyInstructions)
hw, pydantic_instructions = make_instruction_list(PydanticInstructions)


class TestConversion:

    @pytest.mark.parametrize("instruction", legacy_instructions)
    @pytest.mark.parametrize("hw", [hw])
    def test_legacy_to_pydantic(self, instruction, hw, monkeypatch):
        monkeypatch.setattr(LegacyInstructions.Instruction, "__eq__", eq_instructions)
        pydantic_inst = IRConverter()._legacy_to_pydantic_instruction(instruction)
        legacy_inst = IRConverter(hw)._pydantic_to_legacy_instruction(pydantic_inst)
        assert legacy_inst == instruction

    @pytest.mark.parametrize("instruction", pydantic_instructions)
    @pytest.mark.parametrize("hw", [hw])
    def test_pydantic_to_legacy(self, instruction, hw, monkeypatch):
        legacy_inst = IRConverter(hw)._pydantic_to_legacy_instruction(instruction)
        pydantic_inst = IRConverter()._legacy_to_pydantic_instruction(legacy_inst)
        assert pydantic_inst == instruction
        assert instruction.inst == pydantic_inst.inst
