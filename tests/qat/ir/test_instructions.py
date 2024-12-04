from itertools import chain, product

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing
from pydantic import ValidationError

from qat.ir.instructions import (
    Assign,
    Delay,
    FrequencyShift,
    Jump,
    Label,
    PhaseReset,
    PhaseShift,
    QuantumInstruction,
    Repeat,
    Reset,
    ResultsProcessing,
    Return,
    Synchronize,
    Variable,
)
from qat.purr.backends.echo import get_default_echo_hardware


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
        inst = Repeat(repeat_count=repeat_count, repetition_period=repetition_period)
        assert inst.repeat_count == repeat_count
        assert inst.repetition_period == repetition_period


class TestAssign:
    def test_initiate(self):
        inst = Assign(name="test", value=1)
        assert inst.name == "test"
        assert inst.value == 1


class TestReturn:
    def test_single_return(self):
        inst = Return(variables="test")
        assert inst.variables == ["test"]

    @pytest.mark.parametrize("vars", [["test"], ["clark", "kent"]])
    def test_multiple_returns(self, vars):
        inst = Return(variables=vars)
        assert inst.variables == vars

    @pytest.mark.parametrize(
        "vars", [0.4, ["test", 0.4], Variable(name="test", var_type=float, value=0.4)]
    )
    def test_wrong_input(self, vars):
        with pytest.raises(ValidationError):
            Return(variables=vars)


class TestLabel:
    def test_name_gen(self):
        lbl = Label.with_random_name()
        assert isinstance(lbl.name, str)

    @pytest.mark.parametrize("name", [4, 3.14, None])
    def test_validation(self, name):
        with pytest.raises(ValidationError):
            Label(name=name)


class TestJump:
    @pytest.mark.parametrize("name", ["test", Label.with_random_name()])
    def test_jump(self, name):
        inst = Jump(target=name)
        assert isinstance(inst.target, str)


# TODO: add tests to make sure sweep validators work as expected...
class TestSweep:
    pass


class TestResults:
    @pytest.mark.parametrize("res_processing", InlineResultsProcessing)
    def test_res_processing(self, res_processing):
        inst = ResultsProcessing(variable="test", res_processing=res_processing)
        assert inst.variable == "test"
        assert inst.res_processing == res_processing

    @pytest.mark.parametrize("res_processing", ["mean"])
    def test_wrong_results(self, res_processing):
        with pytest.raises(ValidationError):
            ResultsProcessing(variable="test", res_processing=res_processing)


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
            inst = QuantumInstruction(target)
            assert inst.targets == target

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
            inst = QuantumInstruction(target_list)
            for target in inst.targets:
                assert target in target_list


class TestPhaseShift:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [0.0, -np.pi, 0.235, np.random.rand()]),
    )
    def test_initiation(self, chan, val):
        inst = PhaseShift(chan, phase=val)
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
            PhaseShift(chan, phase=val)

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [1j, "pi"]),
    )
    def test_validation_phase(self, chan, val):
        with pytest.raises(ValidationError):
            PhaseShift(chan, phase=val)


class TestFrequencyShift:
    model = get_default_echo_hardware()

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [0.0, -np.pi, 0.235, np.random.rand()]),
    )
    def test_initiation(self, chan, val):
        inst = FrequencyShift(chan, frequency=val)
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
            FrequencyShift(chan, frequency=val)

    @pytest.mark.parametrize(
        ["chan", "val"],
        product(model.pulse_channels.values(), [1j, "pi"]),
    )
    def test_validation_phase(self, chan, val):
        with pytest.raises(ValidationError):
            FrequencyShift(chan, frequency=val)


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
        inst = Delay(target, time=delay)
        assert inst.targets == target.full_id()
        assert inst.time == delay
        assert inst.duration == delay

    @pytest.mark.parametrize("delay", [-1, -1e-8, "pi"])
    def test_validation(self, delay):
        with pytest.raises(ValidationError):
            Delay(self.model.get_qubit(0), time=delay)


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
        assert len(inst.targets) == len(targets) if isinstance(targets, list) else 1

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
        assert len_targets == len(inst.targets)
        target_ids = [
            chan.full_id() for target in targets for chan in target.pulse_channels.values()
        ]
        for target_id in target_ids:
            assert target_id in inst.targets

    def test_add_instruction(self, instruction):
        inst1 = instruction(self.model.get_qubit(0))
        inst2 = instruction(self.model.get_qubit(1))
        inst3 = inst1 + inst2
        targets = inst1.targets
        targets.update(inst2.targets)
        assert inst3.targets == targets

    def test_add_target(self, instruction):
        inst = instruction(self.model.get_qubit(0))
        inst = inst + self.model.get_qubit(1)
        inst2 = instruction([self.model.get_qubit(0), self.model.get_qubit(1)])
        assert inst.targets == inst2.targets

    @pytest.mark.parametrize("target", ["test", 2.54, ["test", 2.54]])
    def test_add_validation(self, instruction, target):
        inst = instruction(self.model.get_qubit(0))
        with pytest.raises(ValueError):
            inst += target

    def test_iadd(self, instruction):
        inst = instruction(self.model.get_qubit(0))
        inst2 = inst + self.model.get_qubit(1)
        inst += self.model.get_qubit(1)
        assert inst.targets == inst2.targets


class TestReset:
    model = get_default_echo_hardware()

    def test_single_qubit(self):
        inst = Reset(self.model.get_qubit(0))
        assert inst.targets == set([self.model.get_qubit(0).get_drive_channel().full_id()])

    def test_multiple_qubits(self):
        inst = Reset(self.model.qubits[0:2])
        assert inst.targets == set(
            [
                self.model.get_qubit(0).get_drive_channel().full_id(),
                self.model.get_qubit(1).get_drive_channel().full_id(),
            ]
        )

    def test_wrong_target(self):
        with pytest.raises(ValueError):
            Reset(self.model.get_qubit(0).measure_device)
