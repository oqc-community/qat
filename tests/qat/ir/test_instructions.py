# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import random
from itertools import product

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing
from pydantic import ValidationError

from qat.ir.instructions import (
    Assign,
    Delay,
    FrequencyShift,
    Instruction,
    InstructionBlock,
    PhaseShift,
    QuantumInstruction,
    QuantumInstructionBlock,
    Repeat,
    ResultsProcessing,
    Return,
    Variable,
)
from qat.purr.utils.logger import LoggerLevel
from qat.utils.hardware_model import generate_hw_model


class TestVariable:
    def test_random_variable(self):
        v1 = Variable.with_random_name()
        assert v1.name

        v2 = Variable.with_random_name()
        assert v2.name
        assert v1.name != v2.name

    def test_serialise(self):
        name = "test_name"
        value = 3.14
        var_type = float
        v = Variable(name=name, value=value, var_type=var_type)

        blob = v.model_dump()
        v_deserialised = Variable(**blob)
        assert v == v_deserialised

        blob2 = v_deserialised.model_dump()
        assert blob == blob2


class TestInstructionBlock:
    def test_single_instruction(self):
        instr = Instruction()
        c_from_list = InstructionBlock(instructions=[instr])
        c_from_instr = InstructionBlock(instructions=instr)

        assert c_from_list.number_of_instructions == c_from_instr.number_of_instructions
        assert instr.number_of_instructions == c_from_list.number_of_instructions
        assert instr.number_of_instructions == 1

    def test_number_of_instructions(self):
        c = InstructionBlock()

        ref_number_of_instructions = 0
        # Test with a random composite isntruction with max. depth = 2.
        # This fits our needs for now, but can be expanded later into
        # a more elaborate randomly generated composite pattern.
        while c.number_of_instructions < 20:
            new_instr = random.choices(
                [Instruction(), InstructionBlock()], weights=[0.5, 0.5]
            )[0]
            if isinstance(new_instr, InstructionBlock):
                k = random.randint(1, 3)
                leaves = [Instruction()] * k
                new_instr.add(*leaves)
                ref_number_of_instructions += k
            else:
                ref_number_of_instructions += 1

            c.add(new_instr)

        assert c.number_of_instructions == ref_number_of_instructions

    @pytest.mark.parametrize("k", list(range(0, 5)))
    def test_flatten(self, k):
        c_sub = InstructionBlock()

        instructions = [Instruction()] * k
        c_sub.add(*instructions)

        c_flatten = InstructionBlock()
        c_flatten.add(c_sub, flatten=True)
        assert len(c_flatten.instructions) == k

        c_composite = InstructionBlock()
        c_composite.add(c_sub, flatten=False)
        assert len(c_composite.instructions) == 1

        assert c_flatten.number_of_instructions == c_composite.number_of_instructions

    def test_ordering(self):
        comp_instr = InstructionBlock()

        instructions = [Instruction(), PhaseShift(), Delay(), FrequencyShift()]
        comp_instr.add(*instructions)

        for instr, ref_instr in zip(comp_instr, instructions):
            assert instr == ref_instr


class TestRepeat:
    @pytest.mark.parametrize(
        ["repeat_count", "repetition_period"], product([0, 10, 1024], [10, None])
    )
    def test_init(self, repeat_count, repetition_period):
        inst = Repeat(repeat_count=repeat_count, repetition_period=repetition_period)
        assert inst.repeat_count == repeat_count
        assert inst.repetition_period == repetition_period


class TestAssign:
    def test_init(self):
        inst = Assign(name="test", value=1)
        assert inst.name == "test"
        assert inst.value == 1

    @pytest.mark.parametrize(
        "inst",
        [
            Assign(name="test", value=["test1", "test2", "test3"]),
            Assign(name="test", value=["test1", "test2", ["test3", "test4"]]),
            Assign(name="test", value=("test1", 2)),
            Assign(name="test", value=[("test1", 1), ("test1", 2)]),
            Assign(name="test", value=[("test1", 1), ("test2", 2), "test3"]),
        ],
    )
    def test_serialization_round_trip(self, inst):
        blob = inst.model_dump()
        new_inst = Assign(**blob)
        assert inst == new_inst


class TestReturn:
    def test_single_return(self):
        inst = Return(variables="test")
        assert inst.variables == ["test"]

    @pytest.mark.parametrize("vars", [["test"], ["clark", "kent"]])
    def test_multiple_returns(self, vars):
        inst = Return(variables=vars)
        assert inst.variables == vars

    @pytest.mark.parametrize("vars", [0.4, ["test", 0.4]])
    def test_wrong_input(self, vars):
        with pytest.raises(ValidationError):
            Return(variables=vars)


class TestResults:
    @pytest.mark.parametrize("res_processing_option", [1, 2, 3, 4, 5])
    def test_res_processing(self, res_processing_option):
        res_processing = InlineResultsProcessing(res_processing_option)
        inst = ResultsProcessing(variable="test", results_processing=res_processing)
        assert inst.variable == "test"
        assert inst.results_processing == res_processing

    @pytest.mark.parametrize("res_processing", ["mean"])
    def test_wrong_results_processing_raises_validation_error(self, res_processing):
        with pytest.raises(ValidationError):
            ResultsProcessing(variable="test", results_processing=res_processing)


hw_model = generate_hw_model(4)
qubits = [qubit.uuid for qubit in hw_model.qubits.values()]
pulse_channels = [
    pulse_channel.uuid
    for qubit in hw_model.qubits.values()
    for pulse_channel in qubit.all_pulse_channels
]


class TestQuantumInstruction:
    def test_create_single_qubit_target(self):
        for target in qubits:
            inst = QuantumInstruction(targets=target)
            assert inst.targets == {target}

    def test_create_single_pulse_channel_target(self):
        for target in pulse_channels:
            inst = QuantumInstruction(targets=target)
            assert inst.targets == {target}

    def test_create_multiple_qubit_targets(self):
        inst = QuantumInstruction(targets=qubits)
        assert len(inst.targets) == len(qubits)

    def test_create_multiple_pulse_channel_targets(self):
        inst = QuantumInstruction(targets=pulse_channels)
        assert len(inst.targets) == len(pulse_channels)

    def test_list_to_set_removes_redundancy(self, caplog):
        with caplog.at_level(LoggerLevel.WARNING.value):
            inst = QuantumInstruction(targets=["test", "test"])
            assert "Duplicates have been removed" in caplog.text
        assert inst.targets == set(["test"])

    @pytest.mark.parametrize(
        "targets", [2, np.array([1.0, 2.0]), [2, np.array([1.0, 2.0])]]
    )
    def test_wrong_targets_yield_validation_error(self, targets):
        with pytest.raises((ValidationError, TypeError)):
            QuantumInstruction(targets=targets)


class TestQuantumInstructionBlock:
    @pytest.mark.parametrize("seed", [21, 22, 23, 24])
    def test_duration(self, seed):
        block = QuantumInstructionBlock()
        instructions = [
            QuantumInstruction(
                targets="t1", duration=random.Random(seed).uniform(1e-08, 1e-02)
            ),
            QuantumInstruction(
                targets="t1", duration=random.Random(seed + 1).uniform(1e-08, 1e-02)
            ),
            QuantumInstruction(
                targets="t1", duration=random.Random(seed + 2).uniform(1e-08, 1e-02)
            ),
            QuantumInstruction(targets="t2", duration=1e-09),
        ]
        block.add(*instructions)

        assert (
            block.duration
            == instructions[0].duration
            + instructions[1].duration
            + instructions[2].duration
        )
        assert (
            block._duration_per_target["t1"]
            == instructions[0].duration
            + instructions[1].duration
            + instructions[2].duration
        )
        assert block._duration_per_target["t2"] == instructions[3].duration

    def test_targets(self):
        block = QuantumInstructionBlock()
        instructions = [QuantumInstruction(targets="t1"), QuantumInstruction(targets="t2")]
        block.add(*instructions)
        assert block.targets == {"t1", "t2"}

        block.add(QuantumInstruction(targets="t2"))
        assert block.targets == {"t1", "t2"}

        extra_instructions = [
            QuantumInstruction(targets="t1"),
            QuantumInstruction(targets="t3"),
            QuantumInstruction(targets="t4"),
        ]
        block.add(*extra_instructions)
        assert block.targets == {"t1", "t2", "t3", "t4"}


class TestPhaseShift:
    @pytest.mark.parametrize("pulse_channel", pulse_channels)
    @pytest.mark.parametrize("val", [0.0, -np.pi, 0.235, np.random.rand()])
    def test_initiation(self, pulse_channel, val):
        inst = PhaseShift(targets=pulse_channel, phase=val)
        assert inst.target == pulse_channel
        assert inst.phase == val

    @pytest.mark.parametrize(
        ["pulse_channel", "val"],
        product(pulse_channels, [1j, "pi"]),
    )
    def test_invalid_phase_raises_validation_error(self, pulse_channel, val):
        with pytest.raises(ValidationError):
            PhaseShift(targets=pulse_channel, phase=val)


class TestFrequencyShift:
    @pytest.mark.parametrize(
        ["pulse_channel", "val"],
        product(pulse_channels, [0.0, -np.pi, 0.235, np.random.rand()]),
    )
    def test_initiation(self, pulse_channel, val):
        inst = FrequencyShift(targets=pulse_channel, frequency=val)
        assert inst.target == pulse_channel
        assert inst.frequency == val

    @pytest.mark.parametrize(
        ["pulse_channel", "val"],
        product(pulse_channels, [1j, "pi"]),
    )
    def test_validation_phase(self, pulse_channel, val):
        with pytest.raises(ValidationError):
            FrequencyShift(targets=pulse_channel, frequency=val)


class TestDelay:
    @pytest.mark.parametrize(
        ["target", "delay"],
        product(
            [hw_model.qubit_with_index(0).uuid, pulse_channels[0]],
            [0.0, 8e-8, 5.6e-7],
        ),
    )
    def test_targets(self, target, delay):
        inst = Delay(targets=target, duration=delay)
        assert inst.targets == {target}
        assert inst.duration == delay

    @pytest.mark.parametrize("delay", [-1, -1e-8, "pi"])
    def test_validation(self, delay):
        with pytest.raises(ValidationError):
            Delay(targets=hw_model.qubit_with_index(0).uuid, duration=delay)
