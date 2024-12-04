import copy

import numpy as np
import pytest

from qat.ir.instruction_list import InstructionList
from qat.ir.instructions import (
    Assign,
    Delay,
    EndSweep,
    FrequencyShift,
    Jump,
    Label,
    PhaseReset,
    PhaseShift,
    Repeat,
    Reset,
    Return,
    Sweep,
    Synchronize,
    Variable,
)
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    MeasureBlock,
    PostProcessing,
    PostProcessType,
)
from qat.ir.waveforms import CustomWaveform, Pulse, PulseShapeType, PulseType, Waveform
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler import instructions as LegacyInstructions
from qat.utils.ir_converter import IRConverter


class TestInstructionList:

    def all_instructions(self):
        model = get_default_echo_hardware()
        instruction_list = [
            Repeat(repeat_count=1000, repetition_period=0.0005),
            Assign(name="test", value=42),
            Return(variables=["dave", "steve"]),
            Sweep(),
            EndSweep(),
            Label.with_random_name(),
            Jump(target="test"),
            PhaseShift(model.get_qubit(0).get_drive_channel(), phase=np.pi),
            FrequencyShift(model.get_qubit(1).get_measure_channel(), frequency=7e8),
            Delay(model.get_qubit(0).get_drive_channel(), time=1e-8),
            Synchronize(model.qubits),
            Pulse(
                model.get_qubit(1).get_measure_channel(),
                type=PulseType.MEASURE,
                waveform=Waveform(
                    shape=PulseShapeType.SQUARE,
                    width=Variable(name="length", var_type=float, value=1e-6),
                ),
            ),
            Pulse(
                model.get_qubit(0).get_drive_channel(),
                type=PulseType.DRIVE,
                waveform=Waveform(shape=PulseShapeType.GAUSSIAN, width=8e-8),
            ),
            Acquire(
                model.get_qubit(0).get_acquire_channel(),
                time=1e-6,
                mode=AcquireMode.INTEGRATOR,
            ),
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
            if instruction != new_instructions.instruction_list[i]:
                print(instruction)
                print(new_instructions.instruction_list[i])
            assert instruction == new_instructions.instruction_list[i]


def make_pydantic_instruction_list():
    hw = get_default_echo_hardware()
    instructions = [
        Repeat(repeat_count=1000, repetition_period=5e-4),
        Assign(name="test", value=5),
        Return(variables=["test", "dave"]),
        Label(name="test"),
        Jump(target="test"),
        # ResultsProcessing("test", PostProcessType.MEAN),
        PhaseShift(hw.get_qubit(0).get_drive_channel(), phase=-0.123),
        FrequencyShift(hw.get_qubit(0).get_drive_channel(), fequency=1e8),
        Delay(hw.get_qubit(0).get_drive_channel(), time=8e-8),
        Synchronize(hw.qubits[0:3]),
        Synchronize(hw.qubits[0]),
        Synchronize(hw.qubits[0].get_measure_channel()),
        Pulse(
            hw.get_qubit(1).get_drive_channel(),
            type=PulseType.OTHER,
            waveform=CustomWaveform(samples=np.ones(101).tolist()),
        ),
        Pulse(
            targets=hw.get_qubit(0).get_drive_channel(),
            waveform=Waveform(shape=PulseShapeType.SQUARE, width=8e-8),
            type=PulseType.DRIVE,
        ),
        Pulse(
            targets=hw.get_qubit(0).get_measure_channel(),
            waveform=Waveform(shape=PulseShapeType.GAUSSIAN, width=8e-8),
            type=PulseType.MEASURE,
        ),
        Pulse(
            targets=hw.get_qubit(0).get_drive_channel(),
            waveform=Waveform(shape=PulseShapeType.SECH, width=8e-8),
            type=PulseType.DRIVE,
        ),
        Pulse(
            targets=hw.get_qubit(0).get_cross_resonance_channel(hw.get_qubit(1)),
            waveform=Waveform(shape=PulseShapeType.SQUARE, width=8e-8),
            type=PulseType.CROSS_RESONANCE,
        ),
        Pulse(
            targets=hw.get_qubit(1).get_cross_resonance_cancellation_channel(
                hw.get_qubit(0)
            ),
            waveform=Waveform(shape=PulseShapeType.SQUARE, width=8e-8),
            type=PulseType.CROSS_RESONANCE_CANCEL,
        ),
        Acquire.with_random_output_variable(hw.get_qubit(0).get_acquire_channel()),
        PostProcessing(
            acquire=Acquire.with_random_output_variable(
                hw.get_qubit(1).get_acquire_channel()
            ),
            process=PostProcessType.MEAN,
        ),
        Reset(hw.get_qubit(0).get_drive_channel()),
        Reset(hw.get_qubit(0)),
        Reset(hw.qubits[0:2]),
        PhaseReset(hw.get_qubit(0).get_drive_channel()),
        PhaseReset(hw.get_qubit(0)),
        PhaseReset(hw.qubits[0:2]),
    ]

    return hw, instructions


def make_legacy_instruction_list(InstructionTypes):
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
        InstructionTypes.CustomPulse(
            hw.get_qubit(1).get_drive_channel(), np.ones(101).tolist()
        ),
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


hw, legacy_instructions = make_legacy_instruction_list(LegacyInstructions)
hw, pydantic_instructions = make_pydantic_instruction_list()


class TestConversion:

    @pytest.mark.parametrize("instruction", legacy_instructions)
    @pytest.mark.parametrize("hw", [hw])
    def test_legacy_to_pydantic(self, instruction, hw, monkeypatch):
        monkeypatch.setattr(LegacyInstructions.Instruction, "__eq__", eq_instructions)
        pydantic_inst = IRConverter()._legacy_to_pydantic_instruction(instruction)
        legacy_inst = IRConverter(hw)._pydantic_to_legacy_instruction(pydantic_inst)
        print(vars(instruction))
        print(vars(legacy_inst))
        assert legacy_inst == instruction

    @pytest.mark.parametrize("instruction", pydantic_instructions)
    @pytest.mark.parametrize("hw", [hw])
    def test_pydantic_to_legacy(self, instruction, hw, monkeypatch):
        legacy_inst = IRConverter(hw)._pydantic_to_legacy_instruction(instruction)
        pydantic_inst = IRConverter()._legacy_to_pydantic_instruction(legacy_inst)
        assert pydantic_inst == instruction
        assert instruction.inst == pydantic_inst.inst
