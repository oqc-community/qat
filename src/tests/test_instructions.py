# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.devices import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    PulseShapeType,
)
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.instructions import (
    Acquire,
    Instruction,
    PostProcessType,
    Pulse,
    Sweep,
    SweepValue,
    Variable,
)
from qat.purr.compiler.runtime import execute_instructions, get_builder


class TestInstruction:
    def test_name_assignment(self):
        builder = get_builder(get_default_echo_hardware())
        label1 = builder.create_label()
        label2 = builder.create_label()
        assert label1.name != label2.name
        assert label1.name in builder.existing_names
        assert label2.name in builder.existing_names

    def test_nested_sweep_iterator(self):
        sweep_iter = SweepIterator(
            Sweep(SweepValue("dave", [1, 2, 3, 4, 5])),
            SweepIterator(
                Sweep(SweepValue("dave", [1, 2, 3])),
                SweepIterator(Sweep(SweepValue("dave", [1, 2, 3, 4, 5, 6, 7, 8]))),
            ),
        )
        incrementor = 0
        while not sweep_iter.is_finished():
            sweep_iter.do_sweep([])
            incrementor += 1

        # Test that actual cycles are both equal the accumulated values, as well as the
        # length
        assert incrementor == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == 120

    def test_sweep_iterator(self):
        sweep_iter = SweepIterator(Sweep(SweepValue("dave", [1, 2, 3, 4, 5])))
        incrementor = 0
        while not sweep_iter.is_finished():
            sweep_iter.do_sweep([])
            incrementor += 1

        # Test that actual cycles are both equal the accumulated values, as well as the
        # length
        assert incrementor == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == 5

    def test_instruction_limit(self):
        qie = EchoEngine()
        with pytest.raises(ValueError):
            qie.validate(
                [
                    Pulse(
                        PulseChannel(
                            "", PhysicalChannel("", 1, PhysicalBaseband("", 1))
                        ),
                        PulseShapeType.SQUARE,
                        0,
                    )
                    for _ in range(201000)
                ]
            )


class TestInstructionExecution:
    @pytest.mark.skip("Needs fixing post processing and down-conversion issues.")
    def test_measure_instructions(hw):
        hw = get_default_echo_hardware(3)
        qubit = hw.get_qubit(0)
        phase_shift_1 = 0.2
        phase_shift_2 = 0.1
        builder = (
            get_builder(hw)
            .phase_shift(qubit, phase_shift_1)
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, phase_shift_2)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit) # triggers a KeyError
            .measure_mean_signal(qubit)
            .measure_single_shot_z(qubit)
            .measure_scope_mode(qubit)
            .measure_single_shot_binned(qubit)
            .measure_single_shot_signal(qubit)
        )
        engine = EchoEngine(hw)
        results = engine.execute(builder.instructions)
        assert results is not None


class TestSweep:
    def test_sweep_runs(self):
        hw = get_default_echo_hardware(2)
        builder = (get_builder(hw).sweep(SweepValue('variable', [0.0, 1.0, 2.0]))
                   .device_assign(hw.get_qubit(0).get_drive_channel(), 'scale', Variable('variable')))
        execute_instructions(EchoEngine(hw), builder)

    def test_sweep_reverts(self):
        hw = get_default_echo_hardware(2)
        hw.get_qubit(0).get_drive_channel().scale = 5.0
        builder = (get_builder(hw).sweep(SweepValue('variable', [0.0, 1.0, 2.0]))
                   .device_assign(hw.get_qubit(0).get_drive_channel(), 'scale', Variable('variable'))
                   .device_assign(hw.get_qubit(0).get_drive_channel(), 'sclae', Variable('variable')))
        with pytest.raises(Exception):
            execute_instructions(EchoEngine(hw), builder)
        assert(hw.get_qubit(0).get_drive_channel().scale == 5.0)


class TestInstructionSerialisation:
    def test_basic_gate(self):
        hw = get_default_echo_hardware(4)
        builder = (
            get_builder(hw)
            .X(hw.get_qubit(0).get_drive_channel(), np.pi / 2.0)
            .measure_mean_z(hw.get_qubit(0))
        )

        seri = builder.serialize()
        deseri = InstructionBuilder.deserialize(seri, hw)

        for original, serialised in zip(builder.instructions, deseri.instructions):
            assert str(original) == str(serialised)

    def test_most_instructions(self):
        hw = get_default_echo_hardware(20, connectivity=None)

        # yapf: disable
        builder = (
            get_builder(hw)
            .X(hw.get_qubit(0).get_drive_channel(), np.pi / 2.0)
            .Y(hw.get_qubit(1))
            .Z(hw.get_qubit(2))
            .reset([hw.get_qubit(7), hw.get_qubit(8)])
            .cnot(hw.get_qubit(2), hw.get_qubit(6))
            .delay(hw.get_qubit(12), 0.2)
            .had(hw.get_qubit(19))
            .assign("dave", 5)
            .returns(["dave"])
            .ECR(hw.get_qubit(15), hw.get_qubit(16))
            .repeat(50, 0.24)
            .T(hw.get_qubit(7))
            .Tdg(hw.get_qubit(7))
            .S(hw.get_qubit(7))
            .Sdg(hw.get_qubit(7))
            .SX(hw.get_qubit(7))
            .SXdg(hw.get_qubit(7))
            .phase_shift(hw.get_qubit(7).get_drive_channel(), 0.72)
            .pulse(hw.get_qubit(12).get_drive_channel(), PulseShapeType.GAUSSIAN, 0.002)
            .acquire(hw.get_qubit(4).get_acquire_channel())
            .results_processing("something", InlineResultsProcessing.Program)
            .post_processing(Acquire(hw.get_qubit(4).get_acquire_channel()), PostProcessType.DOWN_CONVERT)
            .sweep([SweepValue("1", [5]), SweepValue("2", [True])])
            .synchronize([hw.get_qubit(5), hw.get_qubit(7), hw.get_qubit(9)])
            .measure_mean_z(hw.get_qubit(0))
        )
        # yapf: enable

        for inst in builder.instructions:
            inst: Instruction
            iseri = inst.serialize()
            ideseri = Instruction.deserialize(iseri, hw)
            assert str(inst) == str(ideseri)

        seri = builder.serialize()
        deseri = InstructionBuilder.deserialize(seri, hw)

        for original, serialised in zip(builder.instructions, deseri.instructions):
            assert str(original) == str(serialised)
