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
    PostProcessType,
    Pulse,
    Sweep,
    SweepValue,
    Variable,
)
from qat.purr.compiler.runtime import execute_instructions, get_builder

from .utils import ListReturningEngine


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
                        PulseChannel("", PhysicalChannel("", 1, PhysicalBaseband("", 1))),
                        PulseShapeType.SQUARE,
                        0,
                    )
                    for _ in range(201000)
                ]
            )


class TestInstructionExecution:
    @pytest.mark.parametrize(
        "measure_instruction",
        [
            lambda b: b.measure_mean_z,
            lambda b: b.measure_mean_signal,
            lambda b: b.measure_single_shot_z,
            lambda b: b.measure_scope_mode,
            lambda b: b.measure_single_shot_binned,
            lambda b: b.measure_single_shot_signal,
        ],
        ids=lambda v: v.__code__.co_names[0],
    )
    def test_measure_instructions(self, measure_instruction):
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
        )
        measure_instruction(builder)(qubit)
        results = execute_instructions(hw, builder)
        assert results is not None

    def check_size(self, results, expected_shape):
        if isinstance(results, np.ndarray):
            assert results.shape == expected_shape
        else:
            dims = set()

            def _check_size(list_, dim):
                dims.add(dim)
                assert len(list_) == expected_shape[dim]
                if len(list_) > 0 and not isinstance(list_[0], list):
                    return
                for l in list_:
                    _check_size(l, dim + 1)

            _check_size(results, 0)
            assert max(dims) == len(expected_shape) - 1

    @pytest.mark.parametrize(
        "engine, form", [(EchoEngine, np.ndarray), (ListReturningEngine, list)]
    )
    @pytest.mark.parametrize(
        "sweeps",
        [
            {},
            {"amp": [i * 1e6 for i in range(5)]},
            {
                "amp": [i * 1e6 for i in range(5)],
                "width": [i * 100e-9 for i in range(1, 4)],
            },
        ],
        ids=lambda val: f"{len(val)} sweep variables",
    )
    def test_batched_instruction_execution(self, sweeps, engine, form):
        hw = get_default_echo_hardware()
        hw.default_repeat_count = int(hw.shot_limit * 1.5)
        eng = engine(hw)

        vars_ = {"amp": 1e6, "width": 100e-9}
        shape = []

        qubit = hw.get_qubit(0)
        builder = get_builder(hw)
        for n, v in sweeps.items():
            builder.sweep(SweepValue(n, v))
            vars_[n] = Variable(n)
            shape.append(len(v))
        if len(shape) < 1:
            shape.append(1)
        shape = (*shape, hw.default_repeat_count)
        builder.pulse(
            qubit.get_drive_channel(),
            width=vars_["width"],
            shape=PulseShapeType.SQUARE,
            amp=vars_["amp"],
        )
        builder.measure_single_shot_z(qubit)
        results, metrics = execute_instructions(eng, builder)
        assert isinstance(results, form)
        self.check_size(results, shape)

    @pytest.mark.skip("Needs fixing for combining bathes of mean results.")
    def test_batched_instruction_execution_with_mean(self):
        hw = get_default_echo_hardware()
        hw.default_repeat_count = int(hw.shot_limit * 1.5)
        eng = EchoEngine(hw)

        qubit = hw.get_qubit(0)
        amps = [i * 1e6 for i in range(5)]
        builder = get_builder(hw).sweep(SweepValue("amp", amps))
        builder.pulse(
            qubit.get_drive_channel(),
            width=100e-9,
            shape=PulseShapeType.SQUARE,
            amp=Variable("amp"),
        )
        builder.measure_mean_z(qubit)
        results = execute_instructions(eng, builder)[0]
        assert results.shape == (5,)


class TestSweep:
    def test_sweep_runs(self):
        hw = get_default_echo_hardware(2)
        builder = (
            get_builder(hw)
            .sweep(SweepValue("variable", [0.0, 1.0, 2.0]))
            .device_assign(
                hw.get_qubit(0).get_drive_channel(), "scale", Variable("variable")
            )
        )
        execute_instructions(EchoEngine(hw), builder)

    def test_sweep_reverts(self):
        hw = get_default_echo_hardware(2)
        hw.get_qubit(0).get_drive_channel().scale = 5.0
        builder = (
            get_builder(hw)
            .sweep(SweepValue("variable", [0.0, 1.0, 2.0]))
            .device_assign(
                hw.get_qubit(0).get_drive_channel(), "scale", Variable("variable")
            )
            .device_assign(
                hw.get_qubit(0).get_drive_channel(), "sclae", Variable("variable")
            )
        )
        with pytest.raises(Exception):
            execute_instructions(EchoEngine(hw), builder)
        assert hw.get_qubit(0).get_drive_channel().scale == 5.0


class TestInstructionSerialisation:
    def test_basic_gate(self):
        hw = get_default_echo_hardware(4)
        builder = (
            get_builder(hw)
            .X(hw.get_qubit(0).get_drive_channel(), np.pi / 2.0)
            .measure_mean_z(hw.get_qubit(0))
        )

        seri = builder.serialize()
        deseri = InstructionBuilder.deserialize(seri)

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

        seri = builder.serialize()
        deseri = InstructionBuilder.deserialize(seri)

        for original, serialised in zip(builder.instructions, deseri.instructions):
            assert str(original) == str(serialised)
