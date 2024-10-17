# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import math
from functools import reduce
from operator import mul

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing

from qat import qatconfig
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import (
    MaxPulseLength,
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    PulseShapeType,
)
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    Instruction,
    MeasureBlock,
    PostProcessType,
    Pulse,
    Sweep,
    SweepValue,
    Variable,
)
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.utils.serializer import json_dumps, json_loads
from tests.qat.utils.models import ListReturningEngine


class TestInstruction:
    def test_name_assignment(self):
        builder = get_builder(get_default_echo_hardware())
        label1 = builder.create_label()
        label2 = builder.create_label()
        assert label1.name != label2.name
        assert label1.name in builder.existing_names
        assert label2.name in builder.existing_names

    def _get_sweep_size(self, sweep_iter: SweepIterator):
        """
        Returns the size of the entire loop nest seen as a polyhedra.
        The returned structure is a list s where s[i] represents the sweep length at level i
        """
        if sweep_iter.sweep is None:
            return []
        elif sweep_iter.nested_sweep is None:
            return [sweep_iter.sweep.length]
        else:
            return [sweep_iter.sweep.length] + self._get_sweep_size(sweep_iter.nested_sweep)

    def _decompose(self, accumulated: int, sweep_iter: SweepIterator):
        result = []
        sweep_size = self._get_sweep_size(sweep_iter)
        for i in range(len(sweep_size) - 1):
            subspace = reduce(mul, sweep_size[i + 1 :], 1)
            result.append(accumulated // subspace)
            accumulated = accumulated % subspace
        result.append(accumulated)

        return result

    def test_nested_sweep_iterator(self):
        sweep_iter = SweepIterator(
            Sweep(SweepValue("dave", [1, 2, 3, 4, 5])),
            SweepIterator(
                Sweep(SweepValue("dave", [1, 2, 3])),
                SweepIterator(Sweep(SweepValue("dave", [1, 2, 3, 4, 5, 6, 7, 8]))),
            ),
        )
        assert self._get_sweep_size(sweep_iter) == [5, 3, 8]
        assert sweep_iter.get_current_sweep_iteration() == [-1, -1, -1]

        incrementor = 0
        while not sweep_iter.is_finished():
            sweep_iter.do_sweep([])
            assert sweep_iter.get_current_sweep_iteration() == self._decompose(
                incrementor, sweep_iter
            )
            incrementor += 1

        # Test that actual cycles are both equal the accumulated values, as well as the
        # length
        assert incrementor == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == 120

        sweep_iter.reset_iteration()
        assert sweep_iter.get_current_sweep_iteration() == [-1, -1, -1]

    def test_sweep_iterator(self):
        sweep_iter = SweepIterator(Sweep(SweepValue("dave", [1, 2, 3, 4, 5])))
        assert self._get_sweep_size(sweep_iter) == [5]
        assert sweep_iter.get_current_sweep_iteration() == [-1]

        incrementor = 0
        while not sweep_iter.is_finished():
            sweep_iter.do_sweep([])
            assert sweep_iter.get_current_sweep_iteration() == [incrementor]
            incrementor += 1

        assert sweep_iter.get_current_sweep_iteration() == [incrementor - 1]

        # Test that actual cycles are both equal the accumulated values, as well as the
        # length
        assert incrementor == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == sweep_iter.accumulated_sweep_iteration
        assert sweep_iter.length == 5

        sweep_iter.reset_iteration()
        assert sweep_iter.get_current_sweep_iteration() == [-1]

    def test_empty_sweep_iterator(self):
        sweep_iter = SweepIterator()
        assert self._get_sweep_size(sweep_iter) == []
        assert sweep_iter.get_current_sweep_iteration() == []

        sweep_iter.reset_iteration()
        assert sweep_iter.get_current_sweep_iteration() == []

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

    def test_instruction_limit_ignored_by_flag(self):
        hw = get_default_echo_hardware()
        qie = EchoEngine(model=hw)

        invalid_pulse = [
            Pulse(
                PulseChannel("", PhysicalChannel("", 1, PhysicalBaseband("", 1))),
                PulseShapeType.SQUARE,
                width=MaxPulseLength + 1e-09,
            )
        ]

        qatconfig.DISABLE_PULSE_DURATION_LIMITS = True
        qie.validate(invalid_pulse)

        qatconfig.DISABLE_PULSE_DURATION_LIMITS = False
        with pytest.raises(ValueError):
            qie.validate(invalid_pulse)

    def test_instruction_limit_variable_ignored_by_flag(self):
        hw = get_default_echo_hardware()
        qubit = hw.qubits[0]
        qie = EchoEngine(model=hw)

        widths = [i * MaxPulseLength for i in np.arange(0.5, 2.01, 0.5)]
        builder = (
            get_builder(hw)
            .sweep(SweepValue("width", widths))
            .pulse(
                qubit.get_drive_channel(),
                width=Variable("width"),
                shape=PulseShapeType.SQUARE,
            )
        )

        qatconfig.DISABLE_PULSE_DURATION_LIMITS = True
        qie.validate(builder.instructions)

        qatconfig.DISABLE_PULSE_DURATION_LIMITS = False
        with pytest.raises(ValueError):
            qie.validate(builder.instructions)

    def test_no_entanglement(self):
        hw = get_default_echo_hardware(2)
        builder = get_builder(hw)
        qubit0 = hw.get_qubit(0)
        qubit1 = hw.get_qubit(1)
        builder.X(qubit0)
        builder.X(qubit1)
        assert builder._entanglement_map == {qubit0: {qubit0}, qubit1: {qubit1}}

    def test_01_entanglement(self):
        hw = get_default_echo_hardware(3)
        builder = get_builder(hw)
        qubit0 = hw.get_qubit(0)
        qubit1 = hw.get_qubit(1)
        qubit2 = hw.get_qubit(2)
        builder.ECR(qubit0, qubit1)
        assert builder._entanglement_map == {
            qubit0: {qubit0, qubit1},
            qubit1: {qubit1, qubit0},
            qubit2: {qubit2},
        }

    def test_012_entanglement(self):
        hw = get_default_echo_hardware(3)
        builder = get_builder(hw)
        qubit0 = hw.get_qubit(0)
        qubit1 = hw.get_qubit(1)
        qubit2 = hw.get_qubit(2)
        builder.ECR(qubit0, qubit1)
        builder.ECR(qubit1, qubit2)
        assert builder._entanglement_map == {
            qubit0: {qubit0, qubit1, qubit2},
            qubit1: {qubit1, qubit0, qubit2},
            qubit2: {qubit0, qubit1, qubit2},
        }

    @pytest.mark.parametrize(
        "acquire_width",
        np.linspace(1e-6, 6e-6, 10),
    )
    def test_acquire_filter(self, acquire_width):
        hw = get_default_echo_hardware(1)
        measure_ch = hw.get_qubit(0).get_measure_channel()
        acquire_ch = hw.get_qubit(0).get_acquire_channel()
        no = math.floor(round((acquire_width / measure_ch.sample_time), 4))
        samples = np.linspace(0, acquire_width, no, dtype=np.complex64)
        filter = CustomPulse(measure_ch, samples)
        acquire = Acquire(
            acquire_ch,
            time=acquire_width,
            filter=filter,
        )
        assert all(samples == acquire.filter.samples)
        with pytest.raises(ValueError):
            Acquire(
                acquire_ch,
                time=acquire_width + 0.5e-6,
                filter=filter,
            )

    @pytest.mark.parametrize(
        "acquire_width",
        [-1e-6, 0],
    )
    def test_acquire_filter_edge_cases(self, acquire_width):
        hw = get_default_echo_hardware(1)
        measure_ch = hw.get_qubit(0).get_measure_channel()
        acquire_ch = hw.get_qubit(0).get_acquire_channel()

        filter = Pulse(measure_ch, PulseShapeType.SQUARE, acquire_width)

        with pytest.raises(ValueError):
            Acquire(
                acquire_ch,
                time=acquire_width,
                filter=filter,
            )


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

    def test_deviceinjector_reverts(self):
        hw = get_default_echo_hardware(2)
        qubit = hw.get_qubit(0)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        num_points = 10
        freq_range = 50e6
        center_freq = qubit.get_acquire_channel().frequency
        freqs = center_freq + np.linspace(-freq_range, freq_range, num_points)
        var_name = f"freq{qubit.index}"
        output_variable = f"Q{qubit.index}"

        builder = (
            get_builder(hw)
            .sweep(SweepValue(var_name, freqs))
            .device_assign(measure_channel, "frequency", Variable(var_name))
            .device_assign(acquire_channel, "ycneuqerf", Variable(var_name))
            .measure_mean_signal(qubit, output_variable)
            .repeat(1000, 500e-6)
        )

        measure_channel.frequency = 9e9
        with pytest.raises(Exception):
            execute_instructions(EchoEngine(hw), builder)
        assert measure_channel.frequency == 9e9


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
        hw.default_repeat_count = int(hw.repeat_limit * 1.5)
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
        results = execute_instructions(eng, builder)[0]
        assert isinstance(results, form)
        self.check_size(results, shape)

    @pytest.mark.skip("Needs fixing for combining bathes of mean results.")
    def test_batched_instruction_execution_with_mean(self):
        hw = get_default_echo_hardware()
        hw.default_repeat_count = int(hw.repeat_limit * 1.5)
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


class TestInstructionSerialisation:
    @pytest.mark.parametrize(
        "hardware_model_type",
        [get_default_echo_hardware, get_default_qiskit_hardware, get_default_RTCS_hardware],
    )
    def test_basic_gate(self, hardware_model_type):
        hw = hardware_model_type(4)
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
        hw = get_default_echo_hardware(20)
        builder = (
            get_builder(hw)
            .X(hw.get_qubit(0).get_drive_channel(), np.pi / 2.0)
            .Y(hw.get_qubit(1))
            .Z(hw.get_qubit(2))
            .reset([hw.get_qubit(7), hw.get_qubit(8)])
            .cnot(hw.get_qubit(2), hw.get_qubit(3))
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
            .results_processing("something", InlineResultsProcessing.Program)
            .post_processing(
                Acquire(hw.get_qubit(4).get_acquire_channel()),
                PostProcessType.DOWN_CONVERT,
            )
            .sweep([SweepValue("1", [5]), SweepValue("2", [True])])
            .synchronize([hw.get_qubit(5), hw.get_qubit(7), hw.get_qubit(9)])
            .measure_mean_z(hw.get_qubit(0))
        )
        seri = builder.serialize()
        deseri = InstructionBuilder.deserialize(seri)
        for original, serialised in zip(builder.instructions, deseri.instructions):
            assert str(original) == str(serialised)

    def test_json_instructions(self, monkeypatch):
        def equivalent(self, other):
            return isinstance(self, type(other)) and (vars(self) == vars(other))

        monkeypatch.setattr(Instruction, "__eq__", equivalent)

        hw = get_default_echo_hardware(20)
        builder = (
            get_builder(hw)
            .X(hw.get_qubit(0).get_drive_channel(), np.pi / 2.0)
            .Y(hw.get_qubit(1))
            .Z(hw.get_qubit(2))
            .reset([hw.get_qubit(7), hw.get_qubit(8)])
            .cnot(hw.get_qubit(2), hw.get_qubit(3))
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
            .results_processing("something", InlineResultsProcessing.Program)
            .post_processing(
                Acquire(hw.get_qubit(4).get_acquire_channel()),
                PostProcessType.DOWN_CONVERT,
            )
            .sweep([SweepValue("1", [5]), SweepValue("2", [True])])
            .synchronize([hw.get_qubit(5), hw.get_qubit(7), hw.get_qubit(9)])
            .measure_mean_z(hw.get_qubit(0))
        )

        for instruction in builder.instructions:
            js = json_dumps(instruction)
            loaded = json_loads(js, model=hw)
            assert loaded == instruction


class TestInstructionBlocks:
    @pytest.mark.parametrize("mode", list(AcquireMode))
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_simple_measure_block(self, num_qubits, mode):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        mb = MeasureBlock(targets, mode)
        assert isinstance(mb, MeasureBlock)
        assert mb.quantum_targets == targets
        assert mb._target_dict[targets[0].full_id()]["mode"] == mode
        assert mb._entangled_qubits == set(targets)

    @pytest.mark.parametrize("out_vars", [None, "c"])
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_measure_block_with_output_variables(self, num_qubits, out_vars):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        if isinstance(out_vars, str):
            out_vars = [f"{out_vars}[{i}]" for i in range(num_qubits)]

        mb = MeasureBlock(targets, AcquireMode.INTEGRATOR, output_variables=out_vars)
        expected = out_vars or [None] * num_qubits
        assert [val["output_variable"] for val in mb._target_dict.values()] == expected

    @pytest.mark.parametrize("entangled_qubits", [None, 3])
    @pytest.mark.parametrize("num_qubits", [1, 3])
    def test_create_measure_block_with_entanglement(self, num_qubits, entangled_qubits):
        hw = get_default_echo_hardware()
        targets = hw.qubits[:num_qubits]

        if isinstance(entangled_qubits, int):
            entangled_qubits = targets.copy().append(hw.get_qubit(entangled_qubits))

        mb = MeasureBlock(
            targets, AcquireMode.INTEGRATOR, entangled_qubits=entangled_qubits
        )
        assert mb.quantum_targets == targets
        if entangled_qubits is None:
            assert mb._entangled_qubits == set(targets)
        else:
            assert mb._entangled_qubits == set(entangled_qubits)

    def test_add_to_measure_block(self):
        hw = get_default_echo_hardware()
        targets = [hw.qubits[0], hw.qubits[-1]]
        modes = [AcquireMode.INTEGRATOR, AcquireMode.SCOPE]
        out_vars = ["c[0]", "b[1]"]
        entangled_qubits = hw.qubits[:2]
        mb = MeasureBlock(
            targets[0],
            modes[0],
            output_variables=out_vars[:1],
            entangled_qubits=entangled_qubits,
        )
        assert mb.quantum_targets == hw.qubits[:1]
        mb.add_measurements(targets[1], modes[1], output_variables=out_vars[1])
        entangled_qubits.append(targets[1])
        assert mb.quantum_targets == targets
        assert [val["mode"] for val in mb._target_dict.values()] == modes
        assert [val["output_variable"] for val in mb._target_dict.values()] == out_vars
        assert mb._entangled_qubits == set(entangled_qubits)

    def test_cannot_add_duplicate_to_measure_block(self):
        hw = get_default_echo_hardware()
        targets = [hw.qubits[0], hw.qubits[-1]]
        out_vars = ["c[0]", "b[1]"]
        mb = MeasureBlock(
            targets,
            AcquireMode.INTEGRATOR,
            output_variables=out_vars,
        )
        assert mb.quantum_targets == targets
        with pytest.raises(ValueError):
            mb.add_measurements(targets[1], AcquireMode.INTEGRATOR)

    def test_measure_block_duration(self):
        hw = get_default_echo_hardware()
        target = hw.qubits[0]
        mb = MeasureBlock([], AcquireMode.RAW)
        assert mb.duration == 0.0
        mb.add_measurements(target, AcquireMode.INTEGRATOR)
        acq = mb.get_acquires(target)[0]
        assert mb.duration > 0
        assert mb.duration == pytest.approx(acq.delay + acq.duration)
        mb._duration = 1
        assert mb.duration == 1

    def test_sequential_measure_block_from_builder(self):
        """Test that measurements can be added in sequence."""
        hw = get_default_echo_hardware()
        builder = get_builder(hw)
        assert len(builder._instructions) == 0
        builder.measure(hw.get_qubit(0))
        # Adding a non-post-processing Quantum instruction between measurements forces
        # sequential measure blocks.
        builder.synchronize(hw.qubits)
        builder.measure(hw.get_qubit(2))
        assert len(builder._instructions) == 3
        assert builder._instructions[0].quantum_targets == [hw.get_qubit(0)]
        assert builder._instructions[-1].quantum_targets == [hw.get_qubit(2)]

    def test_fused_measure_block_from_builder(self):
        """Test that measurements can be fused together."""
        hw = get_default_echo_hardware()
        builder = get_builder(hw)
        assert len(builder._instructions) == 0
        builder.measure(hw.get_qubit(0))
        builder.measure(hw.get_qubit(2))
        assert len(builder._instructions) == 1
        assert builder._instructions[0].quantum_targets == [
            hw.get_qubit(0),
            hw.get_qubit(2),
        ]
