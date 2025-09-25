# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from pathlib import Path

import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader, meta

from qat.frontend.parsers.qasm import Qasm3Parser
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import (
    Delay,
    FrequencyShift,
    PhaseShift,
    QuantumInstruction,
    Reset,
    Return,
    Synchronize,
)
from qat.ir.measure import Acquire, MeasureBlock, PostProcessing, PostProcessType
from qat.ir.pulse_channel import CustomPulseChannel
from qat.ir.waveforms import (
    DragGaussianWaveform,
    GaussianSquareWaveform,
    GaussianWaveform,
    Pulse,
    SampledWaveform,
    SechWaveform,
    SinWaveform,
    SoftSquareWaveform,
    SquareWaveform,
)
from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.loaders.lucy import LucyModelLoader
from qat.utils.graphs import generate_cyclic_connectivity
from qat.utils.hardware_model import generate_hw_model
from qat.utils.qasm import get_qasm_parser

from tests.unit.utils.instruction import count_number_of_pulses
from tests.unit.utils.qasm_qir import get_default_qasm3_gate_qasms, get_qasm3, qasm3_base


@pytest.mark.parametrize("n_qubits", [8, 16, 32, 64])
class TestQasm3Parser:
    @pytest.mark.parametrize(
        "qasm_file",
        [
            "named_defcal_arg.qasm",
            "delay.qasm",
            "redefine_defcal.qasm",
            "lark_parsing_test.qasm",
            "arb_waveform.qasm",
            "tmp.qasm",
            "cx_override_test.qasm",
            "ecr_test.qasm",
            "openpulse_tests/acquire.qasm",
            "openpulse_tests/expr_list_defcal_different_arg.qasm",
            "openpulse_tests/set_frequency.qasm",
            "openpulse_tests/freq.qasm",
            "openpulse_tests/constant_wf.qasm",
            "openpulse_tests/detune_gate.qasm",
            "openpulse_tests/zmap.qasm",
            "openpulse_tests/expr_list_caldef_1.qasm",
            "openpulse_tests/expr_list_caldef_2.qasm",
            "openpulse_tests/expr_list_caldef_3.qasm",
            "openpulse_tests/waveform_numerical_types.qasm",
            "openpulse_tests/cross_ressonance.qasm",
            "openpulse_tests/shift_phase.qasm",
            "waveform_tests/waveform_test_scale.qasm",
            "waveform_tests/internal_waveform_tests.qasm",
            "waveform_tests/waveform_test_phase_shift.qasm",
            "waveform_tests/waveform_test_sum.qasm",
        ],
    )
    def test_parsing_instructions(self, qasm_file, n_qubits):
        connectivity = generate_cyclic_connectivity(n_qubits)
        hw_builder = PhysicalHardwareModelBuilder(physical_connectivity=connectivity)
        hw = hw_builder.model

        parser = Qasm3Parser()
        qasm = get_qasm3(qasm_file)
        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)
        assert len(builder.instructions) > 0

    @pytest.mark.parametrize("seed", [1, 10, 100])
    def test_expr_list_defcal_different_arg(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        parser = Qasm3Parser()
        builder = parser.parse(
            QuantumInstructionBuilder(hardware_model=hw),
            get_qasm3("openpulse_tests/expr_list_defcal_different_arg.qasm"),
        )
        instructions = builder.instructions
        assert len(instructions) > 2
        for instruction in instructions:
            if isinstance(instruction, Pulse):
                assert not isinstance(instruction.waveform, SoftSquareWaveform)

    def test_ghz(self, n_qubits):
        connectivity = generate_cyclic_connectivity(n_qubits)
        hw_builder = PhysicalHardwareModelBuilder(physical_connectivity=connectivity)
        hw = hw_builder.model

        v3_qasm = get_qasm3("ghz.qasm")
        v2_qasm = get_qasm3("ghz_v2.qasm")

        v3_builder = get_qasm_parser(v3_qasm).parse(
            QuantumInstructionBuilder(hardware_model=hw), v3_qasm
        )
        v2_builder = get_qasm_parser(v2_qasm).parse(
            QuantumInstructionBuilder(hardware_model=hw), v2_qasm
        )

        assert v3_builder.number_of_instructions == v2_builder.number_of_instructions

    def test_complex_gates(self, n_qubits):
        hw = generate_hw_model(n_qubits)
        qasm = get_qasm3("complex_gates_test.qasm")
        parser = Qasm3Parser()

        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)

        assert builder.number_of_instructions > 0

    def test_ecr(self, n_qubits):
        connectivity = generate_cyclic_connectivity(n_qubits)
        hw_builder = PhysicalHardwareModelBuilder(physical_connectivity=connectivity)
        hw = hw_builder.model

        qasm = get_qasm3("ecr_test.qasm")
        parser = Qasm3Parser()

        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)
        assert count_number_of_pulses(builder, "CrossResonance") == 2
        assert count_number_of_pulses(builder, "CrossResonanceCancellation") == 2

    def test_no_header(self, n_qubits):
        hw = generate_hw_model(n_qubits)
        qasm = get_qasm3("no_header.qasm")
        parser = Qasm3Parser()
        with pytest.raises(ValueError):
            parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)

    def test_invalid_syntax(self, n_qubits):
        hw = generate_hw_model(n_qubits)
        qasm = get_qasm3("invalid_syntax.qasm")
        parser = Qasm3Parser()
        with pytest.raises(ValueError):
            parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)

    # ("cx", "cnot") is intentional: qir parses cX as cnot, but echo engine does
    # not support cX.
    @pytest.mark.parametrize(
        "test, gate", [("cx", "cnot"), ("cnot", "cnot"), ("ecr", "ECR")]
    )
    def test_override(self, test, gate, n_qubits):
        # Tests overriding gates using openpulse: checks the overridden gate
        # yields the correct pulses, and that the unchanged gates are the same
        # as those created by the circuit builder.
        connectivity = generate_cyclic_connectivity(n_qubits)
        hw_builder = PhysicalHardwareModelBuilder(physical_connectivity=connectivity)
        hw = hw_builder.model

        parser = Qasm3Parser()
        builder = parser.parse(
            QuantumInstructionBuilder(hardware_model=hw),
            get_qasm3(f"{test}_override_test.qasm"),
        )
        qasm_inst = builder.instructions
        qasm_inst_names = [str(inst) for inst in qasm_inst]

        # test the square pulses are as expected
        ess_pulses = [
            inst
            for inst in qasm_inst
            if isinstance(inst, Pulse) and isinstance(inst.waveform, SquareWaveform)
        ]
        assert len(ess_pulses) == 2
        assert all([len(inst.targets) == 1 for inst in ess_pulses])

        # Test the ecrs on (0, 1) and (2, 3) are unchanged by the override.
        circuit = QuantumInstructionBuilder(hardware_model=hw)
        func = getattr(circuit, gate)
        func(hw.qubit_with_index(0), hw.qubit_with_index(1))
        circ_inst = circuit.instructions
        circ_inst_names = [str(inst) for inst in circ_inst]
        assert qasm_inst_names[0 : len(circ_inst_names)] == circ_inst_names

        circuit = QuantumInstructionBuilder(hardware_model=hw)
        func = getattr(circuit, gate)
        func(hw.qubit_with_index(2), hw.qubit_with_index(3))
        circ_inst = circuit.instructions
        circ_inst_names = [str(inst) for inst in circ_inst]
        assert (
            qasm_inst_names[len(qasm_inst_names) - len(circ_inst_names) :]
            == circ_inst_names
        )

    @pytest.mark.parametrize(
        "params",
        [
            ["pi", "2*pi", "-pi/2", "-7*pi/2", "0", "pi/4"],
            [np.pi, 2 * np.pi, -np.pi / 2, -7 * np.pi / 2, 0.0, np.pi / 4],
            np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(6)),
            [0.0, 0.0, 0.0, 0.0, "pi/2", 0.0],
            ["2*pi", "-pi/2", 0.0, 0.0, "pi/2", "-2*pi"],
        ],
    )
    def test_u_gate(self, params, n_qubits):
        """
        Tests the validty of the U gate with OpenPulse by checking that the
        parsed circuit matches the same circuit created with the circuit builder.
        """
        hw = generate_hw_model(n_qubits)

        # build the circuit from QASM
        qasm = get_qasm3("u_gate.qasm")
        for i in range(6):
            qasm = qasm.replace(f"param{i}", str(params[i]))
        parser = Qasm3Parser()
        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)
        qasm_inst = builder.instructions

        # create the circuit using the circuit builder
        circuit = QuantumInstructionBuilder(hardware_model=hw)
        params = [eval(str(param).replace("pi", "np.pi")) for param in params]
        q1 = hw.qubit_with_index(0)
        q2 = hw.qubit_with_index(1)
        (
            circuit.Z(q1, params[2])
            .Y(q1, params[0])
            .Z(q1, params[1])
            .Z(q2, params[5])
            .Y(q2, params[3])
            .Z(q2, params[4])
        )
        circ_inst = circuit.instructions

        # validate that the circuits match
        assert len(qasm_inst) == len(circ_inst)
        for i in range(len(qasm_inst)):
            assert str(qasm_inst[i]) == str(circ_inst[i])

    @pytest.mark.parametrize(
        "gate_tup", get_default_qasm3_gate_qasms(), ids=lambda val: val[-1]
    )
    def test_default_gates(self, gate_tup, n_qubits):
        """Check that each default gate can be parsed individually."""

        N, gate_string = gate_tup
        qasm = qasm3_base.format(N=N, gate_strings=gate_string)

        # We need a connectivity where qubits 0 and 2 are coupled for this test.
        connectivity = generate_cyclic_connectivity(n_qubits)
        connectivity[0].add(2)
        connectivity[2].add(0)
        hw_builder = PhysicalHardwareModelBuilder(physical_connectivity=connectivity)
        hw = hw_builder.model

        parser = Qasm3Parser()
        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert len(builder.instructions) > 0
        assert isinstance(builder.instructions[-1], Return)

    def test_default_gates_together(self, n_qubits):
        """Check that all default gates can be parsed together."""
        Ns, strings = zip(*get_default_qasm3_gate_qasms())
        N = max(Ns)
        gate_strings = "\n".join(strings)
        qasm = qasm3_base.format(N=N, gate_strings=gate_strings)

        # We need a connectivity where qubits 0 and 2 are coupled for this test.
        connectivity = generate_cyclic_connectivity(n_qubits)
        connectivity[0].add(2)
        connectivity[2].add(0)
        hw_builder = PhysicalHardwareModelBuilder(physical_connectivity=connectivity)
        hw = hw_builder.model

        parser = Qasm3Parser()
        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert len(builder.instructions) > 0
        assert isinstance(builder.instructions[-1], Return)

    def test_frame_creation(self, n_qubits):
        """Tests that frames can be created and used in a simple circuit."""
        hw = generate_hw_model(n_qubits)
        qasm = get_qasm3("openpulse_tests/frames.qasm")
        parser = Qasm3Parser()
        builder = parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)
        assert isinstance(builder, QuantumInstructionBuilder)
        channel_1 = hw.physical_channel_map[1]

        pulse_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Pulse)
        ]
        assert len(pulse_instructions) == 3

        uuids = set()
        channels = []
        for pulse in pulse_instructions:
            uuids.add(pulse.target)
            channel = builder.get_pulse_channel(pulse.target)
            channels.append(channel)
            assert isinstance(channel, CustomPulseChannel)

        assert len(uuids) == 3
        assert channels[0].physical_channel_id == channel_1.uuid
        assert (
            channels[1].physical_channel_id == hw.qubit_with_index(1).physical_channel.uuid
        )
        assert (
            channels[2].physical_channel_id == hw.qubit_with_index(1).physical_channel.uuid
        )


class TestQASM3Features:
    """Tests isolated features of QASM3 and OpenPulse, even those that aren't supported to
    ensure they throw errors."""

    @pytest.fixture
    def feature_testpath(self, testpath):
        """Fixture to provide the path to the test files."""
        return testpath / "files" / "qasm" / "qasm3" / "feature_tests"

    @pytest.fixture
    def model(self):
        return LucyModelLoader(qubit_count=4).load()

    @staticmethod
    def load_source(model, feature_testpath, filename, **kwargs):
        """Opens the source program using Jinja2 to template it."""

        env = Environment(loader=FileSystemLoader(feature_testpath))

        # Extract the parameters we need to provide
        template_source = env.loader.get_source(env, str(filename))[0]
        parsed_content = env.parse(template_source)
        parameters = meta.find_undeclared_variables(parsed_content)

        # Pick the parameters
        qubits = iter(model.qubits)
        for param in parameters:
            if param in kwargs:
                continue
            if param.startswith("angle"):
                kwargs[param] = np.random.uniform(-4 * np.pi, 4 * np.pi)
            elif param.startswith("physical_index"):
                kwargs[param] = next(qubits)
            elif param.startswith("lib"):
                kwargs[param] = str(feature_testpath / "oqc_lib.inc")
            elif param.startswith("frame"):
                kwargs[param] = f"q{next(qubits).index}_drive"
            else:
                raise ValueError(f"Unknown parameter {param} in template {filename}")

        template = env.get_template(str(filename))
        return template.render(**kwargs)

    @staticmethod
    def get_devices_from_builder(builder, model):
        """Extracts the devices from the builder."""
        devices = set()
        for inst in builder.instructions:
            if isinstance(inst, QuantumInstruction):
                targets = inst.targets
                for target in targets:
                    devices.add(model.device_for_pulse_channel_id(target))
        return devices

    def return_builder_and_devices(self, model, feature_testpath, filename, **kwargs):
        """Returns the builder and source code for a given QASM file."""
        qasm = self.load_source(model, feature_testpath, filename, **kwargs)
        parser = Qasm3Parser()
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder = parser.parse(builder, qasm)
        devices = self.get_devices_from_builder(builder, model)
        return builder, devices

    def test_barrier(self, model, feature_testpath):
        qubits = list(model.qubits.keys())[:2]
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "barrier.qasm"),
            physical_index_1=qubits[0],
            physical_index_2=qubits[1],
        )

        syncs = [inst for inst in builder.instructions if isinstance(inst, Synchronize)]
        assert len(syncs) == 1

        expected_targets = []
        for device in devices:
            for channel in device.all_pulse_channels:
                expected_targets.append(channel.uuid)
        assert syncs[0].targets == set(expected_targets)

    @pytest.mark.parametrize("gate", ["cx", "ecr"])
    def test_2q_gate(self, model, feature_testpath, gate):
        qubits = list(model.qubits.keys())[:2]
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", f"{gate}.qasm"),
            physical_index_1=qubits[0],
            physical_index_2=qubits[1],
        )

        # picks up extra qubits via syncs on connected qubits
        for qubit in qubits:
            assert model.qubits[qubit] in devices

    def test_gate_def(self, model, feature_testpath):
        """Overloads X gate onto identity, so we check that there are no pulses."""
        builder, _ = self.return_builder_and_devices(
            model, feature_testpath, Path("gates", "gate_def.qasm")
        )
        for inst in builder.instructions:
            assert not isinstance(inst, Pulse)

    def test_measure(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "measure.qasm"),
            physical_index_1=index,
        )
        assert qubit in devices
        assert qubit.resonator in devices

    def test_reset(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "reset.qasm"),
            physical_index_1=index,
        )
        assert qubit in devices
        assert len([inst for inst in builder.instructions if isinstance(inst, Reset)]) == 3

    def test_U(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "u.qasm"),
            physical_index_1=index,
        )
        assert qubit in devices

    def test_defcal(self, model, feature_testpath):
        """Used here to override the X gate for a qubit with a zero-delay.

        This also tests the keywords `defcalgrammar` and `extern`.
        """
        qubits = list(model.qubits.items())[0:2]
        frame = f"q{qubits[0][0]}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("keywords", "defcal.qasm"),
            physical_index_1=qubits[0][0],
            physical_index_2=qubits[1][0],
            frame=frame,
        )

        assert qubits[0][1] in devices
        assert qubits[1][1] in devices
        delays = [inst for inst in builder.instructions if isinstance(inst, Delay)]
        assert len(delays) == 1
        assert np.isclose(delays[0].duration, 0.0)
        assert delays[0].target == qubits[0][1].drive_pulse_channel.uuid

    def test_get_frequency(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "get_frequency.qasm"),
            frame=frame,
        )

        assert qubit in devices
        frequencies = [
            inst for inst in builder.instructions if isinstance(inst, FrequencyShift)
        ]
        assert len(frequencies) == 1
        channel = qubit.drive_pulse_channel
        assert frequencies[0].target == channel.uuid
        assert np.isclose(frequencies[0].frequency, 0.05 * channel.frequency)

    def test_shift_frequency(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "shift_frequency.qasm"),
            frame=frame,
            physical_index=index,
            frequency=channel.frequency * 0.05,
        )

        assert qubit in devices
        frequencies = [
            inst for inst in builder.instructions if isinstance(inst, FrequencyShift)
        ]
        assert len(frequencies) == 1
        assert frequencies[0].target == channel.uuid
        assert np.isclose(frequencies[0].frequency, 0.05 * channel.frequency)

    def test_set_frequency(self, model, feature_testpath):
        """`set_frequency` actually adds a shift, so test for that."""
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "set_frequency.qasm"),
            frame=frame,
            physical_index=index,
            frequency=channel.frequency * 1.05,
        )

        assert qubit in devices
        frequencies = [
            inst for inst in builder.instructions if isinstance(inst, FrequencyShift)
        ]
        assert len(frequencies) == 1
        assert frequencies[0].target == channel.uuid
        assert np.isclose(frequencies[0].frequency, 0.05 * channel.frequency)

    def test_getphase(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "get_phase.qasm"),
            frame=frame,
        )

        assert qubit in devices
        phases = [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        assert len(phases) == 2
        for phase in phases:
            assert np.isclose(phase.phase, np.pi)
            assert phase.target == channel.uuid

    def test_shift_phase(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "shift_phase.qasm"),
            frame=frame,
            physical_index=index,
            phase=2.0,
        )

        assert qubit in devices
        phases = [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        assert len(phases) == 1
        assert phases[0].target == channel.uuid
        assert np.isclose(phases[0].phase, 0.254)

    def test_set_phase(self, model, feature_testpath):
        """`set_phase` actually adds a shift, so test for that."""
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "set_phase.qasm"),
            frame=frame,
            physical_index=index,
            phase=0.254,
        )

        assert qubit in devices
        phases = [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        assert len(phases) == 2
        assert phases[0].target == channel.uuid
        assert phases[1].target == channel.uuid
        assert np.isclose(phases[0].phase, 0.5)
        assert np.isclose(phases[1].phase, 0.254 - 0.5)

    def test_newframe(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "newframe.qasm"),
            frame=frame,
            physical_index=index,
            frequency=channel.frequency * 1.05,
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 2
        assert pulses[0].target == channel.uuid
        assert pulses[1].target != channel.uuid

        pulse_channel_1 = builder.get_pulse_channel(pulses[1].target)
        assert isinstance(pulse_channel_1, CustomPulseChannel)
        assert pulse_channel_1.physical_channel_id == qubit.physical_channel.uuid
        assert np.isclose(pulse_channel_1.frequency, channel.frequency * 1.05)

    @pytest.mark.parametrize(
        "version",
        [
            pytest.param(
                "0",
                marks=pytest.mark.xfail(
                    reason="Version 0 is not supported.", raises=NotImplementedError
                ),
            ),
            "1",
            "2",
            "3",
            pytest.param(
                "4",
                marks=pytest.mark.xfail(
                    reason="Version 4 is not supported.", raises=NotImplementedError
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("channel", ["acquire", "measure"])
    def test_capture(self, model, feature_testpath, version, channel):
        index, qubit = next(iter(model.qubits.items()))
        # TODO: this usage is inconsistent with extern frames in purr. We should probably
        # align them (COMPILER-716)
        if channel == "measure":
            frame = f"r{index}_measure"
            channel = qubit.measure_pulse_channel
        else:
            frame = f"r{index}_acquire"
            channel = qubit.acquire_pulse_channel

        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "capture.qasm"),
            frame=frame,
            physical_index=index,
            capture_version=version,
        )

        assert qubit.resonator in devices
        acquires = [inst for inst in builder.instructions if isinstance(inst, Acquire)]
        assert len(acquires) == 1
        assert acquires[0].target == channel.uuid

        acquire_name = acquires[0].output_variable
        pps = [inst for inst in builder.instructions if isinstance(inst, PostProcessing)]

        if version == 1:
            # Just returns IQ values: requires no PP
            assert len(pps) == 0
        elif version == 2:
            # returns discriminated bit
            assert len(pps) == 1
            assert pps[0].output_variable == acquire_name
            assert pps[0].process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        elif version == 3:
            # SCOPE mode
            assert len(pps) == 2
            assert pps[0].output_variable == acquire_name
            assert pps[1].output_variable == acquire_name
            assert pps[0].process_type == PostProcessType.MEAN
            assert pps[1].process_type == PostProcessType.DOWN_CONVERT

    def test_delay(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "delay.qasm"),
            frame=frame,
            physical_index=index,
            time="80ns",
        )

        assert qubit in devices
        delays = [inst for inst in builder.instructions if isinstance(inst, Delay)]
        assert len(delays) == 1
        assert delays[0].target == channel.uuid
        assert np.isclose(delays[0].duration, 80e-9)

    def test_play(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "play.qasm"),
            frame=frame,
            physical_index=index,
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid

    def test_arb_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        samples = np.random.rand(160) + 1j * np.random.rand(160)
        samples_as_str = ", ".join(
            [f"{sample.real} + {sample.imag}im" for sample in samples]
        )
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "arb.qasm"),
            frame=frame,
            samples=samples_as_str,
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert isinstance(pulses[0].waveform, SampledWaveform)
        assert np.allclose(pulses[0].waveform.samples, samples)

    def test_constant_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "constant.qasm"),
            frame=frame,
            amp=1e-4,
            width="80ns",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid
        assert np.isclose(pulses[0].duration, 80e-9)
        assert isinstance(pulses[0].waveform, SquareWaveform)

    def test_drag_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "drag.qasm"),
            frame=frame,
            width="80ns",
            amp="1e-7",
            sigma="20ns",
            beta="0.05",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid
        assert np.isclose(pulses[0].duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, DragGaussianWaveform)
        assert np.isclose(waveform.std_dev, 20e-9)
        assert np.isclose(waveform.beta, 0.05)

    def test_gaussian_square_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "gaussian_square.qasm"),
            frame=frame,
            width="80ns",
            amp="1e-4",
            square_width="40ns",
            sigma="20ns",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid
        assert np.isclose(pulses[0].duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, GaussianSquareWaveform)
        assert np.isclose(waveform.std_dev, 20e-9)
        assert np.isclose(waveform.square_width, 40e-9)

    def test_gaussian_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "gaussian.qasm"),
            frame=frame,
            width="80ns",
            amp="1e-4",
            sigma="20ns",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid
        assert np.isclose(pulses[0].duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, GaussianWaveform)
        assert np.isclose(waveform.std_dev, 20e-9)

    def test_sech_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "sech.qasm"),
            frame=frame,
            width="80ns",
            amp="1e-4",
            sigma="20ns",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid
        assert np.isclose(pulses[0].duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, SechWaveform)
        assert np.isclose(waveform.std_dev, 20e-9)

    def test_sine_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        channel = qubit.drive_pulse_channel
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "sine.qasm"),
            frame=frame,
            width="80ns",
            amp="1e-4",
            frequency="1e9",
            phase="0.254",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert pulses[0].target == channel.uuid
        assert np.isclose(pulses[0].duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, SinWaveform)
        assert np.isclose(waveform.frequency, 1e9)
        assert np.isclose(waveform.phase, 0.254)

    def test_mix_waveform(self, model, feature_testpath):
        # Also expecting this to fail because of the hard-coded sample time, so the reason
        # and raises might just need changing after fixing...
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        amp1 = 2.5e-4
        amp2 = 0.5
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "mix.qasm"),
            frame=frame,
            width="80ns",
            amp1=str(amp1),
            amp2=str(amp2),
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1

        pulse = pulses[0]

        waveform = pulse.waveform
        assert isinstance(waveform, SampledWaveform)
        assert np.allclose(waveform.samples, amp1 * amp2)

    def test_phase_shift_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "phase_shift.qasm"),
            frame=frame,
            width="80ns",
            amp="1e-4",
            phase="0.254",
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1

        pulse = pulses[0]
        assert np.isclose(pulse.duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, SquareWaveform)
        assert np.isclose(waveform.phase, 0.254)

    def test_scale_waveform(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        amp1 = 2.5e-4
        amp2 = 0.5
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "scale.qasm"),
            frame=frame,
            width="80ns",
            amp1=str(amp1),
            amp2=str(amp2),
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)

        waveform = pulses[0].waveform
        assert isinstance(waveform, SquareWaveform)
        assert np.isclose(waveform.amp, amp1)
        assert np.isclose(waveform.scale_factor, amp2)

    def test_sum_waveform(self, model, feature_testpath):
        # Also expecting this to fail because of the hard-coded sample time, so the reason
        # and raises might just need changing after fixing...
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        amp1 = 2.5e-4
        amp2 = 0.5
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "sum.qasm"),
            frame=frame,
            width="80ns",
            amp1=str(amp1),
            amp2=str(amp2),
        )

        assert qubit in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 1

        pulse = pulses[0]

        waveform = pulse.waveform
        assert np.allclose(waveform.samples, amp1 + amp2)

    def test_type_physical_index(self, model, feature_testpath):
        index, qubit = next(iter(model.qubits.items()))
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("types", "physical_index.qasm"),
            physical_index=index,
        )
        assert qubit in devices
        assert qubit.resonator in devices

    @pytest.mark.parametrize("device_type", ["qubit", "resonator"])
    def test_type_port(self, model, feature_testpath, device_type):
        """Just a basic smoke test as there's nothing to really inspect."""
        device = next(iter(model.qubits.values()))
        if device_type == "resonator":
            device = device.resonator
        index = device.physical_channel.name_index
        _ = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("types", "port.qasm"),
            port=f"channel_{index}",
        )

    @pytest.mark.parametrize(
        "width", ["80ns", "0.08us", "0.08µs", "80e-9s", "160dt", "80e-6ms"]
    )
    def test_units(self, model, feature_testpath, width):
        """Test that different units are parsed correctly."""
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "delay.qasm"),
            frame=frame,
            time=width,
        )

        assert qubit in devices
        delays = [inst for inst in builder.instructions if isinstance(inst, Delay)]
        assert len(delays) == 1
        assert np.isclose(delays[0].duration, 80e-9)

    @pytest.mark.parametrize(
        "constant, value",
        [
            ("pi", np.pi),
            ("euler", np.e),
            ("tau", 2 * np.pi),
            # unicode symbols for constants
            ("\u03c0", np.pi),
            ("\u2107", np.e),
            ("\U0001d70f", 2 * np.pi),
        ],
    )
    def test_constants(self, model, feature_testpath, constant, value):
        """Test that constants are parsed correctly."""
        index, qubit = next(iter(model.qubits.items()))
        frame = f"q{index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("constants.qasm"),
            frame=frame,
            value=constant,
        )

        assert qubit in devices
        phases = [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        assert len(phases) == 1
        assert np.isclose(phases[0].phase, value)

    @pytest.mark.xfail(raises=ValueError, reason="Control flow is not supported yet.")
    @pytest.mark.parametrize(
        "file",
        [
            "break.qasm",
            "continue.qasm",
            "for.qasm",
            "functions.qasm",
            "if_else.qasm",
            "while.qasm",
        ],
    )
    def test_control_flow(self, model, feature_testpath, file):
        """More precise testing of control flow to come when control flow is supported."""
        file = Path("classical", file)
        self.return_builder_and_devices(model, feature_testpath, file)

    @pytest.mark.xfail(raises=ValueError, reason="Gate modifiers are not supported yet.")
    @pytest.mark.parametrize(
        "file",
        [
            "ctrl.qasm",
            "inv.qasm",
            "negctrl.qasm",
            "pow.qasm",
        ],
    )
    def test_gate_modifiers(self, model, feature_testpath, file):
        """More precise testing of gate modifiers to come when they are supported."""
        file = Path("gates", "modifiers", file)
        self.return_builder_and_devices(model, feature_testpath, file)

    @pytest.mark.xfail(
        raises=ValueError, reason="Constant declarations are not supported yet."
    )
    def test_constant(self, model, feature_testpath):
        """More precise testing of constant declarations to come when they are supported."""
        self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("keywords", "constant.qasm"),
        )

    @pytest.mark.xfail(
        raises=ValueError, reason="Function definitions are not supported yet."
    )
    def test_def(self, model, feature_testpath):
        """More precise testing of function definitions to come when they are supported."""
        self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("keywords", "def.qasm"),
        )

    @pytest.mark.xfail(raises=ValueError, reason="Durationof is not supported yet.")
    def test_durationof(self, model, feature_testpath):
        """More precise testing of durationof to come when supported."""
        index = next(iter(model.qubits))
        frame = f"q{index}_drive"
        self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "durationof.qasm"),
            frame=frame,
            physical_index=index,
        )

    @pytest.mark.xfail(raises=ValueError, reason="This types is not fully supported yet.")
    @pytest.mark.parametrize(
        "file",
        [
            "angle.qasm",
            "array.qasm",
            "complex.qasm",
            "duration.qasm",
            "float.qasm",
            "int.qasm",
            "stretch.qasm",
        ],
    )
    def test_types(self, model, feature_testpath, file):
        """More precise testing of types to come when they are supported."""
        file = Path("types", file)
        self.return_builder_and_devices(
            model,
            feature_testpath,
            file,
        )

    def test_include(self, model, feature_testpath):
        lib = str(feature_testpath / "oqc_lib.inc")
        builder, devices = self.return_builder_and_devices(
            model, feature_testpath, Path("include.qasm"), lib=lib, physical_index=0
        )

        assert model.qubit_with_index(0) in devices
        assert model.qubit_with_index(0).resonator in devices
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        assert len(pulses) == 2
        for pulse in pulses:
            assert pulse.target == model.qubit_with_index(0).drive_pulse_channel.uuid
        measure_blocks = [
            inst for inst in builder.instructions if isinstance(inst, MeasureBlock)
        ]
        assert len(measure_blocks) == 1
