# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2025 Oxford Quantum Circuits Ltd
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    Qasm2Optimizations,
    Qasm3Optimizations,
    QuantumResultsFormat,
)
from jinja2 import Environment, FileSystemLoader, meta
from pytket.qasm import circuit_from_qasm_str

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import (
    get_default_RTCS_hardware,
    qutip_available,
)
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    CrossResonancePulse,
    CustomPulse,
    Delay,
    FrequencyShift,
    Instruction,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    Pulse,
    QuantumInstruction,
    Reset,
    Return,
    Synchronize,
)
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import (
    CloudQasmParser,
    Qasm2Parser,
    Qasm3Parser,
    Qasm3ParserBase,
    QasmContext,
    RestrictedQasm2Parser,
    get_qasm_parser,
)
from qat.purr.integrations.tket import TketBuilder, TketQasmParser
from qat.purr.qat import execute_qasm
from qat.purr.utils.serializer import json_load, json_loads

from tests.unit.utils.matrix_builder import (
    Gates,
    assert_same_up_to_phase,
    get_default_matrix_hardware,
)
from tests.unit.utils.models import get_jagged_echo_hardware, update_qubit_indices
from tests.unit.utils.qasm_qir import (
    get_all_qasm2_paths,
    get_default_qasm2_gate_qasms,
    get_default_qasm3_gate_qasms,
    get_qasm2,
    get_qasm3,
    parse_and_apply_optimizations,
    qasm2_base,
    qasm3_base,
)


class TestQASM2:
    qasm2_base = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[{N}];
    {gate_strings}
    """

    @pytest.mark.parametrize(
        "gate", Qasm2Parser()._get_intrinsics(), ids=lambda val: val.name
    )
    def test_qasm2_gates(self, gate):
        """Check that each QASM2 gate can be parsed individually."""
        # Skip some gates
        if gate.name == "rccx" or gate.name == "rc3x":
            pytest.skip("Gate is defined by its decompositions. Difficult to compare.")
        elif gate.name == "delay":
            pytest.skip("Delay not implemented.")
        elif gate.name == "id" or gate.name == "u0":
            pytest.skip(f"Gate {gate.name} isn't intrinsic and has no body.")

        # Create an assortment of parameters
        thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.321, -1.58]

        if gate.num_params > 0:
            args_list = product(*[thetas for _ in range(gate.num_params)])
        else:
            args_list = [tuple()]

        for args in args_list:
            # Contruct the qasm gate.
            gate_string = gate.name
            if len(args) > 0:
                gate_string += "(" + ", ".join([str(arg) for arg in args]) + ")"
            gate_string += (
                " " + ", ".join([f"q[{i}]" for i in range(gate.num_qubits)]) + ";"
            )
            qasm = self.qasm2_base.format(N=gate.num_qubits, gate_strings=gate_string)

            # Parse it through a fully conencted hardware and verify result.
            connectivity = list(nx.complete_graph(gate.num_qubits).edges)
            hw = get_default_matrix_hardware(gate.num_qubits, connectivity=connectivity)
            parser = Qasm2Parser()
            builder = parser.parse(hw.create_builder(), qasm)
            gate_method = getattr(Gates, gate.name)
            actual_gate = gate_method(*args)
            assert_same_up_to_phase(builder.matrix, actual_gate)


def qasm3_gates():
    context = QasmContext()
    Qasm3ParserBase().load_default_gates(context)
    return context.gates


class TestQASM3:
    """
    Tests for the parsing and instruction generation of OPENQASM3.0.

    In particular these tests currently only operate on a subset of OQ3
    and focus on the OpenPulse side of the language definition.
    """

    @pytest.mark.legacy
    def test_named_defcal_arg(self):
        hw = get_default_echo_hardware(8)
        comp = CompilerConfig()
        comp.results_format.binary_count()
        results = execute_qasm(get_qasm3("named_defcal_arg.qasm"), hw, compiler_config=comp)
        results = next(iter(results.values()), dict())
        assert len(results) == 1
        assert results["00"] == 1000

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    @pytest.mark.legacy
    def test_basic(self):
        hw = get_default_RTCS_hardware()
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute_qasm(get_qasm3("basic.qasm"), hw, compiler_config=config)
        assert len(results["c"]) == 4

        # Assert the distribution is mostly correct. Don't care about absolute accuracy,
        # just that it's spread equally.
        assert not any([val for val in results["c"].values() if (val / 1000) < 0.15])

    @pytest.mark.legacy
    def test_zmap(self):
        hw = get_default_echo_hardware(8)
        results = execute_qasm(get_qasm3("openpulse_tests/zmap.qasm"), hw)
        assert not isinstance(results, dict)

    @pytest.mark.legacy
    def test_frequency(self):
        hw = get_default_echo_hardware(8)
        execute_qasm(get_qasm3("openpulse_tests/freq.qasm"), hw)

    @pytest.mark.parametrize(
        "arg_count",
        [1, 2, 3],  # 2 includes a generic qubit def, should be separate test
    )
    def test_expr_list_defcal(self, arg_count):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw),
            get_qasm3(f"openpulse_tests/expr_list_caldef_{arg_count}.qasm"),
        )
        instructions = result.instructions
        # There is a sync instruction from the implicit defcal barrier first
        assert len(instructions) == 2
        assert instructions[1].shape.value == PulseShapeType.SOFT_SQUARE.value

    def test_expr_list_defcal_different_arg(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw),
            get_qasm3("openpulse_tests/expr_list_defcal_different_arg.qasm"),
        )
        instructions = result.instructions
        assert len(instructions) > 2
        for instruction in instructions:
            if isinstance(instruction, Pulse):
                assert instruction.shape.value != PulseShapeType.SOFT_SQUARE.value

    def test_ghz(self):
        hw = get_default_echo_hardware(4)
        v3_qasm = get_qasm3("ghz.qasm")
        v2_qasm = get_qasm3("ghz_v2.qasm")

        v3_instructions = get_qasm_parser(v3_qasm).parse(get_builder(hw), v3_qasm)
        v2_instructions = get_qasm_parser(v2_qasm).parse(get_builder(hw), v2_qasm)

        assert len(v3_instructions.instructions) == len(v2_instructions.instructions)

    @pytest.mark.legacy
    def test_complex_gates(self):
        hw = get_default_echo_hardware(8)
        execute_qasm(get_qasm3("complex_gates_test.qasm"), hw)

    @pytest.mark.legacy
    def test_execution(self):
        hw = get_default_echo_hardware(8)
        execute_qasm(get_qasm3("lark_parsing_test.qasm"), hw)

    @pytest.mark.legacy
    def test_invalid_qasm_version(self):
        hw = get_default_echo_hardware(8)
        with pytest.raises(ValueError) as context:
            execute_qasm(get_qasm3("invalid_version.qasm"), hw)

        # Assert we've errored and the language is mentioned
        message = str(context.value)
        assert "No valid parser could be found" in message
        assert "Qasm2" in message
        assert "Qasm3" in message

    def test_parsing(self):
        hw = get_default_echo_hardware(8)
        parser = Qasm3Parser()
        parser.parse(get_builder(hw), get_qasm3("lark_parsing_test.qasm"))

    def test_no_header(self):
        hw = get_default_echo_hardware(8)
        parser = Qasm3Parser()
        with pytest.raises(ValueError):
            parser.parse(get_builder(hw), get_qasm3("no_header.qasm"))

    def test_invalid_syntax(self):
        hw = get_default_echo_hardware(8)
        parser = Qasm3Parser()
        with pytest.raises(ValueError):
            parser.parse(get_builder(hw), get_qasm3("invalid_syntax.qasm"))

    @pytest.mark.skip("Test incomplete.")
    def test_wave_forms(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("waveform_test_sum.qasm"))
        instructions = result.instructions
        assert len(instructions) == 2
        assert instructions[0].shape.value == PulseShapeType.GAUSSIAN

    def test_ecr(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("ecr_test.qasm"))
        assert any(isinstance(inst, CrossResonancePulse) for inst in result.instructions)

    # ("cx", "cnot") is intentional: qir parses cX as cnot, but echo engine does
    # not support cX.
    @pytest.mark.parametrize(
        "test, gate", [("cx", "cnot"), ("cnot", "cnot"), ("ecr", "ECR")]
    )
    def test_override(self, test, gate):
        # Tests overriding gates using openpulse: checks the overridden gate
        # yields the correct pulses, and that the unchanged gates are the same
        # as those created by the circuit builder.
        hw = get_default_echo_hardware(4)
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3(f"{test}_override_test.qasm"))
        qasm_inst = result.instructions
        qasm_inst_names = [str(inst) for inst in qasm_inst]

        # test the extra_soft_square pulses are as expected
        ess_pulses = [
            inst
            for inst in qasm_inst
            if hasattr(inst, "shape") and (inst.shape is PulseShapeType.SQUARE)
        ]
        assert len(ess_pulses) == 2
        assert all([len(inst.quantum_targets) == 1 for inst in ess_pulses])
        assert ess_pulses[0].quantum_targets[0].partial_id() == "q1_frame"
        assert ess_pulses[1].quantum_targets[0].partial_id() == "q2_frame"

        # test the ecrs on (0, 1) and (2, 3) are unchanged by the override
        circuit = hw.create_builder()
        func = getattr(circuit, gate)
        func(hw.get_qubit(0), hw.get_qubit(1))
        circ_inst = circuit.instructions
        circ_inst_names = [str(inst) for inst in circ_inst]
        assert qasm_inst_names[0 : len(circ_inst_names)] == circ_inst_names

        circuit = hw.create_builder()
        func = getattr(circuit, gate)
        func(hw.get_qubit(2), hw.get_qubit(3))
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
    def test_u_gate(self, params):
        """
        Tests the validty of the U gate with OpenPulse by checking that the
        parsed circuit matches the same circuit created with the circuit builder.
        """
        hw = get_default_echo_hardware(2)

        # build the circuit from QASM
        qasm = get_qasm3("u_gate.qasm")
        for i in range(6):
            qasm = qasm.replace(f"param{i}", str(params[i]))
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), qasm)
        qasm_inst = result.instructions

        # create the circuit using the circuit builder
        builder = hw.create_builder()
        params = [eval(str(param).replace("pi", "np.pi")) for param in params]
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)
        (
            builder.Z(q1, params[2])
            .Y(q1, params[0])
            .Z(q1, params[1])
            .Z(q2, params[5])
            .Y(q2, params[3])
            .Z(q2, params[4])
        )
        circ_inst = builder.instructions

        # validate that the circuits match
        assert len(qasm_inst) == len(circ_inst)
        for i in range(len(qasm_inst)):
            assert str(qasm_inst[i]) == str(circ_inst[i])

    def test_invalid_frames(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        with pytest.raises(ValueError) as context:
            parser.parse(get_builder(hw), get_qasm3("invalid_frames.qasm"))
        assert "q42_drive" in str(context.value)

    def test_invalid_port(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        with pytest.raises(ValueError) as context:
            parser.parse(get_builder(hw), get_qasm3("invalid_port.qasm"))
        assert "channel_42" in str(context.value)

    def test_invalid_waveform(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        with pytest.raises(ValueError) as context:
            parser.parse(get_builder(hw), get_qasm3("invalid_waveform.qasm"))
        assert "waves_for_days" in str(context.value)

    def test_arb_waveform(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("arb_waveform.qasm"))
        pluses = [val for val in result.instructions if isinstance(val, CustomPulse)]
        pulse: CustomPulse = next(iter(pluses), None)

        assert len(pluses) == 1
        assert pulse is not None
        assert pulse.samples == [1, 1j]

    def test_delay(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("delay.qasm"))
        delays = [val for val in result.instructions if isinstance(val, Delay)]
        delay: Delay = next(iter(delays), None)

        assert len(delays) == 1
        assert delay is not None
        assert delay.time == 42e-9

    def test_redefine_cal(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("redefine_defcal.qasm"))
        pulses = [val for val in result.instructions if isinstance(val, Pulse)]
        assert len(pulses) == 2
        assert pulses[0].shape is PulseShapeType.GAUSSIAN_ZERO_EDGE
        assert pulses[1].shape is PulseShapeType.SQUARE

    @pytest.mark.legacy
    def test_excessive_pulse_width_fails(self):
        hw = get_default_echo_hardware(8)
        with pytest.raises(ValueError):
            execute_qasm(get_qasm3("invalid_pulse_length.qasm"), hardware=hw)
        # pulse width within limits
        execute_qasm(get_qasm3("redefine_defcal.qasm"), hardware=hw)

    @pytest.mark.skip(
        "Double-check that you should be able to call gates this way. Seems dubious."
    )
    def test_simultaneous_frame_gates_fail(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openpulse_tests/frame_collision_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) > 0

    @pytest.mark.skip(
        "Timing isn't considered in pulse definition, only after scheduling. Fix or remove."
    )
    def test_pulse_timing(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openqasm_tests/gate_timing_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1

    @pytest.mark.skip(
        "Timing isn't considered in pulse definition, only after scheduling. Fix or remove."
    )
    def test_barrier_timing(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openqasm_tests/barrier_timing_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1

    @pytest.mark.skip("Test incomplete.")
    def test_phase_tracking(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openpulse_tests/phase_tracking_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1
        assert isinstance(instruction[0], Delay)
        assert np.isclose(instruction[0].time, 42e-9)

    @pytest.mark.skip("Test incomplete.")
    def test_cross_res(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openpulse_tests/cross_resonance.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1
        assert isinstance(instruction[0], Delay)
        assert np.isclose(instruction[0].time, 42e-9)

    @pytest.mark.parametrize(
        "file_name,test_value",
        (
            ("sum", 5e-3),
            ("mix", 6e-4),
        ),
    )
    def test_waveform_processing(self, file_name, test_value):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3(f"waveform_tests/waveform_test_{file_name}.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 2
        assert np.allclose(instruction[1].samples, test_value)

    @pytest.mark.parametrize(
        "file_name,attribute,test_value",
        (("scale", "scale_factor", 42), ("phase_shift", "phase", 4 + 2j)),
    )
    def test_waveform_processing_single_waveform(self, file_name, attribute, test_value):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3(f"waveform_tests/waveform_test_{file_name}.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 2
        assert getattr(instruction[1], attribute) == test_value

    @pytest.mark.parametrize(
        "qasm_name",
        [
            "acquire",
            "constant_wf",
            "cross_ressonance",
            "detune_gate",
            "set_frequency",
            "shift_phase",
            "waveform_numerical_types",
        ],
    )
    @pytest.mark.legacy
    def test_op(self, qasm_name):
        qasm_string = get_qasm3(f"openpulse_tests/{qasm_name}.qasm")
        hardware = get_default_echo_hardware(8)
        config = CompilerConfig(
            repeats=10,
            results_format=QuantumResultsFormat(),
            optimizations=Qasm3Optimizations(),
        )
        assert execute_qasm(qasm_string, hardware=hardware, compiler_config=config)

    @pytest.mark.legacy
    def test_internal_pulses(self):
        qasm_string = get_qasm3("waveform_tests/internal_waveform_tests.qasm")
        hardware = get_default_echo_hardware(8)
        config = CompilerConfig(
            repeats=10,
            results_format=QuantumResultsFormat(),
            optimizations=Qasm3Optimizations(),
        )
        assert execute_qasm(qasm_string, hardware=hardware, compiler_config=config)

    @pytest.mark.parametrize(
        "qasm_file",
        [
            "basic.qasm",
            "ghz.qasm",
        ],
    )
    def test_on_jagged_hardware(self, qasm_file):
        hw = get_jagged_echo_hardware(8)
        qasm_string = get_qasm3(qasm_file)
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), qasm_string)
        assert len(result.instructions) > 0

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
    def test_dollar_on_jagged_hardware(self, qasm_file):
        hw = get_jagged_echo_hardware(8)
        parser = Qasm3Parser()
        qasm_string = get_qasm3(qasm_file)
        with pytest.raises(ValueError):
            parser.parse(get_builder(hw), qasm_string)
        qasm_string = update_qubit_indices(qasm_string, [q.index for q in hw.qubits])
        result = parser.parse(get_builder(hw), qasm_string)
        assert len(result.instructions) > 0

    @pytest.mark.legacy
    def test_execute_different_qat_input_types(self):
        hw = get_default_echo_hardware(5)
        qubit = hw.get_qubit(0)
        phase_shift_1 = 0.2
        phase_shift_2 = 0.1
        builder = (
            get_builder(hw)
            .phase_shift(qubit, phase_shift_1)
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, phase_shift_2)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit)
        )

        with pytest.raises(TypeError):
            execute_qasm(qat_input=builder.instructions, hardware=hw)

    @pytest.mark.parametrize(
        "hw", [get_default_echo_hardware(2), get_default_RTCS_hardware()]
    )
    def test_capture_with_delay(self, hw):
        # Tests that capture v2 in openpulse makes use of the qubit delay for an acquire channel.
        qubit = hw.get_qubit(0)
        parser = Qasm3Parser()
        builder = parser.parse(
            hw.create_builder(), get_qasm3("openpulse_tests/capture.qasm")
        )
        delay = [inst.delay for inst in builder.instructions if isinstance(inst, Acquire)]
        assert delay[0] == qubit.measure_acquire["delay"]

    def test_gaussian_square(self):
        # Checks that the Gaussian Square pulses parse correectly.
        hw = get_default_echo_hardware(2)
        qasm_string = get_qasm3("waveform_tests/gaussian_square.qasm")
        parser = Qasm3Parser()
        builder = parser.parse(get_builder(hw), qasm_string)
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        # Check the properties of the first pulse
        assert np.isclose(pulses[0].width, 100e-9)
        assert np.isclose(pulses[0].amp, 1)
        assert np.isclose(pulses[0].square_width, 50e-9)
        assert pulses[0].zero_at_edges == True
        # Check the properties of the second pulse
        assert np.isclose(pulses[1].width, 200e-9)
        assert np.isclose(pulses[1].amp, 2.5)
        assert np.isclose(pulses[1].square_width, 50e-9)
        assert pulses[1].zero_at_edges == False

    def test_sech(self):
        # Checks that the sech waveforms parse correctly.
        hw = get_default_echo_hardware(2)
        qasm_string = get_qasm3("waveform_tests/sech_waveform.qasm")
        parser = Qasm3Parser()
        builder = parser.parse(get_builder(hw), qasm_string)
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        # Check the properties of the first pulse
        assert np.isclose(pulses[0].width, 100e-9)
        assert np.isclose(pulses[0].amp, 0.2)
        assert np.isclose(pulses[0].std_dev, 50e-9)
        # Check the properties of the second pulse
        assert np.isclose(pulses[1].width, 200e-9)
        assert np.isclose(pulses[1].amp, 0.5)
        assert np.isclose(pulses[1].std_dev, 20e-9)

    @pytest.mark.parametrize(
        "gate_tup", get_default_qasm3_gate_qasms(), ids=lambda val: val[-1]
    )
    def test_default_gates(self, gate_tup, monkeypatch, testpath):
        """Check that each default gate can be parsed individually."""

        def equivalent(self, other):
            return isinstance(self, type(other)) and (vars(self) == vars(other))

        monkeypatch.setattr(Instruction, "__eq__", equivalent)

        N, gate_string = gate_tup
        file_name = gate_string.split(" ")[0].split("(")[0] + ".json"
        qasm = qasm3_base.format(N=N, gate_strings=gate_string)
        hw = get_default_echo_hardware(
            N, [(i, j) for i in range(N) for j in range(i, N) if i != j]
        )
        parser = Qasm3Parser()
        builder = parser.parse(hw.create_builder(), qasm)
        assert isinstance(builder, InstructionBuilder)
        with Path(testpath, "files", "qasm", "instructions", file_name).open("r") as f:
            expectations = [json_loads(i, model=hw) for i in json_load(f)]
        assert len(builder.instructions) > 0
        assert isinstance(builder.instructions[-1], Return)
        for instruction, expected in zip(builder.instructions, expectations):
            assert expected == instruction

    def test_default_gates_together(self):
        """Check that all default gates can be parsed together."""
        Ns, strings = zip(*get_default_qasm3_gate_qasms())
        N = max(Ns)
        gate_strings = "\n".join(strings)
        qasm = qasm3_base.format(N=N, gate_strings=gate_strings)
        hw = get_default_echo_hardware(max(N, 2))
        parser = Qasm3Parser()
        builder = parser.parse(hw.create_builder(), qasm)
        assert isinstance(builder, InstructionBuilder)
        assert len(builder.instructions) > 0
        assert isinstance(builder.instructions[-1], Return)

    qasm3_base = """
    OPENQASM 3.0;
    qreg q[{N}];
    {gate_strings}
    """

    @pytest.mark.parametrize(
        ["name", "gate"],
        qasm3_gates().items(),
    )
    def test_qasm3_gates(self, name, gate):
        """Check that each QASM3 gate can be parsed individually."""
        # Create an assortment of parameters
        thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.321, -1.58]
        num_params = len(gate.arguments)
        qubits = len(gate.qubits)
        if num_params > 0:
            args_list = product(*[thetas for _ in range(num_params)])
        else:
            args_list = [tuple()]

        for args in args_list:
            # contruct the qasm qate
            gate_string = name
            if len(args) > 0:
                gate_string += "(" + ", ".join([str(arg) for arg in args]) + ")"
            gate_string += " " + ", ".join([f"q[{i}]" for i in range(qubits)]) + ";"
            qasm = self.qasm3_base.format(N=qubits, gate_strings=gate_string)

            # parse it through the hardware and verify result
            hw = get_default_matrix_hardware(qubits)
            parser = Qasm3Parser()
            builder = parser.parse(hw.create_builder(), qasm)
            gate_method = getattr(Gates, name)
            actual_gate = gate_method(*args)
            assert_same_up_to_phase(builder.matrix, actual_gate)


class TestQASM3Features:
    """Tests isolated features of QASM3 and OpenPulse, even those that aren't supported to
    ensure they throw errors."""

    @pytest.fixture
    def feature_testpath(self, testpath):
        """Fixture to provide the path to the test files."""
        return testpath / "files" / "qasm" / "qasm3" / "feature_tests"

    @pytest.fixture
    def model(self):
        return get_default_echo_hardware(32)

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
                kwargs[param] = next(qubits).index
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
                targets = inst.quantum_targets
                for target in targets:
                    devices.update(model.get_devices_from_pulse_channel(target.full_id()))
        return devices

    def return_builder_and_devices(self, model, feature_testpath, filename, **kwargs):
        """Returns the builder and source code for a given QASM file."""
        qasm = self.load_source(model, feature_testpath, filename, **kwargs)
        parser = Qasm3Parser()
        builder = parser.parse(get_builder(model), qasm)
        devices = self.get_devices_from_builder(builder, model)
        return builder, devices

    @pytest.mark.parametrize(
        "file, params",
        [
            (Path("classical", "break.qasm"), dict()),
            (Path("classical", "continue.qasm"), dict()),
            (Path("classical", "for.qasm"), dict()),
            (Path("classical", "functions.qasm"), dict()),
            (Path("classical", "if_else.qasm"), dict()),
            (Path("classical", "while.qasm"), dict()),
            (Path("gates", "modifiers", "ctrl.qasm"), dict()),
            (Path("gates", "modifiers", "inv.qasm"), dict()),
            (Path("gates", "modifiers", "negctrl.qasm"), dict()),
            (Path("gates", "modifiers", "pow.qasm"), dict()),
            (Path("keywords", "constant.qasm"), dict()),
            (Path("keywords", "def.qasm"), dict()),
            (Path("pulse", "functions", "durationof.qasm"), dict()),
            (Path("pulse", "instructions", "capture.qasm"), dict(capture_version="0")),
            (Path("pulse", "instructions", "capture.qasm"), dict(capture_version="4")),
            (Path("types", "angle.qasm"), dict()),
            (Path("types", "array.qasm"), dict()),
            (Path("types", "complex.qasm"), dict()),
            (Path("types", "duration.qasm"), dict()),
            (Path("types", "float.qasm"), dict()),
            (Path("types", "int.qasm"), dict()),
            (Path("types", "stretch.qasm"), dict()),
        ],
        ids=lambda file: str(file),
    )
    def test_unsupported_features_raise_errors(self, model, feature_testpath, file, params):
        qasm = self.load_source(model, feature_testpath, file, **params)
        parser = Qasm3Parser()
        # Use Exception to cast a wide net, as some non-supported features might be
        # return caught errors, others might be implicit.
        with pytest.raises(Exception):
            parser.parse(get_builder(model), qasm)

    def test_barrier(self, model, feature_testpath):
        qubits = model.qubits[:2]
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "barrier.qasm"),
            physical_index_1=qubits[0].index,
            physical_index_2=qubits[1].index,
        )

        syncs = [inst for inst in builder.instructions if isinstance(inst, Synchronize)]
        assert len(syncs) == 1

        expected_channels = set(
            [
                chan.full_id()
                for device in devices
                for chan in device.pulse_channels.values()
            ]
        )
        assert (
            set([chan.full_id() for chan in syncs[0].quantum_targets]) == expected_channels
        )

    def test_cx(self, model, feature_testpath):
        qubits = model.qubits[:2]
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "cx.qasm"),
            physical_index_1=qubits[0].index,
            physical_index_2=qubits[1].index,
        )

        # picks up extra qubits via syncs on connected qubits
        for qubit in qubits:
            assert qubit in devices

    def test_ecr(self, model, feature_testpath):
        qubits = model.qubits[:2]
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "ecr.qasm"),
            physical_index_1=qubits[0].index,
            physical_index_2=qubits[1].index,
        )

        # picks up extra qubits via syncs on connected qubits
        for qubit in qubits:
            assert qubit in devices

    def test_gate_def(self, model, feature_testpath):
        """Overloads X gate onto identity, so we check that there are no pulses."""
        qasm = self.load_source(model, feature_testpath, Path("gates", "gate_def.qasm"))
        parser = Qasm3Parser()
        builder = parser.parse(get_builder(model), qasm)
        for inst in builder.instructions:
            assert not isinstance(inst, (Pulse, CustomPulse))

    def test_measure(self, model, feature_testpath):
        qubit = model.qubits[0]
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "measure.qasm"),
            physical_index_1=qubit.index,
        )
        assert qubit in devices
        assert qubit.measure_device in devices

    def test_reset(self, model, feature_testpath):
        qubit = model.qubits[0]
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "reset.qasm"),
            physical_index_1=qubit.index,
        )
        assert qubit in devices
        assert len([inst for inst in builder.instructions if isinstance(inst, Reset)]) == 3

    def test_U(self, model, feature_testpath):
        qubit = model.qubits[0]
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("gates", "u.qasm"),
            physical_index_1=qubit.index,
        )
        assert qubit in devices

    def test_defcal(self, model, feature_testpath):
        """Used here to override the X gate for a qubit with a zero-delay.

        This also tests the keywords `defcalgrammar` and `extern`.
        """
        qubits = model.qubits[0:2]
        frame = f"q{qubits[0].index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("keywords", "defcal.qasm"),
            physical_index_1=qubits[0].index,
            physical_index_2=qubits[1].index,
            frame=frame,
        )

        assert qubits[0] in devices
        assert qubits[1] in devices
        delays = [inst for inst in builder.instructions if isinstance(inst, Delay)]
        assert len(delays) == 1
        assert np.isclose(delays[0].duration, 0.0)
        assert (
            delays[0].quantum_targets[0].full_id()
            == qubits[0].get_drive_channel().full_id()
        )

    def test_get_frequency(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        channel = qubit.get_drive_channel()
        assert frequencies[0].quantum_targets[0].full_id() == channel.full_id()
        assert np.isclose(frequencies[0].frequency, 0.05 * channel.frequency)

    def test_shift_frequency(self, model, feature_testpath):
        qubit = model.qubits[0]
        channel = qubit.get_drive_channel()
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "shift_frequency.qasm"),
            frame=frame,
            physical_index=qubit.index,
            frequency=channel.frequency * 0.05,
        )

        assert qubit in devices
        frequencies = [
            inst for inst in builder.instructions if isinstance(inst, FrequencyShift)
        ]
        assert len(frequencies) == 1
        assert frequencies[0].quantum_targets[0].full_id() == channel.full_id()
        assert np.isclose(frequencies[0].frequency, 0.05 * channel.frequency)

    def test_set_frequency(self, model, feature_testpath):
        """`set_frequency` actually adds a shift, so test for that."""
        qubit = model.qubits[0]
        channel = qubit.get_drive_channel()
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "set_frequency.qasm"),
            frame=frame,
            physical_index=qubit.index,
            frequency=channel.frequency * 1.05,
        )

        assert qubit in devices
        frequencies = [
            inst for inst in builder.instructions if isinstance(inst, FrequencyShift)
        ]
        assert len(frequencies) == 1
        assert frequencies[0].quantum_targets[0].full_id() == channel.full_id()
        assert np.isclose(frequencies[0].frequency, 0.05 * channel.frequency)

    def test_getphase(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
            assert phase.quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()

    def test_shift_phase(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "shift_phase.qasm"),
            frame=frame,
            physical_index=qubit.index,
            phase=2.0,
        )

        assert qubit in devices
        phases = [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        assert len(phases) == 1
        assert phases[0].quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()
        assert np.isclose(phases[0].phase, 0.254)

    def test_set_phase(self, model, feature_testpath):
        """`set_phase` actually adds a shift, so test for that."""
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "set_phase.qasm"),
            frame=frame,
            physical_index=qubit.index,
            phase=0.254,
        )

        assert qubit in devices
        phases = [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        assert len(phases) == 2
        assert phases[0].quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()
        assert phases[1].quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()
        assert np.isclose(phases[0].phase, 0.5)
        assert np.isclose(phases[1].phase, 0.254 - 0.5)

    def test_newframe(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        channel = qubit.get_drive_channel()
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "functions", "newframe.qasm"),
            frame=frame,
            physical_index=qubit.index,
            frequency=channel.frequency * 1.05,
        )

        assert qubit in devices
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 2
        assert pulses[0].quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()

        new_channel = pulses[1].quantum_targets[0]
        assert new_channel.full_id() != qubit.get_drive_channel().full_id()
        assert new_channel.physical_channel == qubit.physical_channel
        assert np.isclose(new_channel.frequency, channel.frequency * 1.05)

    @pytest.mark.parametrize("version", ["1", "2", "3"])
    def test_capture(self, model, feature_testpath, version):
        qubit = model.qubits[0]
        frame = f"r{qubit.index}_measure"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "capture.qasm"),
            frame=frame,
            physical_index=qubit.index,
            capture_version=version,
        )

        assert qubit.measure_device in devices
        acquires = [inst for inst in builder.instructions if isinstance(inst, Acquire)]
        assert len(acquires) == 1
        assert (
            acquires[0].quantum_targets[0].full_id()
            == qubit.get_measure_channel().full_id()
        )

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
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "delay.qasm"),
            frame=frame,
            physical_index=qubit.index,
            time="80ns",
        )

        assert qubit in devices
        delays = [inst for inst in builder.instructions if isinstance(inst, Delay)]
        assert len(delays) == 1
        assert delays[0].quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()
        assert np.isclose(delays[0].duration, 80e-9)

    def test_play(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "instructions", "play.qasm"),
            frame=frame,
            physical_index=qubit.index,
        )

        assert qubit in devices
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert pulses[0].quantum_targets[0].full_id() == qubit.get_drive_channel().full_id()

    def test_arb_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.allclose(pulses[0].samples, samples)

    def test_constant_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "constant.qasm"),
            frame=frame,
            amp=1e-4,
            width="80ns",
        )

        assert qubit in devices
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.SQUARE

    def test_drag_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.GAUSSIAN_DRAG
        assert np.isclose(pulses[0].std_dev, 20e-9)
        assert np.isclose(pulses[0].beta, 0.05)

    def test_gaussian_square_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.GAUSSIAN_SQUARE
        assert np.isclose(pulses[0].std_dev, 20e-9)
        assert np.isclose(pulses[0].square_width, 40e-9)

    def test_gaussian_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.GAUSSIAN_ZERO_EDGE
        assert np.isclose(pulses[0].std_dev, 20e-9)

    def test_sech_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.SECH
        assert np.isclose(pulses[0].std_dev, 20e-9)

    def test_sine_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.SIN
        assert np.isclose(pulses[0].frequency, 1e9)
        assert np.isclose(pulses[0].phase, 0.254)

    @pytest.mark.skip(reason="Fails because of hard-coded sample time in qasm3 parser.")
    def test_mix_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "mix.qasm"),
            frame=frame,
            width="80ns",
            amp1="2.5e-4",
            amp2="0.5",
        )

        assert qubit in devices
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.allclose(pulses[0].samples, [2.5e-4 * 0.5])
        assert np.isclose(pulses[0].duration, 80e-9)

    def test_phase_shift_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.SQUARE
        assert np.isclose(pulses[0].phase, 0.254)

    def test_scale_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "scale.qasm"),
            frame=frame,
            width="80ns",
            amp1="2.5e-4",
            amp2="0.5",
        )

        assert qubit in devices
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.isclose(pulses[0].duration, 80e-9)
        assert pulses[0].shape == PulseShapeType.SQUARE
        assert np.isclose(pulses[0].amp, 2.5e-4)
        assert np.isclose(pulses[0].scale_factor, 0.5)

    @pytest.mark.skip(reason="Fails because of hard-coded sample time in qasm3 parser.")
    def test_sum_waveform(self, model, feature_testpath):
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
        builder, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("pulse", "waveforms", "sum.qasm"),
            frame=frame,
            width="80ns",
            amp1="2.5e-4",
            amp2="5e-4",
        )

        assert qubit in devices
        pulses = [
            inst for inst in builder.instructions if isinstance(inst, (Pulse, CustomPulse))
        ]
        assert len(pulses) == 1
        assert np.allclose(pulses[0].samples, [2.5e-4 + 5e-4])
        assert np.isclose(pulses[0].duration, 80e-9)

    def test_type_physical_index(self, model, feature_testpath):
        qubit = model.qubits[0]
        _, devices = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("types", "physical_index.qasm"),
            physical_index=qubit.index,
        )
        assert qubit in devices
        assert qubit.measure_device in devices

    def test_type_port(self, model, feature_testpath):
        """Just a basic smoke test as there's nothing to really inspect."""
        qubit = model.qubits[0]
        _ = self.return_builder_and_devices(
            model,
            feature_testpath,
            Path("types", "port.qasm"),
            port=qubit.physical_channel.full_id().replace("CH", "channel_"),
        )

    @pytest.mark.parametrize(
        "width", ["80ns", "0.08us", "0.08µs", "80e-9s", "160dt", "80e-6ms"]
    )
    def test_units(self, model, feature_testpath, width):
        """Test that different units are parsed correctly."""
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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
        qubit = model.qubits[0]
        frame = f"q{qubit.index}_drive"
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

    def test_include(self, model, feature_testpath):
        lib = str(feature_testpath / "oqc_lib.inc")
        self.return_builder_and_devices(
            model, feature_testpath, Path("include.qasm"), lib=lib
        )


class TestParsing:
    echo = get_default_echo_hardware(6)

    @pytest.mark.parametrize("file_name", get_all_qasm2_paths(), ids=lambda val: val.name)
    def test_compare_tket_parser(self, file_name):
        if file_name.name == "example_if.qasm":
            pytest.skip(reason="If's not supported")
        qasm = get_qasm2(file_name)
        circ = None
        try:
            circ = circuit_from_qasm_str(qasm)
        except Exception:
            pass

        tket_builder: TketBuilder = TketQasmParser().parse(TketBuilder(), qasm)
        if circ is not None:
            assert circ.n_gates <= tket_builder.circuit.n_gates

    def test_invalid_gates(self):
        with pytest.raises(ValueError):
            RestrictedQasm2Parser({"cx"}).parse(
                get_builder(self.echo), get_qasm2("example.qasm")
            )

    def test_example(self):
        builder = parse_and_apply_optimizations("example.qasm")
        assert 347 == len(builder.instructions)

    def test_parallel(self):
        builder = parse_and_apply_optimizations("parallel_test.qasm", qubit_count=8)
        assert 2116 == len(builder.instructions)

    def test_mid_circuit_measurements(self):
        with pytest.raises(ValueError, match="No mid-circuit measurements allowed."):
            parse_and_apply_optimizations("mid_circuit_measure.qasm")

    def test_example_if(self):
        with pytest.raises(ValueError, match="IfElseOp is not currently supported."):
            parse_and_apply_optimizations("example_if.qasm")

    def test_move_measurements(self):
        # We need quite a few more qubits for this test.
        builder = parse_and_apply_optimizations("move_measurements.qasm", qubit_count=12)
        assert 97467 == len(builder.instructions)

    def test_random_n5_d5(self):
        builder = parse_and_apply_optimizations("random_n5_d5.qasm")
        assert 4956 == len(builder.instructions)

    def test_ordered_keys(self):
        builder = parse_and_apply_optimizations(
            "ordered_cregs.qasm", parser=CloudQasmParser()
        )
        ret_node: Return = builder.instructions[-1]
        assert isinstance(ret_node, Return)
        assert list(ret_node.variables) == ["a", "b", "c"]

    def test_restrict_if(self):
        with pytest.raises(ValueError, match="If's are currently unable to be used."):
            RestrictedQasm2Parser(disable_if=True).parse(
                get_builder(self.echo), get_qasm2("example_if.qasm")
            )

    def test_invalid_arbitrary_gate(self):
        with pytest.raises(KeyError):
            Qasm2Parser().parse(
                get_builder(self.echo), get_qasm2("invalid_custom_gate.qasm")
            )

    def test_valid_arbitrary_gate(self):
        parse_and_apply_optimizations("valid_custom_gate.qasm")

    def test_ecr_intrinsic(self):
        builder = parse_and_apply_optimizations("ecr.qasm")
        assert any(isinstance(inst, CrossResonancePulse) for inst in builder.instructions)
        assert 181 == len(builder.instructions)

    def test_rewiring_qubits_ecr(self):
        hardware = get_default_echo_hardware(
            4, connectivity=[(0, 2), (0, 3), (1, 2), (1, 3)]
        )
        qasm = get_qasm2("ecr.qasm")
        parser = Qasm2Parser()

        # Parsing fails without QASM optimisation for hardware topology.
        with pytest.raises(KeyError):
            parser.parse(get_builder(hardware), qasm)

        qasm = DefaultOptimizers().optimize_qasm(qasm, hardware, Qasm2Optimizations())
        builder = parser.parse(get_builder(hardware), qasm)
        assert any(isinstance(inst, CrossResonancePulse) for inst in builder.instructions)

    def test_ecr_already_exists(self):
        Qasm2Parser().parse(get_builder(self.echo), get_qasm2("ecr_exists.qasm"))

    @pytest.mark.parametrize(
        "qasm_file",
        [
            "basic.qasm",
            "basic_results_formats.qasm",
            "basic_single_measures.qasm",
            "ecr_exists.qasm",
            "ecr.qasm",
            "invalid_mid_circuit_measure.qasm",
            "logic_example.qasm",
            "more_basic.qasm",
            "parallel_test.qasm",
            "primitives.qasm",
            "valid_custom_gate.qasm",
        ],
    )
    def test_on_jagged_hardware(self, qasm_file):
        hw = get_jagged_echo_hardware(8)
        qasm_string = get_qasm2(qasm_file)
        parser = Qasm2Parser()
        result = parser.parse(get_builder(hw), qasm_string)
        assert len(result.instructions) > 0

    # TODO: Remove gates from list as support is added.
    _unsupported_gates = ("id", "u0", "rc3x", "c3x", "c3sqrtx", "c4x", "delay")

    @pytest.mark.parametrize(
        "gate_tup", get_default_qasm2_gate_qasms(), ids=lambda val: val[-1]
    )
    def test_default_gates(self, gate_tup):
        """Check that each default gate can be parsed individually."""
        N, gate_string = gate_tup
        if gate_string.startswith(self._unsupported_gates):
            pytest.skip("Gate not yet supported.")
        qasm = qasm2_base.format(N=N, gate_strings=gate_string)
        hw = get_default_echo_hardware(max(N, 2))
        parser = Qasm2Parser()
        builder = parser.parse(hw.create_builder(), qasm)
        assert isinstance(builder, InstructionBuilder)
        assert len(builder.instructions) > 0
        assert isinstance(builder.instructions[-1], Return)

    def test_default_gates_together(self):
        """Check that all default gates can be parsed together."""
        Ns, strings = zip(*get_default_qasm2_gate_qasms())
        N = max(Ns)
        # TODO: Remove filtering when all gates are supported.
        strings = filter(lambda s: not s.startswith(self._unsupported_gates), strings)
        gate_strings = "\n".join(strings)
        qasm = qasm2_base.format(N=N, gate_strings=gate_strings)
        hw = get_default_echo_hardware(
            N, [(i, j) for i in range(N) for j in range(i, N) if i != j]
        )
        parser = Qasm2Parser()
        builder = parser.parse(hw.create_builder(), qasm)
        assert isinstance(builder, InstructionBuilder)
        assert len(builder.instructions) > 0
        assert isinstance(builder.instructions[-1], Return)
