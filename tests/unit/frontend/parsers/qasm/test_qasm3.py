# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


import numpy as np
import pytest

from qat.frontend.parsers.qasm import Qasm3Parser
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import Return
from qat.ir.waveforms import ExtraSoftSquareWaveform, Pulse, SoftSquareWaveform
from qat.model.builder import PhysicalHardwareModelBuilder
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

        # test the extra_soft_square pulses are as expected
        ess_pulses = [
            inst
            for inst in qasm_inst
            if isinstance(inst, Pulse)
            and isinstance(inst.waveform, ExtraSoftSquareWaveform)
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
