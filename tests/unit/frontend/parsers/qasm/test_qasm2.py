# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import re

import pytest
from compiler_config.config import Qasm2Optimizations

from qat.frontend.parsers.qasm import Qasm2Parser, RestrictedQasm2Parser
from qat.integrations.tket import run_pyd_tket_optimizations
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import InstructionBlock, Return
from qat.ir.measure import Acquire
from qat.model.hardware_model import PhysicalHardwareModel
from qat.utils.hardware_model import generate_hw_model

from tests.unit.utils.instruction import count_number_of_pulses
from tests.unit.utils.qasm_qir import get_default_qasm2_gate_qasms, get_qasm2, qasm2_base


def parse_qasm2_and_apply_optimisations(qasm, hw, filename=True):
    if filename:
        qasm = get_qasm2(qasm)
    qasm = run_pyd_tket_optimizations(qasm, opts=Qasm2Optimizations(), hardware=hw)

    parser = Qasm2Parser()
    return parser.parse(QuantumInstructionBuilder(hardware_model=hw), qasm)


@pytest.mark.parametrize("n_qubits", [8, 32, 64])
@pytest.mark.parametrize("seed", [8, 16])
class TestQasm2Parser:
    def test_invalid_gates(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)

        with pytest.raises(ValueError):
            RestrictedQasm2Parser({"cx"}).parse(
                QuantumInstructionBuilder(hardware_model=hw), get_qasm2("example.qasm")
            )

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
    def test_basic(self, qasm_file, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations(qasm_file, hw)
        assert builder.number_of_instructions > 0
        assert count_number_of_pulses(builder, "measure") >= 1

    def test_example(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations("example.qasm", hw)
        assert 363 == builder.number_of_instructions

    def test_parallel(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations("parallel_test.qasm", hw)
        assert count_number_of_pulses(builder, "drive") == 100  # we have 100 'sx gates
        assert count_number_of_pulses(builder, "measure") == 2

    def test_mid_circuit_measurements(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        with pytest.raises(ValueError, match="No mid-circuit measurements allowed."):
            parse_qasm2_and_apply_optimisations("mid_circuit_measure.qasm", hw)

    def test_example_if(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        with pytest.raises(ValueError, match="IfElseOp is not currently supported."):
            parse_qasm2_and_apply_optimisations("example_if.qasm", hw)

    def test_random_n5_d5(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations("random_n5_d5.qasm", hw)
        assert count_number_of_pulses(builder, "Drive") >= 60
        assert count_number_of_pulses(builder, "CrossResonance") >= 60
        assert count_number_of_pulses(builder, "CrossResonanceCancellation") >= 60
        assert count_number_of_pulses(builder, "Measure") == 5

    def test_restrict_if(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        with pytest.raises(ValueError, match="If's are currently unable to be used."):
            RestrictedQasm2Parser(disable_if=True).parse(
                QuantumInstructionBuilder(hardware_model=hw), get_qasm2("example_if.qasm")
            )

    def test_invalid_arbitrary_gate(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        with pytest.raises(ValueError):
            Qasm2Parser().parse(
                QuantumInstructionBuilder(hardware_model=hw),
                get_qasm2("invalid_custom_gate.qasm"),
            )

    def test_valid_arbitrary_gate(self, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations("valid_custom_gate.qasm", hw)
        assert count_number_of_pulses(builder, "drive") == 3
        assert count_number_of_pulses(builder, "measure") == 3

    @pytest.mark.parametrize("qasm_file", ["ecr.qasm", "ecr_exists.qasm"])
    def test_ecr(self, qasm_file, n_qubits, seed):
        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations(qasm_file, hw)
        assert count_number_of_pulses(builder, "CrossResonance") == 2
        assert count_number_of_pulses(builder, "CrossResonanceCancellation") == 2

    # TODO: Remove gates from list as support is added.
    _unsupported_gates = ("id", "u0", "rc3x", "c3x", "c3sqrtx", "c4x", "delay")

    @pytest.mark.parametrize(
        "gate_tup", get_default_qasm2_gate_qasms(), ids=lambda val: val[-1]
    )
    def test_default_gates(self, gate_tup, n_qubits, seed):
        """Check that each default gate can be parsed individually."""
        N, gate_string = gate_tup
        if gate_string.startswith(self._unsupported_gates):
            pytest.skip("Gate not yet supported.")
        qasm = qasm2_base.format(N=N, gate_strings=gate_string)

        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations(qasm, hw, filename=False)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert builder.number_of_instructions > 0
        assert isinstance(builder._ir.tail, Return)

    def test_default_gates_together(self, n_qubits, seed):
        """Check that all default gates can be parsed together."""
        Ns, strings = zip(*get_default_qasm2_gate_qasms())
        N = max(Ns)
        # TODO: Remove filtering when all gates are supported.
        strings = filter(lambda s: not s.startswith(self._unsupported_gates), strings)
        gate_strings = "\n".join(strings)
        qasm = qasm2_base.format(N=N, gate_strings=gate_strings)

        hw = generate_hw_model(n_qubits, seed=seed)
        builder = parse_qasm2_and_apply_optimisations(qasm, hw, filename=False)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert builder.number_of_instructions > 0
        assert isinstance(builder._ir.tail, Return)


def test_move_measurements():
    # Expensive test is put separately.
    hw = generate_hw_model(16)
    parse_qasm2_and_apply_optimisations("move_measurements.qasm", hw)


mapping_setup1 = (
    """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg b[2];
        measure q[0] -> b[1];
        measure q[1] -> b[0];
        """,
    {"0": 1, "1": 0},
)
mapping_setup2 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q[0] -> b[2];
    measure q[1] -> b[1];
    measure q[2] -> b[0];
    """,
    {"0": 2, "1": 1, "2": 0},
)
mapping_setup3 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q -> b;
    """,
    {"0": 0, "1": 1, "2": 2},
)


@pytest.mark.parametrize(
    "qasm, expected_mapping", [mapping_setup1, mapping_setup2, mapping_setup3]
)
def test_cl2qu_index_mapping(qasm, expected_mapping):
    hw = generate_hw_model(3)
    builder = parse_qasm2_and_apply_optimisations(qasm, hw, filename=False)
    mapping = get_cl2qu_index_mapping(builder._ir, hw)
    assert mapping == expected_mapping


def get_cl2qu_index_mapping(instructions: InstructionBlock, hw: PhysicalHardwareModel):
    """
    Returns a Dict[str, str] mapping creg to qreg indices.
    Classical register indices are extracted following the pattern <clreg_name>[<clreg_index>]
    """
    mapping = {}
    pattern = re.compile(r"(.*)\[([0-9]+)\]")

    for instruction in instructions:
        if not isinstance(instruction, Acquire):
            continue

        qubit_id, qubit = next(
            (
                (qubit_id, qubit)
                for qubit_id, qubit in hw.qubits.items()
                if qubit.acquire_pulse_channel.uuid == instruction.target
            ),
            None,
        )
        if qubit is None:
            raise ValueError(
                f"Could not find any qubits by acquire channel {instruction.target}."
            )

        result = pattern.match(instruction.output_variable)
        if result is None:
            raise ValueError(
                f"Could not extract cl register index from {instruction.output_variable}."
            )

        clbit_index = result.group(2)
        mapping[clbit_index] = qubit_id

    return mapping
