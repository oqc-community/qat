# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd
import re
from math import isclose, pi

import pytest
from compiler_config.config import (
    InlineResultsProcessing,
    Qasm2Optimizations,
    Qasm3Optimizations,
    Tket,
    TketOptimizations,
)
from pytket import Circuit, OpType, Qubit
from sympy import pi as sympy_pi

from qat.exceptions.compilation import UnsupportedProgramError
from qat.integrations.tket import (
    TketBuilder,
    TketOptimisationHelper,
    TketToQatIRConverter,
    run_pyd_tket_optimizations,
)
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import PhaseSet, PhaseShift, ResultsProcessing
from qat.ir.measure import Acquire, Discriminate, Equalise, PostProcessing, PostSelect
from qat.model.loaders.converted import JaggedEchoModelLoader, PydEchoModelLoader
from qat.model.loaders.lucy import LucyModelLoader
from qat.utils.hardware_model import generate_hw_model, random_error_mitigation

from tests.unit.utils.qasm_qir import get_qasm2, get_qasm3

qasm2_files = [
    "basic.qasm",
    "basic_results_formats.qasm",
    "basic_single_measures.qasm",
    "ecr_exists.qasm",
    "ecr.qasm",
    "invalid_mid_circuit_measure.qasm",
    "logic_example.qasm",
    "more_basic.qasm",
    "primitives.qasm",
    "valid_custom_gate.qasm",
]
qasm3_files = [
    "basic.qasm",
    "u_gate.qasm",
    "ecr_test.qasm",
    "ghz.qasm",
    "complex_gates_test.qasm",
    "arb_waveform.qasm",
]


@pytest.mark.parametrize("n_qubits", [4, 8, 32, 64])
@pytest.mark.parametrize("seed", [7, 8, 9])
class TestPydTketOptimisation:
    @pytest.mark.parametrize("qasm_file", qasm2_files)
    def test_pyd_tket_qasm2_optimization(self, n_qubits, seed, qasm_file):
        qasm_string = get_qasm2(qasm_file)
        hw_model = generate_hw_model(n_qubits, seed=seed)

        run_pyd_tket_optimizations(
            qasm_string, opts=Qasm2Optimizations(), hardware=hw_model
        )

    @pytest.mark.parametrize("qasm_file", qasm3_files)
    def test_pyd_tket_qasm3_optimization(self, n_qubits, seed, qasm_file):
        qasm_string = get_qasm3(qasm_file)
        hw_model = generate_hw_model(n_qubits, seed=seed)

        run_pyd_tket_optimizations(
            qasm_string, opts=Qasm3Optimizations(), hardware=hw_model
        )


echo_error_mitigation = random_error_mitigation(set(range(4)), seed=42)
echo = PydEchoModelLoader(4, error_mitigation=echo_error_mitigation).load()
jagged_indices = {1, 2, 5, 7}
jagged_error_mitigation = random_error_mitigation(jagged_indices, seed=42)
jagged = JaggedEchoModelLoader(
    4,
    qubit_indices=jagged_indices,
    error_mitigation=jagged_error_mitigation,
    random_seed=254,
).load()


@pytest.mark.parametrize(
    "hw, best_qubit", [(echo, 2), (jagged, 2)], ids=["EchoLoader", "JaggedLoader"]
)
def test_one_qubit_circuit_optimisation(hw, best_qubit):
    qasm_1q_x_input = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q;
    measure q -> c;
    """
    opts = Tket().default()
    optimiser = TketOptimisationHelper(qasm_1q_x_input, opts, hw)
    optimiser.run_one_qubit_optimizations()
    circ = optimiser.circ
    assert best_qubit == circ.qubits[0].index[0]


@pytest.mark.parametrize("hw", [echo, jagged], ids=["EchoLoader", "JaggedLoader"])
def test_two_qubit_circuit_optimisation(hw):
    index_map = {phys: logic for logic, phys in enumerate(hw.qubits.keys())}

    qubit_pair_qualities = {}
    for q, coupled_qs in hw.logical_connectivity.items():
        for coupled_q in coupled_qs:
            quality = hw.logical_connectivity_quality[(q, coupled_q)]
            qubit_quality = hw.qubit_quality(q)
            coupled_qubit_quality = hw.qubit_quality(coupled_q)
            quality *= qubit_quality * coupled_qubit_quality
            qubit_pair_qualities[(index_map[q], index_map[coupled_q])] = quality

    best_qubit_pair = max(qubit_pair_qualities, key=qubit_pair_qualities.get)

    qasm_2q_bell_input = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];

    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    opts = Tket().default()
    optimiser = TketOptimisationHelper(qasm_2q_bell_input, opts, hw)
    optimiser.run_multi_qubit_optimizations(use_1q_quality=True)
    circ = optimiser.circ

    circ_indices = [idx for qubit in circ.qubits for idx in qubit.index]

    for qubit in best_qubit_pair:
        assert qubit in circ_indices


def _ghz_qasm2(n_used: int, n_declared: int | None = None) -> str:
    """A QASM2 GHZ program on the first ``n_used`` qubits, measuring each of them.

    Registers have ``n_declared`` qubits and bits, defaulting to ``n_used``.
    """
    n_declared = n_declared or n_used
    cx_chain = "".join(f"cx q[{i}], q[{i + 1}];\n" for i in range(n_used - 1))
    measures = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n_used))
    return (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        f"qreg q[{n_declared}];\ncreg c[{n_declared}];\n"
        f"h q[0];\n{cx_chain}{measures}"
    )


class TestCircuitSize:
    """Circuits too big for the device get a clear error."""

    model_32q = PydEchoModelLoader(32).load()
    opts = Qasm2Optimizations().tket_optimizations
    too_many_qubits = re.escape("Circuit uses 34 qubits, only 32 available.")

    def test_more_qubits_than_device_raises(self):
        with pytest.raises(UnsupportedProgramError, match=self.too_many_qubits):
            run_pyd_tket_optimizations(
                _ghz_qasm2(34), opts=self.opts, hardware=self.model_32q
            )

    def test_more_qubits_than_device_raises_without_mapping_pass(self):
        opts = self.opts & ~TketOptimizations.DefaultMappingPass
        with pytest.raises(UnsupportedProgramError, match=self.too_many_qubits):
            run_pyd_tket_optimizations(_ghz_qasm2(34), opts=opts, hardware=self.model_32q)

    def test_no_connected_part_fits_raises(self):
        model = PydEchoModelLoader(4, connectivity=[(0, 1), (2, 3)]).load()
        with pytest.raises(
            UnsupportedProgramError,
            match=re.escape(
                "Circuit uses 3 qubits, but the largest connected group of qubits on "
                "the device has 2."
            ),
        ):
            run_pyd_tket_optimizations(_ghz_qasm2(3), opts=self.opts, hardware=model)

    def test_circuit_matching_device_size_compiles(self):
        qasm = run_pyd_tket_optimizations(
            _ghz_qasm2(32), opts=self.opts, hardware=self.model_32q
        )
        assert isinstance(qasm, str)

    def test_declared_but_unused_qubits_count_against_device(self):
        # QASM keeps declared qubits, so users can pick physical qubits by index.
        with pytest.raises(UnsupportedProgramError, match=self.too_many_qubits):
            run_pyd_tket_optimizations(
                _ghz_qasm2(32, n_declared=34), opts=self.opts, hardware=self.model_32q
            )

    def test_circuit_with_one_used_qubit_at_high_index_is_placed(self):
        # Circuit input has unused qubits stripped, leaving q[33] as its only qubit.
        circuit = Circuit(34, 1).X(33).Measure(33, 0)
        qasm = run_pyd_tket_optimizations(circuit, opts=self.opts, hardware=self.model_32q)
        assert "qreg q[" not in qasm
        assert "qreg node[" in qasm

    def test_classical_register_wider_than_32_bits(self):
        program = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[40];\nh q[0];\ncx q[0], q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[39];\n"
        )
        qasm = run_pyd_tket_optimizations(program, opts=self.opts, hardware=self.model_32q)
        assert "creg c[40];" in qasm

    def test_circuit_using_no_qubits_compiles_on_uncoupled_device(self):
        # Circuit input has unused qubits stripped, leaving none to route.
        model = PydEchoModelLoader(1, connectivity=[]).load()
        qasm = run_pyd_tket_optimizations(Circuit(1, 1), opts=self.opts, hardware=model)
        assert "qreg" not in qasm


class TestTketBuilder:
    model = LucyModelLoader(3).load()

    def test_get_qubit(self):
        builder = TketBuilder(self.model)
        qubit = builder.get_logical_qubit(1)
        assert isinstance(qubit, Qubit)
        assert qubit.index[0] == 1

    @pytest.mark.parametrize(
        "method, angle, expected",
        [
            ("X", None, OpType.X),
            ("Y", None, OpType.Y),
            ("Z", None, OpType.Z),
            ("X", 0.254 * pi, OpType.Rx),
            ("Y", 0.5 * pi, OpType.Ry),
            ("Z", 2 * pi, OpType.Rz),
        ],
    )
    def test_X_Y_Z(self, method, angle, expected):
        builder = TketBuilder(self.model)
        qubit = builder.qubits[1]
        if angle is not None:
            getattr(builder, method)(qubit, angle)
        else:
            getattr(builder, method)(qubit)
        commands = builder.circuit.commands_of_type(expected)
        assert len(commands) == 1
        assert commands[0].qubits == [qubit]
        if angle is not None:
            assert isclose(commands[0].op.params[0], angle / pi)

    def test_U(self):
        angles = [0.254, 0.7, 1.2]
        builder = TketBuilder(self.model)
        qubit = builder.qubits[1]
        builder.U(qubit, angles[0] * pi, angles[1] * pi, angles[2] * pi)
        commands = builder.circuit.commands_of_type(OpType.U3)
        assert len(commands) == 1
        assert commands[0].qubits == [qubit]
        for i in range(3):
            assert isclose(commands[0].op.params[i], angles[i])

    @pytest.mark.parametrize(
        "method, expected",
        [
            ("swap", OpType.SWAP),
            ("cnot", OpType.CX),
            ("ECR", OpType.ECR),
        ],
    )
    def test_2q(self, method, expected):
        builder = TketBuilder(self.model)
        qubit1 = builder.qubits[1]
        qubit2 = builder.qubits[2]
        getattr(builder, method)(qubit1, qubit2)
        commands = builder.circuit.commands_of_type(expected)
        assert len(commands) == 1
        assert commands[0].qubits == [qubit1, qubit2]

    @pytest.mark.parametrize(
        "method, angle, expected",
        [
            ("cX", 0.254, OpType.CRx),
            ("cY", 0.5, OpType.CRy),
            ("cZ", 0.7, OpType.CRz),
        ],
    )
    def test_parametrized_2q(self, method, angle, expected):
        builder = TketBuilder(self.model)
        qubit1 = builder.qubits[1]
        qubit2 = builder.qubits[2]
        getattr(builder, method)(qubit1, qubit2, angle * pi)
        commands = builder.circuit.commands_of_type(expected)
        assert len(commands) == 1
        assert commands[0].qubits == [qubit1, qubit2]
        assert isclose(commands[0].op.params[0], angle)

    def test_measure(self):
        builder = TketBuilder(self.model)
        qubit = builder.qubits[1]
        builder.measure_with_granular_post_processing(qubit)
        commands = builder.circuit.commands_of_type(OpType.Measure)
        assert len(commands) == 1
        assert commands[0].qubits == [qubit]
        assert commands[0].bits == [builder.circuit.bits[0]]

    def test_reset(self):
        builder = TketBuilder(self.model)
        qubit = builder.qubits[1]
        builder.reset(qubit)
        commands = builder.circuit.commands_of_type(OpType.Reset)
        assert len(commands) == 1
        assert commands[0].qubits == [qubit]


class TestTketToQatIRConverter:
    @pytest.mark.parametrize(
        "params",
        [
            ("0.5", 0.5),
            ("1/2", 0.5),
            ("0", 0.0),
            ("0.254", 0.254),
            ("1", 1.0),
            ("-4/3", -4 / 3),
        ],
    )
    def test_convert_parameter(self, params):
        isclose(TketToQatIRConverter.convert_parameter(params[0]), params[1] * pi)

    model = LucyModelLoader(10).load()
    qubit1 = model.qubit_with_index(0)
    qubit2 = model.qubit_with_index(1)

    def compare_builders(self, builder1, builder2):
        builder1.flatten()
        builder2.flatten()
        assert len(builder1.instructions) == len(builder2.instructions)
        for i, inst in enumerate(builder1.instructions):
            qat_inst = builder2.instructions[i]
            if isinstance(inst, Acquire):
                # output_variable might be different, but check other things
                assert type(inst) is type(qat_inst)
                assert inst.target == qat_inst.target
                assert inst.mode == qat_inst.mode
                assert inst.duration == qat_inst.duration
            elif isinstance(inst, PostProcessing | Equalise | Discriminate | PostSelect):
                # Post-processing instructions may have differing output_variable values;
                # just verify the instruction type and pipeline-structure matches.
                assert type(inst) is type(qat_inst)
                if isinstance(inst, PostProcessing):
                    assert inst.process_type == qat_inst.process_type
                    assert inst.axes == qat_inst.axes
            elif isinstance(inst, PhaseShift | PhaseSet):
                assert type(inst) is type(qat_inst)
                assert inst.target == qat_inst.target
                assert isclose(inst.phase, qat_inst.phase)
            else:
                assert inst == qat_inst

    @pytest.mark.parametrize(
        "tket_method, tket_args, qat_method, qat_args",
        [
            ("X", (0,), "X", (qubit1,)),
            ("Y", (0,), "Y", (qubit1,)),
            ("Z", (0,), "Z", (qubit1,)),
            ("Rx", (0.254, 0), "X", (qubit1, 0.254 * pi)),
            ("Ry", (0.454, 0), "Y", (qubit1, 0.454 * pi)),
            ("Rz", (0.7, 0), "Z", (qubit1, 0.7 * pi)),
            ("H", (0,), "had", (qubit1,)),
            ("S", (0,), "S", (qubit1,)),
            ("Sdg", (0,), "Sdg", (qubit1,)),
            ("T", (0,), "T", (qubit1,)),
            ("Tdg", (0,), "Tdg", (qubit1,)),
            ("U1", (0.254, 0), "Z", (qubit1, 0.254 * pi)),
            ("U2", (0.254, 0.7, 0), "U", (qubit1, pi / 2, 0.254 * pi, 0.7 * pi)),
            ("U3", (0.254, 0.7, 1.2, 0), "U", (qubit1, 0.254 * pi, 0.7 * pi, 1.2 * pi)),
            ("CX", (0, 1), "cnot", (qubit1, qubit2)),
            ("ECR", (0, 1), "ECR", (qubit1, qubit2)),
            ("Measure", (0, 0), "measure_with_granular_post_processing", (qubit1,)),
            ("Reset", (0,), "reset", (qubit1,)),
            # The following tests expressions that are symbolic (regression tests)
            ("Rx", (sympy_pi, 0), "X", (qubit1, pi**2)),
            ("Rx", (sympy_pi / 2.54, 0), "X", (qubit1, pi**2 / 2.54)),
            ("Rx", (2.54 * sympy_pi, 0), "X", (qubit1, 2.54 * pi**2)),
            ("Rx", (2.54 / sympy_pi, 0), "X", (qubit1, 2.54)),
        ],
    )
    def test_supported_commands(self, tket_method, tket_args, qat_method, qat_args):
        builder = QuantumInstructionBuilder(self.model)
        converter = TketToQatIRConverter()

        # Build from tket
        circ = Circuit(2, 2)
        getattr(circ, tket_method)(*tket_args)
        tket_builder = TketBuilder(self.model)
        tket_builder.circuit = circ
        if qat_method == "measure_with_granular_post_processing":
            tket_builder._output_variables[0] = "0"
        builder = converter.convert(builder, tket_builder)

        # Build directly in QAT
        direct_builder = QuantumInstructionBuilder(self.model)
        getattr(direct_builder, qat_method)(*qat_args)

        self.compare_builders(builder, direct_builder)

    def test_bell_state(self):
        """Basic smoke test to check composite circuits work."""
        builder = QuantumInstructionBuilder(self.model)
        converter = TketToQatIRConverter()

        # Build from tket
        tket_builder = TketBuilder(self.model)
        qubit1 = tket_builder.qubits[0]
        qubit2 = tket_builder.qubits[1]
        tket_builder.had(qubit1)
        tket_builder.cnot(qubit1, qubit2)
        tket_builder.measure_with_granular_post_processing(qubit1, output_variable="0")
        tket_builder.measure_with_granular_post_processing(qubit2, output_variable="1")
        builder = converter.convert(builder, tket_builder)

        # Build directly in QAT
        direct_builder = QuantumInstructionBuilder(self.model)
        qubit1 = direct_builder.qubits[0]
        qubit2 = direct_builder.qubits[1]
        direct_builder.had(qubit1)
        direct_builder.cnot(qubit1, qubit2)
        direct_builder.measure_with_granular_post_processing(qubit1, output_variable="0")
        direct_builder.measure_with_granular_post_processing(qubit2, output_variable="1")

        self.compare_builders(builder, direct_builder)

    def test_extra_rp_instructions_are_added(self):
        builder = QuantumInstructionBuilder(self.model)
        converter = TketToQatIRConverter()

        # Build from tket
        tket_builder = TketBuilder(self.model)
        qubit1 = tket_builder.qubits[0]
        tket_builder.had(qubit1)
        tket_builder.measure_with_granular_post_processing(qubit1, output_variable="0")
        tket_builder.results_processing("0", InlineResultsProcessing.Program)
        builder = converter.convert(builder, tket_builder)
        assert isinstance(builder.instructions[-1], ResultsProcessing)
        assert (
            builder.instructions[-1].results_processing == InlineResultsProcessing.Program
        )
        assert builder.instructions[-1].variable == "0"
