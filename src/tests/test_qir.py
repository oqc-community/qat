# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import base64
from os.path import abspath, dirname, join

import numpy as np
import pytest

from qat.purr.backends.echo import get_default_echo_hardware, apply_setup_to_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import qutip_available
from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.config import CompilerConfig
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.interrupt import Interrupt
from qat.purr.compiler.runtime import QuantumRuntime
from qat.qat import execute, execute_qir
from .qasm_utils import TestFileType, get_test_file_path
from .utils import get_jagged_echo_hardware


def _get_qir_path(file_name):
    return join(abspath(join(dirname(__file__), "files", "qir", file_name)))


def _get_contents(file_path):
    """Get QASM from a file."""
    with open(_get_qir_path(file_path)) as ifile:
        return ifile.read()


class QIRTestingModel(QuantumHardwareModel):
    def __init__(self):
        super().__init__()
        self.builder_mock = QIRBuilderMock(self)

    def create_builder(self) -> "InstructionBuilder":
        return self.builder_mock

    class QIRExecutionEngine(QuantumExecutionEngine):
        def _execute_on_hardware(self, sweep_iterator, package: "QatFile", interrupt: Interrupt):
            return dict()

    def create_runtime(self, existing_engine=None):
        return QIRRuntimeMock(QIRTestingModel.QIRExecutionEngine(self))


class QIRRuntimeMock(QuantumRuntime):
    def execute(
        self, instructions, results_format=None, repeats=None, error_mitigation=None
    ):
        return dict()

    def execute_with_interrupt(
        self,
        instructions,
        interrupt,
        results_format=None,
        repeats=None,
        error_mitigation=None,
    ):
        return dict()


class QIRBuilderMock(QuantumInstructionBuilder):
    def __init__(self, hardware_model):
        super().__init__(hardware_model)
        self.recorded = []

    def X(self, target, radii=None):
        self.recorded.append(("x", target, radii))

    def Y(self, target, radii=None):\
        self.recorded.append(("y", target, radii))

    def Z(self, target, radii=None):
        self.recorded.append(("z", target, radii))

    def cnot(self, controlled_qubit, target_qubit):
        self.recorded.append(("cnot", controlled_qubit, target_qubit))

    def measure_single_shot_z(
        self, target, axis: str = None, output_variable: str = None
    ):
        self.recorded.append(("measure_single_shot_z", target, axis, output_variable))

    def reset(self, qubits):
        self.recorded.append(("reset", qubits))


def get_builder_for_introspection(file: str, qubits: int = 4) -> QIRBuilderMock:
    model = QIRTestingModel()
    hardware = apply_setup_to_hardware(model, qubits)
    execute_qir(_get_qir_path(file), hardware)
    return model.builder_mock


class TestQIR:
    def test_invalid_paths(self):
        with pytest.raises(ValueError):
            execute(abspath("\\very\\wrong.ll"))

        with pytest.raises(ValueError):
            execute(abspath("/very/wrong.ll"))

        with pytest.raises(ValueError):
            execute("/very/wrong.bc")

    def test_valid_ll_path(self):
        execute(
            get_test_file_path(TestFileType.QIR, "generator-bell.ll"),
            get_default_echo_hardware(2),
        )

    def test_qir_bell(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        results = execute_qir(
            _get_qir_path("generator-bell.ll"), get_default_echo_hardware(4), config
        )
        assert results == {"00": 1000}

    @pytest.mark.skip("Needs base profile label on results.")
    def test_cudaq_input(self):
        """ TODO: Needs base profile label on results. """
        results = execute(
            get_test_file_path(TestFileType.QIR, "basic_cudaq.ll"),
            get_default_echo_hardware(6),
        )

        assert results.get("r00000") == [0]

    def test_bell_measure_bitcode(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(
            _get_qir_path("bell_qir_measure.bc"), get_default_echo_hardware(4), config
        )

    def test_hello_bitcode(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(_get_qir_path("hello.bc"), get_default_echo_hardware(4), config)

    @pytest.mark.skip("Needs zext instruction.")
    def test_select_bitcode(self):
        """ TODO: Implement zext. """
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        results = execute_qir(
            _get_qir_path("select.bc"), get_default_echo_hardware(4), config
        )
        assert results == "00"

    @pytest.mark.skip("Needs sext instruction.")
    def test_teleport_chain_bitcode(self):
        """ TODO: Implement sext. """
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(_get_qir_path("teleportchain.ll"), get_default_echo_hardware(6), config)

    def test_common_entrypoint_file(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute(
            _get_qir_path("generator-bell.ll"), get_default_echo_hardware(4), config
        )
        assert results == {"00": 1000}

    def test_common_entrypoint_string(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute(
            _get_contents("generator-bell.ll"), get_default_echo_hardware(4), config
        )
        assert results == {"00": 1000}

    @pytest.mark.skip(reason="Memory violation in LLVM wrapper. Need to look at what is going wrong here.")
    def test_common_entrypoint_bitcode(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        program = _get_contents("base64_bitcode_ghz")
        program = base64.b64decode(program)
        results = execute(program, get_default_echo_hardware(4), config)
        assert results == {"0": 1000}

    def test_invalid_QIR(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        with pytest.raises(ValueError):
            execute("", get_default_echo_hardware(4), config)

    def test_parser_bell_psi_plus(self):
        mock_builder = get_builder_for_introspection("bell_psi_plus.ll", 2)

        q0 = mock_builder.model.get_qubit(0)
        q1 = mock_builder.model.get_qubit(1)

        generated_one = mock_builder.recorded[3][3]
        generated_two = mock_builder.recorded[4][3]

        # Assert the names aren't empty.
        assert generated_one
        assert generated_two
        assert mock_builder.recorded == [
            ("z", q0, np.pi),
            ("y", q0, np.pi/2),
            ("cnot", q0, q1),
            ("measure_single_shot_z", q0, None, generated_one),
            ("measure_single_shot_z", q1, None, generated_two)
        ]

    def test_parser_bell_psi_minus(self):
        mock_builder = get_builder_for_introspection("bell_psi_minus.ll", 2)

        q0 = mock_builder.model.get_qubit(0)
        q1 = mock_builder.model.get_qubit(1)

        generated_one = mock_builder.recorded[4][3]
        generated_two = mock_builder.recorded[5][3]

        # Assert the names aren't empty.
        assert generated_one
        assert generated_two
        assert mock_builder.recorded == [
            ("x", q0, np.pi),
            ("z", q0, np.pi),
            ("y", q0, np.pi / 2),
            ("cnot", q0, q1),
            ("measure_single_shot_z", q0, None, generated_one),
            ("measure_single_shot_z", q1, None, generated_two)
        ]

    def test_parser_bell_theta_plus(self):
        mock_builder = get_builder_for_introspection("bell_theta_plus.ll", 2)

        q0 = mock_builder.model.get_qubit(0)
        q1 = mock_builder.model.get_qubit(1)

        generated_one = mock_builder.recorded[4][3]
        generated_two = mock_builder.recorded[5][3]

        # Assert the names aren't empty.
        assert generated_one
        assert generated_two
        assert mock_builder.recorded == [
            ("x", q1, np.pi),
            ("z", q0, np.pi),
            ("y", q0, np.pi / 2),
            ("cnot", q0, q1),
            ("measure_single_shot_z", q0, None, generated_one),
            ("measure_single_shot_z", q1, None, generated_two)
        ]

    def test_parser_bell_theta_minus(self):
        mock_builder = get_builder_for_introspection("bell_theta_minus.ll", 2)

        q0 = mock_builder.model.get_qubit(0)
        q1 = mock_builder.model.get_qubit(1)

        generated_one = mock_builder.recorded[5][3]
        generated_two = mock_builder.recorded[6][3]

        # Assert the names aren't empty.
        assert generated_one
        assert generated_two
        assert mock_builder.recorded == [
            ("x", q1, np.pi),
            ("x", q0, np.pi),
            ("z", q0, np.pi),
            ("y", q0, np.pi / 2),
            ("cnot", q0, q1),
            ("measure_single_shot_z", q0, None, generated_one),
            ("measure_single_shot_z", q1, None, generated_two)
        ]

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_qir_bell_binary_count(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute_qir(_get_qir_path("generator-bell.ll"), get_default_qiskit_hardware(2), compiler_config=config)
        assert len(results) == 2
        assert results["00"] > 400
        assert results["11"] > 400

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_qir_out_of_order_measure_declaration(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute_qir(
            _get_qir_path("out_of_order_measure.ll"), get_default_qiskit_hardware(2), compiler_config=config
        )
        assert len(results) == 2

    @pytest.mark.skip(reason="With jagged hardware requires routing to execute correctly.")
    @pytest.mark.parametrize(
        "qir_file",
        [
            # "base_profile_ops.ll", MCM needed
            "basic_cudaq.ll",
            "bell_psi_minus.ll",
            "bell_psi_plus.ll",
            "bell_theta_minus.ll",
            "bell_theta_plus.ll",
            # "complicated.ll",  MCM needed
            "generator-bell.ll",
            "out_of_order_measure.ll",
            # "select.bc", Needs zext implemented
        ],
    )
    def test_on_jagged_hardware(self, qir_file):
        model = QIRTestingModel()
        hardware: QIRTestingModel = get_jagged_echo_hardware(8, applied_model=model)
        execute_qir(_get_qir_path(qir_file), hardware)
        assert len(hardware.builder_mock.recorded) > 0

    def test_execute_different_qat_input_types(self):
        hw = get_default_echo_hardware(5)
        qubit = hw.get_qubit(0)
        phase_shift_1 = 0.2
        phase_shift_2 = 0.1
        builder = (
            hw.create_builder()
            .phase_shift(qubit, phase_shift_1)
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, phase_shift_2)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit)
        )

        with pytest.raises(TypeError):
            execute_qir(qat_input=builder.instructions)
