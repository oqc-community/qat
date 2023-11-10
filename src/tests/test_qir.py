# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import base64
from os.path import abspath, dirname, join
from unittest import mock

import pytest
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import qutip_available
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig
from qat.purr.integrations.qir import QIRParser
from qat.qat import execute, execute_qir

from tests.qasm_utils import TestFileType, get_test_file_path


def _get_qir_path(file_name):
    return join(abspath(join(dirname(__file__), "files", "qir", file_name)))


def _get_contents(file_path):
    """Get QASM from a file."""
    with open(_get_qir_path(file_path)) as ifile:
        return ifile.read()


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
        assert results == "00"

    def test_base_profile_ops(self):
        parser = QIRParser(get_default_echo_hardware(7))
        builder = parser.parse(_get_qir_path("base_profile_ops.ll"))
        assert len(builder.instructions) == 181

    def test_cudaq_input(self):
        results = execute(
            get_test_file_path(TestFileType.QIR, "basic_cudaq.ll"),
            get_default_echo_hardware(6),
        )

        assert results.get("r00000") == [0]

    @pytest.mark.skip("Needs full runtime.")
    def test_bell_measure_bitcode(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(
            _get_qir_path("bell_qir_measure.bc"), get_default_echo_hardware(4), config
        )

    @pytest.mark.skip("Needs full runtime.")
    def test_complicated(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(
            _get_qir_path("complicated.ll"), get_default_echo_hardware(4), config
        )

    @pytest.mark.skip("Needs full runtime.")
    def test_hello_bitcode(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(_get_qir_path("hello.bc"), get_default_echo_hardware(4), config)

    @pytest.mark.skip("Needs full runtime.")
    def test_select_bitcode(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        results = execute_qir(
            _get_qir_path("select.bc"), get_default_echo_hardware(4), config
        )
        assert results == "00"

    @pytest.mark.skip("Needs full runtime.")
    def test_teleport_chain_bitcode(self):
        config = CompilerConfig()
        config.results_format.squash_binary_result_arrays()
        execute_qir(
            _get_qir_path("teleportchain.ll"), get_default_echo_hardware(6), config
        )

    def test_qir_instruction_builder(self):
        parser = QIRParser(get_default_echo_hardware(4))
        builder = parser.parse(_get_qir_path("generator-bell.ll"))
        assert len(builder.instructions) == 97

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
        mock_builder = mock.create_autospec(InstructionBuilder)

        hardware = get_default_echo_hardware(qubit_count=2)
        parser = QIRParser(hardware, mock_builder)

        parser.parse(qir_file=_get_qir_path("bell_psi_plus.ll"))
        q0 = hardware.get_qubit(0)
        q1 = hardware.get_qubit(1)

        mock_builder.had.assert_called_once_with(q0)
        mock_builder.cnot.assert_called_once_with(q0, q1)

        expected_calls = [
            mock.call.measure_single_shot_z(q1, output_variable="1"),
            mock.call.measure_single_shot_z(q0, output_variable="0"),
        ]
        mock_builder.assert_has_calls(expected_calls, any_order=True)

    def test_parser_bell_psi_minus(self):
        mock_builder = mock.create_autospec(InstructionBuilder)
        hardware = get_default_echo_hardware(qubit_count=2)
        parser = QIRParser(hardware, mock_builder)

        parser.parse(qir_file=_get_qir_path("bell_psi_minus.ll"))

        q0 = hardware.get_qubit(0)
        q1 = hardware.get_qubit(1)

        mock_builder.X.assert_called_once_with(q0)
        mock_builder.had.assert_called_once_with(q0)
        mock_builder.cnot.assert_called_once_with(q0, q1)

        expected_calls = [
            mock.call.measure_single_shot_z(q1, output_variable="1"),
            mock.call.measure_single_shot_z(q0, output_variable="0"),
        ]
        mock_builder.assert_has_calls(expected_calls, any_order=True)

    def test_parser_bell_theta_plus(self):
        mock_builder = mock.create_autospec(InstructionBuilder)
        hardware = get_default_echo_hardware(qubit_count=2)
        parser = QIRParser(hardware, mock_builder)

        parser.parse(qir_file=_get_qir_path("bell_theta_plus.ll"))

        q0 = hardware.get_qubit(0)
        q1 = hardware.get_qubit(1)

        mock_builder.X.assert_called_once_with(q1)
        mock_builder.had.assert_called_once_with(q0)
        mock_builder.cnot.assert_called_once_with(q0, q1)

        expected_calls = [
            mock.call.measure_single_shot_z(q1, output_variable="1"),
            mock.call.measure_single_shot_z(q0, output_variable="0"),
        ]
        mock_builder.assert_has_calls(expected_calls, any_order=True)

    def test_parser_bell_theta_minus(self):
        mock_builder = mock.create_autospec(InstructionBuilder)
        hardware = get_default_echo_hardware(qubit_count=2)
        parser = QIRParser(hardware, mock_builder)

        parser.parse(qir_file=_get_qir_path("bell_theta_minus.ll"))
        q0 = hardware.get_qubit(0)
        q1 = hardware.get_qubit(1)

        mock_builder.had.assert_called_once_with(q0)
        mock_builder.cnot.assert_called_once_with(q0, q1)

        expected_calls = [
            mock.call.X(q1),
            mock.call.X(q0),
            mock.call.measure_single_shot_z(q1, output_variable="1"),
            mock.call.measure_single_shot_z(q0, output_variable="0"),
        ]

        mock_builder.assert_has_calls(expected_calls, any_order=True)

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_qir_bell_binary_count(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute_qir(
            _get_qir_path("generator-bell.ll"), compiler_config=config
        )
        assert len(results) == 4
        assert results["00"] > 1
        assert results["01"] > 1
        assert results["10"] > 1
        assert results["11"] > 1

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_qir_out_of_order_measure_declaration(self):
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute_qir(
            _get_qir_path("out_of_order_measure.ll"), compiler_config=config
        )
        assert len(results) == 4
