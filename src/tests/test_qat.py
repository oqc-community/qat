# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.config import CompilerConfig, MetricsType
from qat.qat import execute_with_metrics

from tests.qasm_utils import TestFileType, get_test_file_path

@pytest.mark.parametrize(("input_string", "file_type", "instruction_length"),
                         [("ghz.qasm", TestFileType.QASM2, 500)])
def test_all_metrics_are_returned_qasm(input_string, file_type, instruction_length):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    results, metrics = execute_with_metrics(program, hardware, config)
    assert len(metrics["optimized_circuit"]) > 0
    assert metrics["total_duration"] > 0
    assert metrics["execution_duration"] > 0
    assert metrics["optimization_duration"] > 0
    assert metrics["parse_duration"] > 0
    assert metrics["engine_call_duration"] > 0
    assert metrics["optimized_instruction_count"] > 0

@pytest.mark.parametrize(("input_string", "file_type", "instruction_length"),
                         [("generator-bell.ll", TestFileType.QIR, 97)])
def test_all_metrics_are_returned_qir(input_string, file_type, instruction_length):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is None
    assert metrics["total_duration"] > 0
    assert metrics["execution_duration"] > 0
    assert metrics["optimization_duration"] is None
    assert metrics["parse_duration"] is not None
    assert metrics["engine_call_duration"] > 0
    assert metrics["optimized_instruction_count"] is instruction_length


@pytest.mark.parametrize(("input_string", "file_type"),
                         [("ghz.qasm", TestFileType.QASM2)])
def test_only_optim_circuit_metrics_are_returned_qasm(input_string, file_type):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    config.metrics = MetricsType.OptimizedCircuit
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is not None and len(metrics["optimized_circuit"]) > 0
    assert metrics["optimized_instruction_count"] is None
    assert metrics["total_duration"] is None
    assert metrics["execution_duration"] is None
    assert metrics["optimization_duration"] is None
    assert metrics["parse_duration"] is None
    assert metrics["engine_call_duration"] is None


@pytest.mark.parametrize(("input_string", "file_type", "instruction_length"),
                         [("ghz.qasm", TestFileType.QASM2, 500),
                         ("generator-bell.ll", TestFileType.QIR, 97)])
def test_only_inst_len_circuit_metrics_are_returned(input_string, file_type, instruction_length):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    config.metrics = MetricsType.OptimizedInstructionCount
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is None
    assert metrics["optimized_instruction_count"] == instruction_length
    assert metrics["total_duration"] is None
    assert metrics["execution_duration"] is None
    assert metrics["optimization_duration"] is None
    assert metrics["parse_duration"] is None
    assert metrics["engine_call_duration"] is None
