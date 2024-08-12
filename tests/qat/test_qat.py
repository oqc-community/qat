# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.compiler.config import CompilerConfig, MetricsType, QuantumResultsFormat
from qat.qat import execute_with_metrics
from tests.qat.qasm_utils import TestFileType, get_test_file_path
from tests.qat.utils import ListReturningEngine


@pytest.mark.parametrize(
    ("input_string", "file_type", "instruction_length"),
    [("ghz.qasm", TestFileType.QASM2, 196)],
)
def test_all_metrics_are_returned(input_string, file_type, instruction_length):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is not None
    assert metrics["optimized_instruction_count"] == instruction_length


@pytest.mark.parametrize(("input_string", "file_type"), [("ghz.qasm", TestFileType.QASM2)])
def test_only_optim_circuitmetrics_are_returned(input_string, file_type):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    config.metrics = MetricsType.OptimizedCircuit
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is not None
    assert metrics["optimized_instruction_count"] is None


@pytest.mark.parametrize(("input_string", "file_type"), [("ghz.qasm", TestFileType.QASM2)])
def test_only_inst_len_circuitmetrics_are_returned(input_string, file_type):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    config.metrics = MetricsType.OptimizedInstructionCount
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is None
    assert metrics["optimized_instruction_count"] is not None


@pytest.mark.parametrize("engine", [EchoEngine, ListReturningEngine])
@pytest.mark.parametrize(
    ("input_string", "file_type"),
    [
        ("ghz.qasm", TestFileType.QASM2),
        ("basic.qasm", TestFileType.QASM3),
        ("generator-bell.ll", TestFileType.QIR),
    ],
)
def test_batched_execution(input_string, file_type, engine):
    hardware = get_default_echo_hardware()
    program = get_test_file_path(file_type, input_string)
    config = CompilerConfig(
        repeats=int(hardware.repeat_limit * 1.5),
        results_format=QuantumResultsFormat().raw(),
    )

    result, _ = execute_with_metrics(program, engine(hardware), config)
    if isinstance(result, dict):
        result = list(result.values())[0]
    assert isinstance(result, list)
    for res in result:
        # list of measurements for each qubit
        assert len(res) == config.repeats
