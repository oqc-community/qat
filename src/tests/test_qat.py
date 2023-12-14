# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.backends import live
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.live import get_default_lucy_hardware
from qat.purr.compiler.config import CompilerConfig, MetricsType, Tket, TketOptimizations
from qat.purr.compiler.devices import Calibratable
from qat.purr.compiler.frontends import LanguageFrontend
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.qat import execute_with_metrics, validate_circuit_length, fetch_frontend, _return_or_build
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware

from tests.qasm_utils import TestFileType, get_test_file_path


@pytest.mark.parametrize(
    ("input_string", "file_type", "instruction_length"),
    [("ghz.qasm", TestFileType.QASM2, 184)],
)
def test_all_metrics_are_returned(input_string, file_type, instruction_length):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is not None
    assert metrics["optimized_instruction_count"] == instruction_length


@pytest.mark.parametrize(
    ("input_string", "file_type"), [("ghz.qasm", TestFileType.QASM2)]
)
def test_only_optim_circuitmetrics_are_returned(input_string, file_type):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    config.metrics = MetricsType.OptimizedCircuit
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is not None
    assert metrics["optimized_instruction_count"] is None


@pytest.mark.parametrize(
    ("input_string", "file_type"), [("ghz.qasm", TestFileType.QASM2)]
)
def test_only_inst_len_circuitmetrics_are_returned(input_string, file_type):
    program = get_test_file_path(file_type, input_string)
    hardware = get_default_echo_hardware()
    config = CompilerConfig()
    config.metrics = MetricsType.OptimizedInstructionCount
    results, metrics = execute_with_metrics(program, hardware, config)
    assert metrics["optimized_circuit"] is None
    assert metrics["optimized_instruction_count"] is not None

@pytest.mark.parametrize(("input_string", "file_type", "expected_result"),
                         [("primitives.qasm", TestFileType.QASM2, True),
                         ("ghz.qasm", TestFileType.QASM3, True),
                         ("ghz.qasm", TestFileType.QASM3, True),
                         ("bell_psi_plus.ll", TestFileType.QIR, True),
                         ("cross_ressonance.qasm", TestFileType.QASM3, True),
                         ("long_qasm.qasm", TestFileType.QASM2, False)])
def test_circuit_length_validation(input_string, file_type, expected_result):
    program = get_test_file_path(file_type, input_string)

    optim = Tket()
    optim.disable()
    config = CompilerConfig(optimizations=optim)

    # Test with program

    # default parameters
    assert validate_circuit_length(program = program, compiler_config = config) == expected_result

    # max circuit duration input
    assert validate_circuit_length(program=program, compiler_config=config, max_circuit_duration= 1) == False

    # Test with instruction builder
    live: QuantumHardwareModel = get_default_lucy_hardware()

    frontend: LanguageFrontend = fetch_frontend(program)
    builder, _ = _return_or_build(program, frontend.parse, hardware=live, compiler_config=config)

    # default parameters
    assert validate_circuit_length(program = program, compiler_config = config) == expected_result

    # max circuit duration input
    assert validate_circuit_length(program=program, compiler_config=config, max_circuit_duration= 1) == False










