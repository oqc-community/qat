# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from enum import Enum, auto
from os.path import abspath, dirname, join

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import Qasm2Optimizations
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import Qasm2Parser, qasm_from_file


class TestFileType(Enum):
    QASM2 = (auto(),)
    QASM3 = (auto(),)
    QIR = auto()


def get_test_files_dir(ir_type: TestFileType):
    if ir_type == TestFileType.QASM3:
        return abspath(join(dirname(__file__), "files", "qasm", "qasm3"))
    elif ir_type == TestFileType.QASM2:
        return abspath(join(dirname(__file__), "files", "qasm", "qasm2"))
    elif ir_type == TestFileType.QIR:
        return abspath(join(dirname(__file__), "files", "qir"))
    else:
        raise ValueError("Test file directory dosen't exist for this IR type.")


def get_test_file_path(ir_type: TestFileType, file_name):
    return join(get_test_files_dir(ir_type), file_name)


def get_qasm3(file_name):
    return qasm_from_file(get_test_file_path(TestFileType.QASM3, file_name))


def get_qasm2(file_name):
    return qasm_from_file(get_test_file_path(TestFileType.QASM2, file_name))


def parse_and_apply_optimiziations(
    qasm_file_name, qubit_count=6, parser=None, opt_config=None
) -> InstructionBuilder:
    """
    Helper that builds a basic hardware, applies general optimizations, parses the QASM
    then returns the resultant builder.
    """
    hardware = get_default_echo_hardware(qubit_count)
    qasm = get_qasm2(qasm_file_name)

    if opt_config is None:
        opt_config = Qasm2Optimizations()

    qasm = DefaultOptimizers().optimize_qasm(qasm, hardware, opt_config)

    if parser is None:
        parser = Qasm2Parser()

    builder = parser.parse(get_builder(hardware), qasm)
    return builder
