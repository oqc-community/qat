# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from enum import Enum, auto
from os.path import abspath, dirname, join

from compiler_config.config import Qasm2Optimizations

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import (
    Qasm2Parser,
    Qasm3ParserBase,
    QasmContext,
    qasm_from_file,
)


class ProgramFileType(Enum):
    QASM2 = (auto(),)
    QASM3 = (auto(),)
    QIR = auto()
    OPENPULSE = auto()


def get_test_files_dir(ir_type: ProgramFileType):
    if ir_type == ProgramFileType.QASM3:
        return abspath(join(dirname(__file__), "files", "qasm", "qasm3"))
    elif ir_type == ProgramFileType.QASM2:
        return abspath(join(dirname(__file__), "files", "qasm", "qasm2"))
    elif ir_type == ProgramFileType.QIR:
        return abspath(join(dirname(__file__), "files", "qir"))
    elif ir_type == ProgramFileType.OPENPULSE:
        return abspath(join(dirname(__file__), "files", "qasm", "qasm3", "openpulse_tests"))
    else:
        raise ValueError("Test file directory dosen't exist for this IR type.")


def get_test_file_path(ir_type: ProgramFileType, file_name):
    return join(get_test_files_dir(ir_type), file_name)


def get_qasm3(file_name):
    return qasm_from_file(get_test_file_path(ProgramFileType.QASM3, file_name))


def get_qasm2(file_name):
    return qasm_from_file(get_test_file_path(ProgramFileType.QASM2, file_name))


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


def get_default_qasm3_gate_qasms():
    context = QasmContext()
    Qasm3ParserBase().load_default_gates(context)
    gate_list = []
    for name, defi in context.gates.items():
        needed_num_args = len(defi.arguments)
        arg_string = (
            "" if needed_num_args == 0 else "(" + ", ".join(["0"] * needed_num_args) + ")"
        )
        needed_num_qubits = len(defi.qubits)
        N = max(needed_num_qubits, 2)
        qubit_string = ", ".join([f"q[{i}]" for i in range(needed_num_qubits)])
        gate_list.append(
            f"""OPENQASM 3.0;\nbit[{N}] c;\nqubit[{N}] q;\n{name}{arg_string} {qubit_string};\nmeasure q -> c;"""
        )
    return gate_list
