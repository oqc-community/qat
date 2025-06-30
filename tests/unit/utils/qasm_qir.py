# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
from enum import Enum, auto
from importlib.util import find_spec
from pathlib import Path

from compiler_config.config import Qasm2Optimizations
from openqasm3 import ast

from qat.frontend.qir import load_qir_file
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import Pulse
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import (
    Qasm2Parser,
    Qasm3ParserBase,
    QasmContext,
    qasm_from_file,
)

from tests.conftest import tests_dir


class ProgramFileType(Enum):
    QASM2 = (auto(),)
    QASM3 = (auto(),)
    QIR = auto()
    OPENPULSE = auto()


def get_test_files_dir(ir_type: ProgramFileType) -> Path:
    if ir_type == ProgramFileType.QASM3:
        return Path(tests_dir, "files", "qasm", "qasm3")
    elif ir_type == ProgramFileType.QASM2:
        return Path(tests_dir, "files", "qasm", "qasm2")
    elif ir_type == ProgramFileType.QIR:
        return Path(tests_dir, "files", "qir")
    elif ir_type == ProgramFileType.OPENPULSE:
        return Path(tests_dir, "files", "qasm", "qasm3", "openpulse_tests")
    else:
        raise ValueError("Test file directory doesn't exist for this IR type.")


def get_test_file_path(ir_type: ProgramFileType, file_name) -> Path:
    return Path(get_test_files_dir(ir_type), file_name)


def get_qasm3_path(file_name):
    return get_test_file_path(ProgramFileType.QASM3, file_name)


def get_qasm2_path(file_name):
    return get_test_file_path(ProgramFileType.QASM2, file_name)


def get_qasm3(file_name):
    return qasm_from_file(get_test_file_path(ProgramFileType.QASM3, file_name))


def get_qasm2(file_name):
    return qasm_from_file(get_test_file_path(ProgramFileType.QASM2, file_name))


def get_qir_path(file_name: str) -> Path:
    return get_test_file_path(ProgramFileType.QIR, file_name)


def get_qir(file_name):
    file_path = get_test_file_path(ProgramFileType.QIR, file_name)
    if file_path.suffix == "":
        with file_path.open("r") as file_:
            return file_.read()
    return load_qir_file(file_path)


def get_openpulse(file_name):
    return qasm_from_file(get_test_file_path(ProgramFileType.OPENPULSE, file_name))


def get_all_qasm2_paths() -> set[Path]:
    dir = get_test_files_dir(ProgramFileType.QASM2)
    return set(Path(dir).glob("*.qasm"))


def get_all_qasm3_paths() -> set[Path]:
    dir = get_test_files_dir(ProgramFileType.QASM3)
    return set(Path(dir).glob("*.qasm"))


def get_all_qir_paths() -> set[Path]:
    dir = get_test_files_dir(ProgramFileType.QIR)
    return set(Path(dir).glob("*.ll")) | set(Path(dir).glob("*.bc"))


def get_all_openpulse_paths() -> set[Path]:
    dir = get_test_files_dir(ProgramFileType.OPENPULSE)
    return set(Path(dir).glob("*.qasm"))


def parse_and_apply_optimizations(
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


qasm3_base = """
OPENQASM 3.0;
bit[{N}] c;
qubit[{N}] q;
{gate_strings}
measure q -> c;
"""


qasm3_gates = {}


def get_default_qasm3_gate_qasms():
    if len(qasm3_gates) == 0:
        context = QasmContext()
        file_path = Path(
            find_spec("qiskit.qasm.libs").submodule_search_locations[0], "stdgates.inc"
        )
        node = ast.Include(filename=file_path)
        Qasm3ParserBase().visit(node, context)
        for name, defi in context.gates.items():
            needed_num_args = len(defi.arguments)
            arg_string = (
                ""
                if needed_num_args == 0
                else "(" + ", ".join(["1.2"] * needed_num_args) + ")"
            )
            N = len(defi.qubits)
            qubit_string = ", ".join([f"q[{i}]" for i in range(N)])
            gate_string = f"{name}{arg_string} {qubit_string};"
            qasm3_gates[name] = (N, gate_string)
    return list(qasm3_gates.values())


qasm2_base = """
OPENQASM 2.0;
include "qelib1.inc";
creg c[{N}];
qreg q[{N}];
{gate_strings}
measure q -> c;
"""


qasm2_gates = {}


def get_default_qasm2_gate_qasms():
    if len(qasm2_gates) == 0:
        intrinsics = Qasm2Parser()._get_intrinsics()
        for defi in intrinsics:
            name = defi.name
            needed_num_args = defi.num_params
            arg_string = (
                ""
                if needed_num_args == 0
                else "(" + ", ".join(["1.2"] * needed_num_args) + ")"
            )
            N = defi.num_qubits
            qubit_string = ", ".join([f"q[{i}]" for i in range(N)])
            gate_string = f"{name}{arg_string} {qubit_string};"
            qasm2_gates[name] = (N, gate_string)
    return list(qasm2_gates.values())


def get_pulses_from_builder(builder, shape_type=PulseShapeType.GAUSSIAN):
    """Get the gaussian pulses from the builder"""
    return [
        inst
        for inst in builder.instructions
        if (isinstance(inst, Pulse) and inst.shape == shape_type)
    ]


def filename_ids(val):
    if type(val) is str:
        strs = val.split("/")
        return str(strs[-1])
    elif hasattr(val, "name"):
        return str(val.name)
    else:
        return str(val)


def short_file_name(path):
    return "-".join([path.parent.name, path.name])


# Files that are not expected to pass tests are skipped.
skip_qasm2 = [
    "over_index_register.qasm",
    "invalid_w3x_couplings.qasm",
    "20qb.qasm",
    "huge.qasm",
    "mid_circuit_measure.qasm",
    "example_if.qasm",
]


skip_qasm3 = [
    "invalid_version.qasm",
    "no_header.qasm",
    "invalid_waveform.qasm",
    "invalid_port.qasm",
    "invalid_syntax.qasm",
    "invalid_frames.qasm",
    "invalid_couplings.qasm",
    "u_gate.qasm",
    "invalid_pulse_length.qasm",
]


skip_qir = [
    "teleportchain.ll",
    "bell_qir_measure.bc",
    "cudaq-ghz.ll",  # test is designed to fail for no TKET optims
    "base64_bitcode_ghz",
]


skip_exec = [
    (ProgramFileType.QASM3, "lark_parsing_test.qasm"),
    (ProgramFileType.QASM3, "ecr_override_test.qasm"),
    (ProgramFileType.QASM3, "tmp.qasm"),
    (ProgramFileType.QASM3, "cx_override_test.qasm"),
    (ProgramFileType.QASM3, "cnot_override_test.qasm"),
    (ProgramFileType.QASM3, "invalid_pulse_length.qasm"),
    (ProgramFileType.QIR, "complicated.ll"),
    (ProgramFileType.QIR, "base_profile_ops.ll"),
]


multi_reg_files = [
    "qasm2-split_measure_assign.qasm",
    "qasm2-basic_results_formats.qasm",
    "qasm2-ordered_cregs.qasm",
    "qasm2-example.qasm",
    "qasm2-decoupled.qasm",
]


no_acquire_files = [
    "qasm3-delay.qasm",
    "qasm3-ecr_test.qasm",
    "qasm3-redefine_defcal.qasm",
    "qasm3-arb_waveform.qasm",
    "qasm3-complex_gates_test.qasm",
]


skip_qasm2 = [
    get_test_file_path(ProgramFileType.QASM2, file_name) for file_name in skip_qasm2
]
qasm2_files = set(get_all_qasm2_paths()) - set(skip_qasm2)

skip_qasm3 = [
    get_test_file_path(ProgramFileType.QASM3, file_name) for file_name in skip_qasm3
]
qasm3_files = set(get_all_qasm3_paths()) - set(skip_qasm3)

skip_qir = [get_test_file_path(ProgramFileType.QIR, file_name) for file_name in skip_qir]
qir_files = set(get_all_qir_paths()) - set(skip_qir)

skip_exec = [get_test_file_path(file_type, file_name) for file_type, file_name in skip_exec]
