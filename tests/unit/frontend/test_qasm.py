# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

import pytest

from qat.frontend.qasm import (
    Qasm2Frontend,
    Qasm3Frontend,
    get_qasm_version,
    is_qasm_path,
    is_qasm_str,
    load_qasm_file,
)
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import Repeat as PydRepeat
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.purr.backends.echo import Connectivity, get_default_echo_hardware
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import Repeat

from tests.unit.utils.qasm_qir import (
    ProgramFileType,
    get_all_openpulse_paths,
    get_all_qasm2_paths,
    get_all_qasm3_paths,
    get_openpulse,
    get_qasm2,
    get_qasm3,
    get_qir,
    get_test_files_dir,
)

qasm3_skips = [
    "ghz_v2.qasm",  # wrong header
    "no_header.qasm",  # no header
    "invalid_version.qasm",  # wrong header
]
qasm3_skips = [
    Path(get_test_files_dir(ProgramFileType.QASM3), file_name) for file_name in qasm3_skips
]

openpulse_skips = [
    "rabi_specstroscopy.qasm",  # no header
    "qubit_spectroscopy.qasm",  # no header
    "phase_tracking_test.qasm",  # no header
]

openpulse_skips = [
    Path(get_test_files_dir(ProgramFileType.OPENPULSE), file_name)
    for file_name in openpulse_skips
]

qasm3_tests = (
    (get_all_qasm3_paths() | get_all_openpulse_paths())
    - set(qasm3_skips)
    - set(openpulse_skips)
)


class TestIsQasmPath:
    @pytest.mark.parametrize("path", ["test.qasm", "test/test.qasm"])
    def test_is_qasm_path_returns_true(self, path):
        assert is_qasm_path(path)

    @pytest.mark.parametrize(
        "path", ["test.ll", "test.bc", "qasm.qir", "test.qasm.ll", "OPENQASM 2.0;", "test"]
    )
    def test_is_qasm_path_returns_false(self, path):
        assert is_qasm_path(path) == False


class TestIsQasmStr:
    @pytest.mark.parametrize(
        "qasm_str",
        [
            "OPENQASM 2.0;",
            "OPENQASM 3.0;",
            "OPENQASM 2.0; \\n // this is a comment",
            "// this is a comment \\n OPENQASM 2.0;",
            "OPENQASM 2.0; \\n qreg q[2]; \\n creg c[2] \\n h q; measure q -> c;",
        ],
    )
    def test_is_qasm_str_returns_true(self, qasm_str):
        assert is_qasm_str(qasm_str)

    @pytest.mark.parametrize(
        "qasm_str",
        [
            "test.qasm",
            "test/test.qasm",
            "qreg q[2]; \\n creg c[2] \\n h q; measure q -> c;",
        ],
    )
    def test_is_qasm_str_returns_false(self, qasm_str):
        assert is_qasm_str(qasm_str) == False


@pytest.mark.parametrize(
    ["qasm_str", "version"],
    [
        ("OPENQASM 2.0;", 2),
        ("OPENQASM 3.0;", 3),
        ("OPENQASM 2.0; \\n // this is a comment", 2),
        ("OPENQASM 3.0; \\n // this is a comment", 3),
        ("// this is a comment \\n OPENQASM 2.0;", 2),
        ("// this is a comment \\n OPENQASM 3.0;", 3),
        ("OPENQASM 2.0; \\n qreg q[2]; \\n creg c[2] \\n h q; measure q -> c;", 2),
        ("OPENQASM 3.0; \\n qreg q[2]; \\n creg c[2] \\n h q; measure q -> c;", 3),
    ],
)
def test_get_qasm_version(qasm_str, version):
    assert get_qasm_version(qasm_str) == version


class TestLoadQasmFile:
    @pytest.mark.parametrize("qasm_path", (get_all_qasm2_paths() | qasm3_tests))
    def test_files_load(self, qasm_path: str):
        src = load_qasm_file(qasm_path)
        assert isinstance(src, str)

    def test_wrong_extension_raises_error(self):
        src = "test.qir"
        with pytest.raises(ValueError):
            load_qasm_file(src)


class TestQasm2Frontend:

    @pytest.mark.parametrize("qasm_path", get_all_qasm2_paths())
    def test_check_and_return_source_with_qasm_2_files(self, qasm_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = str(qasm_path)
        qasm_str = Qasm2Frontend(None).check_and_return_source(qasm_path)
        assert qasm_str
        assert qasm_str != qasm_path
        qasm_str2 = Qasm2Frontend(None).check_and_return_source(qasm_str)
        assert qasm_str == qasm_str2

    @pytest.mark.parametrize(
        "qasm_path",
        [
            get_qasm3("basic.qasm"),
            get_openpulse("acquire.qasm"),
            get_qir("bell_psi_plus.ll"),
        ],
    )
    def test_check_and_return_source_with_invalid_programs(self, qasm_path: str):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = str(qasm_path)
        res = Qasm2Frontend(None).check_and_return_source(qasm_path)
        assert res == False

    def test_emit_qasm_2_files(self):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the QasmParser tests."""
        qasm2_str = get_qasm2("basic.qasm")

        # Legacy hardware model.
        model = get_default_echo_hardware(32, connectivity=Connectivity.Ring)
        builder = Qasm2Frontend(model).emit(qasm2_str)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], Repeat)

        # Pydantic hardware model.
        builder = Qasm2Frontend(convert_legacy_echo_hw_to_pydantic(model)).emit(qasm2_str)
        assert isinstance(builder, PydQuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], PydRepeat)


class TestQasm3Frontend:

    @pytest.mark.parametrize("qasm_path", qasm3_tests)
    def test_check_and_return_source_with_qasm_3_files(self, qasm_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = str(qasm_path)
        qasm_str = Qasm3Frontend(None).check_and_return_source(qasm_path)
        assert qasm_str
        assert qasm_str != qasm_path
        qasm_str2 = Qasm3Frontend(None).check_and_return_source(qasm_str)
        assert qasm_str == qasm_str2

    @pytest.mark.parametrize(
        "qasm_path",
        [
            get_qasm2("basic.qasm"),
            get_qir("bell_psi_plus.ll"),
        ],
    )
    def test_check_and_return_source_with_invalid_programs(self, qasm_path: str):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = str(qasm_path)
        res = Qasm3Frontend(None).check_and_return_source(qasm_path)
        assert res == False

    def test_emit_qasm_3_files(self):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the QasmParser tests."""
        qasm3_str = get_qasm3("basic.qasm")

        # Legacy hardware model.
        model = get_default_echo_hardware(32, connectivity=Connectivity.Ring)
        builder = Qasm3Frontend(model).emit(qasm3_str)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], Repeat)

        # Pydantic hardware model.
        builder = Qasm3Frontend(convert_legacy_echo_hw_to_pydantic(model)).emit(qasm3_str)
        assert isinstance(builder, PydQuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], PydRepeat)
