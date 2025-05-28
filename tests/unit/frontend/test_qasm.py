# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

import pytest
from compiler_config.config import CompilerConfig, Languages, Tket

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager, ResultModel
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
from qat.model.loaders.legacy.echo import Connectivity, EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import Repeat

from tests.unit.utils.qasm_qir import (
    ProgramFileType,
    filename_ids,
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
            "OpenQasm 2.0;",
            "OPEN QASM 2.0;",
        ],
    )
    def test_is_qasm_str_returns_false(self, qasm_str):
        assert not is_qasm_str(qasm_str)


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
    @pytest.mark.parametrize(
        "qasm_path", (get_all_qasm2_paths() | qasm3_tests), ids=filename_ids
    )
    def test_files_load(self, qasm_path: str):
        src = load_qasm_file(qasm_path)
        assert isinstance(src, str)

    def test_wrong_extension_raises_error(self):
        src = "test.qir"
        with pytest.raises(ValueError):
            load_qasm_file(src)


class TestQasm2Frontend:
    @staticmethod
    def qasm2_frontend():
        model = EchoModelLoader(32, connectivity=Connectivity.Ring).load()
        return Qasm2Frontend(model)

    @pytest.mark.parametrize("qasm_path", get_all_qasm2_paths(), ids=filename_ids)
    def test_check_and_return_source_with_qasm_2_files(self, qasm_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = str(qasm_path)
        qasm_str = self.qasm2_frontend().check_and_return_source(qasm_path)
        assert qasm_str
        assert qasm_str != qasm_path
        qasm_str2 = self.qasm2_frontend().check_and_return_source(qasm_str)
        assert qasm_str == qasm_str2

    @pytest.mark.parametrize(
        "qasm_path",
        [
            get_qasm2("nonsense/nonsense.qasm"),
            get_qasm3("basic.qasm"),
            get_openpulse("acquire.qasm"),
            get_qir("bell_psi_plus.ll"),
            1337,
        ],
    )
    def test_check_and_return_source_with_invalid_programs(self, qasm_path: str):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        res = self.qasm2_frontend().check_and_return_source(qasm_path)
        assert not res

    def test_emit_qasm_2_files(self):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the QasmParser tests."""
        qasm2_str = get_qasm2("basic.qasm")

        # Legacy hardware model.
        model = EchoModelLoader(32, connectivity=Connectivity.Ring).load()
        builder = self.qasm2_frontend().emit(qasm2_str)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], Repeat)

        # Pydantic hardware model.
        builder = Qasm2Frontend(convert_legacy_echo_hw_to_pydantic(model)).emit(qasm2_str)
        assert isinstance(builder, PydQuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], PydRepeat)

    @pytest.mark.parametrize(
        "qasm_path",
        [
            get_qasm2("nonsense/nonsense.qasm"),
            get_qasm3("basic.qasm"),
            get_openpulse("acquire.qasm"),
            get_qir("bell_psi_plus.ll"),
            1337,
        ],
    )
    def test_emit_raises_error_with_invalid_programs(self, qasm_path):
        with pytest.raises(ValueError):
            self.qasm2_frontend().emit(qasm_path)

    def test_results_manager_collects_results(self):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the respective parser tests."""
        qasm2_str = get_qasm2("basic.qasm")
        result_manager = ResultManager()
        assert len(result_manager.results) == 0
        self.qasm2_frontend().emit(qasm2_str, res_mgr=result_manager)
        assert len(result_manager.results) == 1
        result = list(result_manager.results)[0]
        assert isinstance(result, ResultModel)
        assert result.value.language == Languages.Qasm2
        assert result.value.raw_input == qasm2_str

    @pytest.mark.parametrize("disable_optimizations", [True, False])
    def test_metrics_manager_collects_metrics(self, disable_optimizations):
        qasm2_str = get_qasm2("basic.qasm")
        metrics_manager = MetricsManager()
        assert metrics_manager.optimized_circuit is None
        assert metrics_manager.optimized_instruction_count is None
        compiler_config = (
            CompilerConfig(optimizations=Tket().disable())
            if disable_optimizations
            else None
        )
        self.qasm2_frontend().emit(
            qasm2_str, met_mgr=metrics_manager, compiler_config=compiler_config
        )

        if disable_optimizations:
            assert metrics_manager.optimized_circuit == qasm2_str
        else:
            assert metrics_manager.optimized_circuit is not None
            assert metrics_manager.optimized_circuit != qasm2_str
        assert metrics_manager.optimized_instruction_count is None


class TestQasm3Frontend:
    @staticmethod
    def qasm3_frontend():
        model = EchoModelLoader(32, connectivity=Connectivity.Ring).load()
        return Qasm3Frontend(model)

    @pytest.mark.parametrize("qasm_path", qasm3_tests, ids=filename_ids)
    def test_check_and_return_source_with_qasm_3_files(self, qasm_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = str(qasm_path)
        qasm_str = self.qasm3_frontend().check_and_return_source(qasm_path)
        assert qasm_str
        assert qasm_str != qasm_path
        qasm_str2 = self.qasm3_frontend().check_and_return_source(qasm_str)
        assert qasm_str == qasm_str2

    @pytest.mark.parametrize(
        "qasm_path",
        [
            get_qasm3("nonsense/nonsense.qasm"),
            get_qasm2("basic.qasm"),
            get_qir("bell_psi_plus.ll"),
            1337,
        ],
    )
    def test_check_and_return_source_with_invalid_programs(self, qasm_path: str):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        qasm_path = qasm_path
        res = self.qasm3_frontend().check_and_return_source(qasm_path)
        assert res == False

    def test_emit_qasm_3_files(self):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the QasmParser tests."""
        qasm3_str = get_qasm3("basic.qasm")

        # Legacy hardware model.
        model = EchoModelLoader(32, connectivity=Connectivity.Ring).load()
        builder = self.qasm3_frontend().emit(qasm3_str)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], Repeat)

        # Pydantic hardware model.
        builder = Qasm3Frontend(convert_legacy_echo_hw_to_pydantic(model)).emit(qasm3_str)
        assert isinstance(builder, PydQuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], PydRepeat)

    @pytest.mark.parametrize(
        "qasm_path",
        [
            get_qasm3("nonsense/nonsense.qasm"),
            get_qasm2("basic.qasm"),
            get_qir("bell_psi_plus.ll"),
            1337,
        ],
    )
    def test_emit_raises_error_with_invalid_programs(self, qasm_path):
        with pytest.raises(ValueError):
            self.qasm3_frontend().emit(qasm_path)
