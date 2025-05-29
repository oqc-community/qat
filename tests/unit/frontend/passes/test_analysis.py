# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from pathlib import Path

import pytest
from compiler_config.config import Languages

from qat.core.pass_base import ResultManager
from qat.frontend.passes.analysis import InputAnalysis, InputAnalysisResult

from tests.conftest import tests_dir
from tests.unit.utils.qasm_qir import ProgramFileType, get_test_file_path


class TestInputAnalysis:
    @pytest.mark.parametrize(
        "input_file, language, input_type",
        [
            ("basic.qasm", Languages.Qasm2, ProgramFileType.QASM2),
            ("basic.qasm", Languages.Qasm3, ProgramFileType.QASM3),
            ("zmap.qasm", Languages.Qasm3, ProgramFileType.OPENPULSE),
            ("basic.ll", Languages.QIR, ProgramFileType.QIR),
            ("hello.bc", Languages.QIR, ProgramFileType.QIR),
        ],
    )
    def test_input_path(self, input_file, language, input_type):
        file_path = str(get_test_file_path(input_type, input_file))

        res_mgr = ResultManager()
        InputAnalysis().run(file_path, res_mgr=res_mgr)

        result = res_mgr.lookup_by_type(InputAnalysisResult)
        assert result.language == language
        assert type(result.raw_input) in (str, bytes)

    @pytest.mark.parametrize(
        "input_file, language, input_type",
        [
            ("basic.qasm", Languages.Qasm2, ProgramFileType.QASM2),
            ("basic.qasm", Languages.Qasm3, ProgramFileType.QASM3),
            ("zmap.qasm", Languages.Qasm3, ProgramFileType.OPENPULSE),
            ("basic.ll", Languages.QIR, ProgramFileType.QIR),
            ("hello.bc", Languages.QIR, ProgramFileType.QIR),
        ],
    )
    def test_input_string(self, input_file, language, input_type):
        file_path = get_test_file_path(input_type, input_file)
        file_mode = "rb" if input_file.endswith(".bc") else "r"
        with file_path.open(file_mode) as file:
            input_string = file.read()

        res_mgr = ResultManager()
        InputAnalysis().run(input_string, res_mgr=res_mgr)

        result = res_mgr.lookup_by_type(InputAnalysisResult)
        assert result.raw_input == input_string
        assert result.language == language

    def test_nonsense_qasm_throws(self):
        invalid_file = str(
            Path(tests_dir, "files", "qasm", "qasm3", "nonsense", "nonsense.qasm")
        )

        with pytest.raises(ValueError, match="Unable to determine input language."):
            InputAnalysis().run(invalid_file, res_mgr=ResultManager())
