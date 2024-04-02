# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import pytest

from qat import execute
from qat.purr.backends.verification import (
    Lucy,
    QPUVersion,
    VerificationError,
    get_verification_model,
    verify_program,
)
from qat.purr.compiler.config import CompilerConfig, Tket

from .qasm_utils import TestFileType, get_qasm2, get_test_file_path


class TestFirmwareVerificationEngines:
    def test_latest_lucy(self):
        model = get_verification_model(Lucy.Latest)
        execute(get_qasm2("basic.qasm"), model)

    def test_unknown_make(self):
        model = get_verification_model(QPUVersion("something", "123"))
        assert model is None

    @pytest.mark.parametrize(
        ("input_string", "file_type", "is_valid"),
        [
            ("primitives.qasm", TestFileType.QASM2, True),
            ("ghz.qasm", TestFileType.QASM3, True),
            ("ghz.qasm", TestFileType.QASM3, True),
            # ("bell_psi_plus.ll", TestFileType.QIR, True),
            ("cross_ressonance.qasm", TestFileType.OPENPULSE, True),
            ("long.qasm", TestFileType.QASM2, False),
        ],
    )
    def test_circuit_length_validation(self, input_string, file_type, is_valid):
        program = get_test_file_path(file_type, input_string)

        optim = Tket()
        optim.disable()
        config = CompilerConfig(optimizations=optim)

        try:
            verify_program(program, config, Lucy.Latest)
            assert is_valid is True
        except VerificationError as ex:
            assert is_valid is False
            assert "duration exceeds maximum" in str(ex)
