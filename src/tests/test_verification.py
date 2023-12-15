# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from qat.purr.backends.verification import get_verification_model, Lucy, QPUVersion, verify_program
from qat.purr.compiler.config import Tket, CompilerConfig
from qat.qat import execute
from .qasm_utils import get_qasm2, TestFileType, get_test_file_path

import pytest


class TestFirmwareVerificationEngines:
    def test_latest_lucy(self):

        model = get_verification_model(Lucy.Latest)
        assert execute(get_qasm2("basic.qasm"), model)

    def test_unknown_make(self):
        model = get_verification_model(QPUVersion("something", "123"))
        assert model is None

    @pytest.mark.parametrize(("input_string", "file_type", "expected_result"),
                              [("primitives.qasm", TestFileType.QASM2, True),
                               ("ghz.qasm", TestFileType.QASM3, True),
                               ("ghz.qasm", TestFileType.QASM3, True),
                               ("bell_psi_plus.ll", TestFileType.QIR, True),
                               ("cross_ressonance.qasm", TestFileType.OPENPULSE, True),
                               ("long_qasm.qasm", TestFileType.QASM2, False)])
    def test_circuit_length_validation(self, input_string, file_type, expected_result):
        program = get_test_file_path(file_type, input_string)

        optim = Tket()
        optim.disable()
        config = CompilerConfig(optimizations=optim)

        assert verify_program(program, config, QPUVersion(make="Lucy", version="latest")) == expected_result

