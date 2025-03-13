# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.frontend.auto import AutoFrontend
from qat.frontend.qasm import Qasm2Frontend, Qasm3Frontend, load_qasm_file
from qat.frontend.qir import QIRFrontend, load_qir_file
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.purr.backends.echo import Connectivity, get_default_echo_hardware
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import Repeat

from tests.qat.frontend.test_qasm import qasm3_tests
from tests.qat.qasm_qir_utils import (
    get_all_qasm2_paths,
    get_all_qir_paths,
    get_qasm2,
    get_qasm3,
    get_qir,
)

qasm2_tests = get_all_qasm2_paths()
qir_tests = [
    test for test in get_all_qir_paths() if not test.endswith("base64_bitcode_ghz")
]


class TestAutoFrontend:

    # The hardware model being required here isn't ideal. It's because on instantiation
    # of the parser, get_builder(model) is called. Something to resolve later.
    frontend = AutoFrontend(get_default_echo_hardware(32))

    @pytest.mark.parametrize("invalid_type", ["invalid", True, 3.14])
    def invalid_type(self, invalid_type):
        with pytest.raises(TypeError):
            AutoFrontend(invalid_type)

    @pytest.mark.parametrize("qasm2_path", qasm2_tests)
    def test_assign_frontend_qasm2(self, qasm2_path):
        assigned_frontend = self.frontend.assign_frontend(qasm2_path)
        assert isinstance(assigned_frontend, Qasm2Frontend)
        qasm2_str = load_qasm_file(qasm2_path)
        assigned_frontend = self.frontend.assign_frontend(qasm2_str)
        assert isinstance(assigned_frontend, Qasm2Frontend)

    @pytest.mark.parametrize("qasm3_path", qasm3_tests)
    def test_assign_frontend_qasm3(self, qasm3_path):
        assigned_frontend = self.frontend.assign_frontend(qasm3_path)
        assert isinstance(assigned_frontend, Qasm3Frontend)
        qasm3_str = load_qasm_file(qasm3_path)
        assigned_frontend = self.frontend.assign_frontend(qasm3_str)
        assert isinstance(assigned_frontend, Qasm3Frontend)

    @pytest.mark.parametrize("qir_path", qir_tests)
    def test_assign_frontend_qir(self, qir_path):
        assigned_frontend = self.frontend.assign_frontend(qir_path)
        assert isinstance(assigned_frontend, QIRFrontend)
        qir_str = load_qir_file(qir_path)
        assigned_frontend = self.frontend.assign_frontend(qir_str)
        assert isinstance(assigned_frontend, QIRFrontend)

    @pytest.mark.parametrize(
        "program",
        [get_qasm2("basic.qasm"), get_qasm3("basic.qasm"), get_qir("bell_psi_plus.ll")],
    )
    def test_emit(self, program):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the respective parser tests."""
        builder = self.frontend.emit(program)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], Repeat)

    @pytest.mark.parametrize(
        "program",
        [get_qasm2("basic.qasm"), get_qasm3("basic.qasm"), get_qir("bell_psi_plus.ll")],
    )
    def test_legacy_vs_pydantic_hw(self, program):
        leg_hw = get_default_echo_hardware(32, connectivity=Connectivity.Ring)
        pyd_hw = convert_legacy_echo_hw_to_pydantic(leg_hw)

        leg_frontend = AutoFrontend(leg_hw)
        leg_builder = leg_frontend.emit(program)
        assert isinstance(leg_builder, QuantumInstructionBuilder)
        assert len(leg_builder.instructions) > 0

        pyd_frontend = AutoFrontend(pyd_hw)
        pyd_builder = pyd_frontend.emit(program)
        assert isinstance(pyd_builder, PydQuantumInstructionBuilder)
        assert pyd_builder.number_of_instructions > 0
