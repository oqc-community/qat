# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

import pytest
from compiler_config.config import (
    CompilerConfig,
    InlineResultsProcessing,
    QuantumResultsFormat,
    Tket,
)

from qat.frontend.parsers.qir import QIRParser as PydQIRParser
from qat.frontend.qir import QIRFrontend, is_qir_path, is_qir_str, load_qir_file
from qat.ir.instruction_builder import InstructionBuilder as PydInstructionBuilder
from qat.ir.instructions import Assign as PydAssign
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.model.loaders.legacy.echo import EchoModelLoader
from qat.purr.compiler.builders import InstructionBuilder as LegInstructionBuilder
from qat.purr.compiler.devices import QubitCoupling
from qat.purr.compiler.instructions import Assign, Pulse, ResultsProcessing
from qat.purr.integrations.qir import QIRParser as LegQIRParser

from tests.unit.utils.models import get_jagged_echo_hardware
from tests.unit.utils.qasm_qir import (
    filename_ids,
    get_all_qir_paths,
    get_openpulse,
    get_qasm2,
    get_qasm3,
    get_qir_path,
)


def _get_qir_path(file_name):
    return str(get_qir_path(file_name))


class TestQIRLoadingMethods:
    @pytest.mark.parametrize(
        "file_name, expected",
        [
            ("mock-file.ll", True),
            ("mock-file.bc", True),
            ("mock-file.qasm", False),
        ],
    )
    def test_is_qir_path(self, file_name, expected):
        assert is_qir_path(file_name) == expected

    @pytest.mark.parametrize(
        "file_name, expected",
        [
            ("@__quantum__qis", True),
            ("@__quantum__qis__h", True),
            ("not a qir string", False),
        ],
    )
    def test_is_qir_str(self, file_name, expected):
        assert is_qir_str(file_name) == expected

    @pytest.mark.parametrize(
        "file_name, out_type",
        [
            ("basic.ll", str),
            ("hello.bc", bytes),
        ],
    )
    def test_load_qir_file(self, file_name, out_type):
        path = _get_qir_path(file_name)
        content = load_qir_file(path)
        assert isinstance(content, out_type)

    def test_load_qir_file_invalid_file_type_throws_error(self):
        path = get_qasm2("basic.qasm")
        with pytest.raises(ValueError, match="String expected to end in `.ll` or `.bc`."):
            load_qir_file(path)


class TestQIRFrontend:
    @pytest.fixture(scope="class")
    def legacy_model(self):
        return EchoModelLoader(32).load()

    @pytest.fixture(scope="class")
    def pyd_model(self, legacy_model):
        return convert_legacy_echo_hw_to_pydantic(legacy_model)

    @pytest.mark.parametrize(
        "model_type, parser_type",
        [("legacy_model", LegQIRParser), ("pyd_model", PydQIRParser)],
    )
    def test_parser_matching_model(self, request, model_type, parser_type):
        model = request.getfixturevalue(model_type)
        frontend = QIRFrontend(model)
        assert isinstance(frontend.parser, parser_type)

    def test_invalid_paths(self, legacy_model):
        frontend = QIRFrontend(legacy_model)
        with pytest.raises(FileNotFoundError):
            frontend.check_and_return_source(str(Path("\\very\\wrong.ll").absolute()))

        with pytest.raises(FileNotFoundError):
            frontend.check_and_return_source(str(Path("/very/wrong.ll").absolute()))

        with pytest.raises(FileNotFoundError):
            frontend.check_and_return_source("/very/wrong.bc")

    def test_check_valid_path(self, legacy_model):
        checked = QIRFrontend(legacy_model).check_and_return_source(
            _get_qir_path("generator-bell.ll")
        )
        assert isinstance(checked, str)
        assert checked != "generator-bell.ll"

    @pytest.mark.parametrize(
        "optim_config, place",
        [
            pytest.param(Tket().disable(), False, id="Disabled"),
            pytest.param(Tket().minimum(), True, id="Minimal"),
            pytest.param(Tket().default(), True, id="Default"),
        ],
    )
    def test_qir_placement(self, optim_config, place):
        model = get_jagged_echo_hardware(4, [2, 4, 5, 10])
        model.qubit_direction_couplings.extend(
            [
                QubitCoupling((10, 5), 98),
                QubitCoupling((5, 4), 85),
                QubitCoupling((4, 2), 92),
            ]
        )
        frontend = QIRFrontend(model)
        config = CompilerConfig(optimizations=optim_config)
        builder = frontend.emit(_get_qir_path("generator-bell.ll"), compiler_config=config)
        assert isinstance(builder, LegInstructionBuilder)
        active_qubits = set()
        for instruction in builder.instructions:
            if isinstance(instruction, Pulse):
                for target in instruction.quantum_targets:
                    active_qubits.add(int(target.id.split(".")[0][1:]))
        if place:
            assert list(active_qubits) == [10, 5]
        else:
            assert list(active_qubits) == [2, 4]

    @pytest.mark.parametrize(
        "model_type, builder_type, instruction_count",
        [
            ("legacy_model", LegInstructionBuilder, 181),
            ("pyd_model", PydInstructionBuilder, 182),
        ],
    )
    def test_base_profile_ops(self, request, model_type, builder_type, instruction_count):
        model = request.getfixturevalue(model_type)
        frontend = QIRFrontend(model)
        builder = frontend.emit(_get_qir_path("base_profile_ops.ll"))
        assert isinstance(builder, builder_type)
        assert len(builder.instructions) == instruction_count

    def test_resets_results_format(self, legacy_model):
        frontend = QIRFrontend(legacy_model)
        assert frontend.parser.results_format is QuantumResultsFormat().binary().format
        config = CompilerConfig(results_format=QuantumResultsFormat().raw())
        builder_raw = frontend.emit(
            _get_qir_path("generator-bell.ll"), compiler_config=config
        )
        result_processing = filter(
            lambda inst: isinstance(inst, ResultsProcessing), builder_raw.instructions
        )
        assert all(
            [
                inst.results_processing is InlineResultsProcessing.Raw
                for inst in result_processing
            ]
        )
        assert frontend.parser.results_format is QuantumResultsFormat().binary().format
        builder_bin = frontend.emit(_get_qir_path("generator-bell.ll"))
        result_processing = filter(
            lambda inst: isinstance(inst, ResultsProcessing), builder_bin.instructions
        )
        assert all(
            [
                inst.results_processing is InlineResultsProcessing.Binary
                for inst in result_processing
            ]
        )
        assert builder_bin is not builder_raw

    @pytest.mark.parametrize("qir_path", get_all_qir_paths(), ids=filename_ids)
    def test_check_and_return_source_with_qasm_2_files(self, qir_path, legacy_model):
        qir_path = str(qir_path)
        qir_str = QIRFrontend(legacy_model).check_and_return_source(qir_path)
        assert qir_str
        assert qir_str != qir_path
        qasm_str2 = QIRFrontend(legacy_model).check_and_return_source(qir_str)
        assert qir_str == qasm_str2

    @pytest.mark.parametrize(
        "qasm_path",
        [
            _get_qir_path("nonsense/nonsense.ll"),
            get_qasm3("basic.qasm"),
            get_qasm3("basic.qasm"),
            get_openpulse("acquire.qasm"),
            1337,
        ],
    )
    def test_check_and_return_source_with_invalid_programs(
        self, qasm_path: str, legacy_model
    ):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        res = QIRFrontend(legacy_model).check_and_return_source(qasm_path)
        assert not res

    def test_return_variables_are_reset_legacy(self, legacy_model):
        frontend = QIRFrontend(legacy_model)
        qir_str = _get_qir_path("bell_psi_plus.ll")
        builder = frontend.emit(qir_str)
        assert isinstance(builder, LegInstructionBuilder)
        assert frontend.parser.result_variables == []
        assign_insts = [inst for inst in builder.instructions if isinstance(inst, Assign)]
        assert len(assign_insts) == 1
        assign_vars = assign_insts[0].value
        assert isinstance(assign_vars, list)
        assert len(assign_vars) > 0

        new_builder = frontend.emit(qir_str)
        assert isinstance(new_builder, LegInstructionBuilder)
        new_assign_insts = [
            inst for inst in new_builder.instructions if isinstance(inst, Assign)
        ]
        assert len(new_assign_insts) == 1
        new_assign_vars = new_assign_insts[0].value
        assert isinstance(new_assign_vars, list)
        assert len(new_assign_vars) > 0
        assert len(new_assign_vars) == len(assign_vars)

    def test_return_variables_are_reset(self, pyd_model):
        frontend = QIRFrontend(pyd_model)
        qir_str = _get_qir_path("bell_psi_plus.ll")
        builder = frontend.emit(qir_str)
        assert isinstance(builder, PydInstructionBuilder)
        assert frontend.parser.result_variables == []
        assign_insts = [
            inst for inst in builder.instructions if isinstance(inst, PydAssign)
        ]
        assert len(assign_insts) == 1
        assign_vars = assign_insts[0].value
        assert isinstance(assign_vars, list)
        assert len(assign_vars) > 0

        new_builder = frontend.emit(qir_str)
        assert isinstance(new_builder, PydInstructionBuilder)
        new_assign_insts = [
            inst for inst in new_builder.instructions if isinstance(inst, PydAssign)
        ]
        assert len(new_assign_insts) == 1
        new_assign_vars = new_assign_insts[0].value
        assert isinstance(new_assign_vars, list)
        assert len(new_assign_vars) > 0
        assert len(new_assign_vars) == len(assign_vars)
