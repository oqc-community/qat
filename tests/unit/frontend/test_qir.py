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
from qat.frontend.qir import QIRFrontend
from qat.ir.instruction_builder import InstructionBuilder as PydInstructionBuilder
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.model.loaders.legacy.echo import EchoModelLoader
from qat.purr.compiler.builders import InstructionBuilder as LegInstructionBuilder
from qat.purr.compiler.devices import QubitCoupling
from qat.purr.compiler.instructions import Pulse, ResultsProcessing
from qat.purr.integrations.qir import QIRParser as LegQIRParser

from tests.unit.utils.models import get_jagged_echo_hardware
from tests.unit.utils.qasm_qir import get_qir_path


def _get_qir_path(file_name):
    return str(get_qir_path(file_name))


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
            ("pyd_model", PydInstructionBuilder, 178),
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
