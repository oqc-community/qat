# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import (
    CompilerConfig,
    InlineResultsProcessing,
    Languages,
    MetricsType,
    Qiskit,
    QuantumResultsFormat,
    Tket,
)

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.frontend.passes.analysis import InputAnalysisResult
from qat.frontend.passes.legacy.transform import InputOptimisation, Parse
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import QubitCoupling
from qat.purr.compiler.instructions import ResultsProcessing

from tests.unit.utils.qasm_qir import get_qasm2, get_qasm3, get_qir


class TestInputOptimisation:
    @pytest.fixture(scope="class")
    def model(self):
        model = EchoModelLoader().load()
        model.qubit_direction_couplings = [
            QubitCoupling((0, 1)),
            QubitCoupling((1, 2)),
            QubitCoupling((2, 3), 98),
            QubitCoupling((3, 0)),
        ]
        return model

    @pytest.mark.parametrize(
        "program,language,raw_input",
        [
            pytest.param(
                "basic.qasm", Languages.Qasm2, get_qasm2("basic.qasm"), id="basic_qasm2"
            ),
            pytest.param(
                "basic.qasm", Languages.Qasm3, get_qasm3("basic.qasm"), id="basic_qasm3"
            ),
            pytest.param("basic.ll", Languages.QIR, get_qir("basic.ll"), id="basic_ll"),
            pytest.param("hello.bc", Languages.QIR, get_qir("hello.bc"), id="hello_bc"),
        ],
    )
    @pytest.mark.parametrize(
        "optimizations,expect_altered",
        [
            pytest.param(Tket().default(), True, id="Tket_default"),
            pytest.param(Tket().disable(), False, id="Tket_disable"),
            pytest.param(Tket().minimum(), True, id="Tket_minimum"),
            pytest.param(None, True, id="None"),
            pytest.param(Qiskit().default(), False, id="Qiskit_default"),
        ],
    )
    def test_optimizations(
        self, model, program, language, raw_input, optimizations, expect_altered
    ):
        met_mgr = MetricsManager()
        res_mgr = ResultManager()
        res_mgr.add(InputAnalysisResult(language=language, raw_input=raw_input))
        config = CompilerConfig(optimizations=optimizations)
        raw_output = InputOptimisation(model).run(
            program, res_mgr, met_mgr, compiler_config=config
        )
        if program.endswith(".bc"):
            assert isinstance(raw_output, bytes)
        else:
            assert isinstance(raw_output, str)
        if expect_altered and language is Languages.Qasm2:
            assert raw_output != raw_input
        else:
            assert raw_output == raw_input

    @pytest.mark.parametrize(
        "program,language,raw_input",
        [
            pytest.param(
                "basic.qasm", Languages.Qasm2, get_qasm2("basic.qasm"), id="basic_qasm2"
            ),
            pytest.param(
                "basic.qasm", Languages.Qasm3, get_qasm3("basic.qasm"), id="basic_qasm3"
            ),
            pytest.param("basic.ll", Languages.QIR, get_qir("basic.ll"), id="basic_ll"),
            pytest.param("hello.bc", Languages.QIR, get_qir("hello.bc"), id="hello_bc"),
        ],
    )
    @pytest.mark.parametrize("metrics", MetricsType)
    def test_metrics_collection(self, model, program, language, raw_input, metrics):
        met_mgr = MetricsManager(enabled_metrics=metrics)
        res_mgr = ResultManager()
        res_mgr.add(InputAnalysisResult(language=language, raw_input=raw_input))
        config = CompilerConfig(metrics=metrics)
        raw_output = InputOptimisation(model).run(
            program, res_mgr, met_mgr, compiler_config=config
        )
        if MetricsType.OptimizedCircuit in metrics and language in (
            Languages.Qasm2,
            Languages.Qasm3,
        ):
            assert met_mgr.optimized_circuit == raw_output
        else:
            assert met_mgr.optimized_circuit is None


class TestParse:
    @pytest.fixture(scope="class")
    def model(self):
        return EchoModelLoader().load()

    @pytest.mark.parametrize(
        "program,language",
        [
            pytest.param(get_qasm2("basic.qasm"), Languages.Qasm2, id="basic_qasm2"),
            pytest.param(get_qasm3("basic.qasm"), Languages.Qasm3, id="basic_qasm3"),
            pytest.param(get_qir("basic.ll"), Languages.QIR, id="basic_ll"),
            pytest.param(get_qir("hello.bc"), Languages.QIR, id="hello_bc"),
        ],
    )
    @pytest.mark.parametrize(
        "results_format",
        [
            pytest.param(QuantumResultsFormat().raw(), id="raw"),
            pytest.param(QuantumResultsFormat().binary(), id="binary"),
            pytest.param(QuantumResultsFormat().binary_count(), id="binary_count"),
            pytest.param(None, id="None"),
        ],
    )
    def test_parse(self, model, program, language, results_format):
        res_mgr = ResultManager()
        res_mgr.add(InputAnalysisResult(language=language, raw_input=""))
        config = CompilerConfig(results_format=results_format)
        builder = Parse(model).run(program, res_mgr, compiler_config=config)
        assert isinstance(builder, InstructionBuilder)
        result_proccessing = filter(
            lambda inst: isinstance(inst, ResultsProcessing), builder.instructions
        )
        for inst in result_proccessing:
            if results_format is None:
                assert inst.results_processing == InlineResultsProcessing.Binary
            else:
                assert inst.results_processing == results_format.format
