# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random
import re

import pytest
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    QuantumResultsFormat,
    Tket,
    TketOptimizations,
)

from qat import QAT
from qat.backend import FallthroughBackend
from qat.frontend import AutoFrontend
from qat.middleend import CustomMiddleend
from qat.model.loaders.purr import EchoModelLoader
from qat.pipelines.legacy.echo import (
    LegacyEchoCompilePipeline,
    LegacyEchoExecutePipeline,
    LegacyEchoPipeline,
    PipelineConfig,
)
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.purr.backends.echo import EchoEngine
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    Instruction,
    PostProcessing,
    ResultsProcessing,
    Return,
    Variable,
)
from qat.purr.qat import _return_or_build, fetch_frontend
from qat.runtime import LegacyRuntime

from tests.unit.utils.qasm_qir import (
    get_qasm2_path,
    get_qir_path,
    multi_reg_files,
    no_acquire_files,
    qasm2_files,
    qasm3_files,
    qir_files,
    short_file_name,
    skip_exec,
)


class TestEchoPipelines:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader().load()
        pipeline = LegacyEchoPipeline._build_pipeline(
            config=PipelineConfig(name="legacy_echo"),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "legacy_echo"
        assert pipeline.model == model
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, CustomMiddleend)
        assert isinstance(pipeline.backend, FallthroughBackend)
        assert isinstance(pipeline.runtime, LegacyRuntime)
        assert isinstance(pipeline.engine, EchoEngine)

    def test_build_compile_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader().load()
        pipeline = LegacyEchoCompilePipeline._build_pipeline(
            config=PipelineConfig(name="legacy_echo_compile"),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, CompilePipeline)
        assert pipeline.name == "legacy_echo_compile"
        assert pipeline.model == model
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, CustomMiddleend)
        assert isinstance(pipeline.backend, FallthroughBackend)

    def test_build_execute_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader().load()
        pipeline = LegacyEchoExecutePipeline._build_pipeline(
            config=PipelineConfig(name="legacy_echo_execute"),
            model=model,
            target_data=None,
        )
        assert isinstance(pipeline, ExecutePipeline)
        assert pipeline.name == "legacy_echo_execute"
        assert pipeline.model == model
        assert isinstance(pipeline.runtime, LegacyRuntime)
        assert isinstance(pipeline.engine, EchoEngine)


class EchoModelLoaderWithErrorMitigation(EchoModelLoader):
    """A model loader that applies a random error mitigation configuration to the loaded
    model."""

    def load(self, seed=None):
        """Load the Echo model and apply error mitigation."""
        hw = super().load()
        return self.apply_error_mitigation_setup(hw, seed=seed)

    def generate_random_linear(self, qubit_indices, random_data=True, seed=None):
        if seed is not None:
            random.seed(seed)
        output = {}
        for index in qubit_indices:
            random_0 = random.random() if random_data else 1
            random_1 = random.random() if random_data else 1
            output[str(index)] = {
                "0|0": random_0,
                "1|0": 1 - random_0,
                "1|1": random_1,
                "0|1": 1 - random_1,
            }
        return output

    def apply_error_mitigation_setup(self, hw, seed=None):
        lin_mit = self.generate_random_linear([q.index for q in hw.qubits], seed=seed)
        hw.error_mitigation = ErrorMitigation(ReadoutMitigation(linear=lin_mit))
        return hw


def equivalent_vars(self, other):
    """Used to overload '__eq__' so that a==b if they are equivalent."""
    return isinstance(self, type(other)) and (vars(self) == vars(other))


def equivalent_generated(self, other):
    """Like 'equivalent_vars' but does not care about the random number part of
    generated names."""

    gen_name_pattern = re.compile(r"generated_name_[0-9]+")

    def _strip_generated_rng(o):
        """Strip away the random number part of 'generated_name_4826354253464' allowing for
        simple equivalency check."""
        if isinstance(o, dict):
            new_o = {}  # new dictionary to create a copy
            for k, v in o.items():
                new_o[k] = _strip_generated_rng(v)
            return new_o
        elif isinstance(o, list):
            return [_strip_generated_rng(i) for i in o]
        elif isinstance(o, str):
            return gen_name_pattern.sub("generated_name", o)
        else:
            return o

    s_vars = _strip_generated_rng(vars(self))
    o_vars = _strip_generated_rng(vars(other))
    for d in [s_vars, o_vars]:
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = gen_name_pattern.sub("generated_name", v)
    return isinstance(self, type(other)) and (s_vars == o_vars)


files = [get_qir_path("bell_psi_minus.ll"), get_qasm2_path("logic_example.qasm")]


@pytest.mark.parametrize(
    "config_settings",
    [
        pytest.param({"enabled": False}, id="No_config"),
        pytest.param(
            {
                "enabled": True,
                "optimisation": False,
                "binary_count": False,
                "error_mitigation": False,
            },
            id="Default_config",
        ),
        pytest.param(
            {
                "enabled": True,
                "optimisation": True,
                "binary_count": False,
                "error_mitigation": False,
            },
            id="TketOptimizations.Two",
        ),
        pytest.param(
            {
                "enabled": True,
                "optimisation": False,
                "binary_count": True,
                "error_mitigation": True,
            },
            id="ErrorMitigation",
        ),
    ],
    scope="class",
)
@pytest.mark.parametrize(
    "source_file",
    qasm2_files | qasm3_files | qir_files,
    ids=short_file_name,
    scope="class",
)
class TestLegacyEchoPipelineParity:
    """Tests that compilation with the LegacyEchoPipeline has parity with purr."""

    model = EchoModelLoaderWithErrorMitigation(32).load()
    engine = EchoEngine(model)
    pipeline = LegacyEchoPipeline(config=PipelineConfig(name="legacy_echo"), model=model)

    def compiler_config(self, config_settings):
        """The compiler config is instantiated in this way so that a new instance is
        created each time, as the compiler has the ability to mutate the config, meaning
        a different config could be passed from program-to-program."""

        if not config_settings["enabled"]:
            return None
        kwargs = {}
        if config_settings["optimisation"]:
            kwargs["optimizations"] = Tket(TketOptimizations.Two)
        if config_settings["binary_count"]:
            kwargs["results_format"] = QuantumResultsFormat().binary_count()
        if config_settings["error_mitigation"]:
            kwargs["error_mitigation"] = (ErrorMitigationConfig.LinearMitigation,)
        return CompilerConfig(**kwargs)

    @pytest.fixture(scope="class")
    def purr_config(self, config_settings, source_file):
        # source file is an arg purposely to ensure that the config is created anew for
        # each source file, as the compiler has the ability to mutate the config.
        return self.compiler_config(config_settings)

    @pytest.fixture(scope="class")
    def qat_config(self, config_settings, source_file):
        # source file is an arg purposely to ensure that the config is created anew for
        # each source file, as the compiler has the ability to mutate the config.
        return self.compiler_config(config_settings)

    @pytest.fixture(scope="class")
    def purr_package(self, source_file, purr_config):
        frontend = fetch_frontend(str(source_file))
        builder, metrics = _return_or_build(
            str(source_file),
            frontend.parse,
            hardware=self.model,
            compiler_config=purr_config,
        )
        builder._instructions = self.engine.optimize(builder._instructions)
        self.engine.validate(builder._instructions)
        return builder, metrics

    @pytest.fixture(scope="class")
    def qat_package(self, source_file, qat_config):
        temp = QAT().compile(
            str(source_file),
            compiler_config=qat_config,
            pipeline=self.pipeline,
        )
        return temp

    def test_purr_compiles(self, purr_package):
        assert isinstance(purr_package[0], QuantumInstructionBuilder)

    def test_qat_compiles(self, qat_package):
        assert isinstance(qat_package[0], QuantumInstructionBuilder)

    def test_compile(self, purr_package, qat_package, monkeypatch):
        purr_package, purr_metrics = purr_package
        qat_package, qat_metrics = qat_package

        assert purr_metrics.optimized_circuit == qat_metrics.optimized_circuit
        with monkeypatch.context() as m:
            m.setattr(Instruction, "__eq__", equivalent_vars)
            m.setattr(Variable, "__eq__", equivalent_vars)
            m.setattr(Acquire, "__eq__", equivalent_generated)
            m.setattr(Assign, "__eq__", equivalent_generated)
            m.setattr(PostProcessing, "__eq__", equivalent_generated)
            m.setattr(ResultsProcessing, "__eq__", equivalent_generated)
            m.setattr(Return, "__eq__", equivalent_generated)
            assert purr_package.instructions == qat_package.instructions

    def test_execute(
        self,
        source_file,
        config_settings,
        purr_config,
        qat_config,
        purr_package,
        qat_package,
    ):
        short_name = short_file_name(source_file)
        if source_file in skip_exec:
            pytest.skip(
                f"Skipping execution for {short_name} as this program cannot be executed."
            )
        if config_settings.get("enabled", False) and config_settings.get(
            "error_mitigation", False
        ):
            if short_name in multi_reg_files:
                pytest.skip("ErrorMitigation not implemented for multiple registers.")
            elif short_name.startswith("qir-"):
                pytest.skip("ErrorMitigation not implemented for QIR files.")
            elif short_name in no_acquire_files:
                pytest.skip("ErrorMitigation not implemented for program with no acquire.")

        frontend = fetch_frontend(str(source_file))
        purr_res, _ = frontend.execute(
            purr_package[0], hardware=self.model, compiler_config=purr_config
        )

        qat_res, _ = QAT().execute(
            qat_package[0], compiler_config=qat_config, pipeline=self.pipeline
        )

        assert purr_res == qat_res
