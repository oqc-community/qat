# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Optional

from compiler_config.config import CompilerConfig, MetricsType

from qat.backend.validation_passes import HardwareConfigValidity
from qat.compiler.analysis_passes import InputAnalysis
from qat.compiler.transform_passes import (
    InputOptimisation,
    InputOptimisationResult,
    Parse,
    PhaseOptimisation,
    PostProcessingOptimisation,
)
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.ir.pass_base import PassManager, QatIR
from qat.ir.result_base import ResultManager
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel, get_cl2qu_index_mapping
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.compiler.runtime import get_runtime
from qat.purr.qatconfig import QatConfig
from qat.qat import QATInput
from qat.runtime.analysis_passes import CalibrationAnalysis, CalibrationAnalysisResult
from qat.runtime.transform_passes import ErrorMitigation, ResultTransform


class QAT:
    def __init__(self, hardware_model: Optional[QuantumHardwareModel] = None):
        self.config = QatConfig()
        self.hardware_model = hardware_model

    def compile(self, program: QATInput, compiler_config: Optional[CompilerConfig] = None):
        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)
        compilation_results = ResultManager()
        pipeline = self.build_compile_pipeline(compiler_config)
        ir = QatIR(program)
        pipeline.run(ir, compilation_results)
        metrics.record_metric(
            MetricsType.OptimizedCircuit,
            compilation_results.lookup_by_type(InputOptimisationResult).optimised_circuit,
        )
        return ir.value, metrics

    def execute(
        self,
        builder: InstructionBuilder,
        compiler_config: Optional[CompilerConfig] = None,
    ):
        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        execution_results = ResultManager()
        metrics = CompilationMetrics()
        metrics.enable(compiler_config.metrics)
        engine = self.hardware_model.create_engine()
        pipeline = self.build_pre_execute_pipeline(compiler_config)
        ir = QatIR(builder)
        pipeline.run(ir, execution_results, self.hardware_model, engine)

        # TODO: Improve calibration handling
        calibrations = execution_results.lookup_by_type(
            CalibrationAnalysisResult
        ).calibration_executables
        active_runtime = get_runtime(self.hardware_model)
        active_runtime.run_quantum_executable(calibrations)

        metrics.record_metric(
            MetricsType.OptimizedInstructionCount, len(builder.instructions)
        )
        results = engine.execute(
            builder.instructions,
        )

        # TODO: Should this be a pass in a pre-execution Pipeline?
        try:
            index_mapping = get_cl2qu_index_mapping(
                builder.instructions, self.hardware_model
            )
        except ValueError:
            index_mapping = {}

        # Result processing pipeline
        pipeline = self.build_result_processing_pipeline(compiler_config)
        ir = QatIR(value=results)
        pipeline.run(ir, execution_results, mapping=index_mapping)
        return ir.value, metrics

    @property
    def hardware_model(self):
        return self._hardware_model

    @hardware_model.setter
    def hardware_model(self, model: Optional[QuantumHardwareModel]):
        if model is None:
            model = get_default_RTCS_hardware()
        elif not isinstance(model, QuantumHardwareModel):
            raise ValueError(
                f"Expected value of type 'QuantumHardwareModel', got type '{type(model)}'"
            )
        self._hardware_model = model

    def build_compile_pipeline(self, compiler_config: CompilerConfig):
        pipeline = PassManager()
        return (
            pipeline
            | InputAnalysis()
            | InputOptimisation(self.hardware_model, compiler_config)
            | Parse(self.hardware_model, compiler_config)
        )

    def build_pre_execute_pipeline(self, compiler_config: CompilerConfig):
        pipeline = PassManager()
        return (
            pipeline
            | HardwareConfigValidity(self.hardware_model, compiler_config)
            | CalibrationAnalysis(compiler_config)
            | PhaseOptimisation()
            | PostProcessingOptimisation()
            | InstructionValidation()
            | ReadoutValidation()
        )

    def build_result_processing_pipeline(self, compiler_config: CompilerConfig):
        pipeline = PassManager()
        return (
            pipeline
            | ResultTransform(compiler_config)
            | ErrorMitigation(self.hardware_model, compiler_config)
        )
