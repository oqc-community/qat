# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import os
import tempfile

from compiler_config.config import (
    CompilerConfig,
    Languages,
    MetricsType,
    Qiskit,
    QiskitOptimizations,
    Tket,
    TketOptimizations,
)

from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.frontend.parsers import CloudQasmParser as PydCloudQasmParser
from qat.frontend.parsers import Qasm3Parser as PydQasm3Parser
from qat.frontend.passes.analysis import InputAnalysisResult
from qat.frontend.passes.purr.transform import InputOptimisation
from qat.integrations.tket import run_pyd_tket_optimizations
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel


class PydInputOptimisation(InputOptimisation):
    def run_qasm_optimisation(self, qasm_string, optimizations, met_mgr, *args, **kwargs):
        """Extracted from DefaultOptimizers.optimize_qasm"""

        if (
            isinstance(optimizations, Tket)
            and optimizations.tket_optimizations != TketOptimizations.Empty
        ):
            qasm_string = run_pyd_tket_optimizations(
                qasm_string, optimizations.tket_optimizations, self.hardware
            )

        if (
            isinstance(optimizations, Qiskit)
            and optimizations.qiskit_optimizations != QiskitOptimizations.Empty
        ):
            qasm_string = self.run_qiskit_optimization(
                qasm_string, optimizations.qiskit_optimizations
            )

        met_mgr.record_metric(MetricsType.OptimizedCircuit, qasm_string)
        return qasm_string


class PydParse(TransformPass):
    def __init__(self, hw_model: PydHardwareModel):
        self.hw_model = hw_model

    def run(
        self,
        program: str,
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        input_results = res_mgr.lookup_by_type(InputAnalysisResult)
        language = input_results.language
        builder = QuantumInstructionBuilder(self.hw_model)
        parser = None
        if language == Languages.QIR:
            builder = self.parse_qir(program, compiler_config)
        elif language == Languages.Qasm2:
            parser = PydCloudQasmParser()
        elif language == Languages.Qasm3:
            parser = PydQasm3Parser()
        if parser is not None:
            if compiler_config.results_format.format is not None:
                parser.results_format = compiler_config.results_format.format
            builder = parser.parse(builder, program)

        return (
            QuantumInstructionBuilder(self.hw_model)
            .repeat(compiler_config.repeats)
            .__add__(builder)
        )

    def parse_qir(self, qir_string, compiler_config):
        """Extracted from QIRFrontend"""
        # TODO: Resolve circular import
        from qat.frontend.parsers import QIRParser

        # TODO: Remove need for saving to file before parsing
        suffix = ".bc" if isinstance(qir_string, bytes) else ".ll"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            if suffix == ".ll":
                fp.write(qir_string.encode())
            else:
                fp.write(qir_string)
            fp.close()
            try:
                parser = QIRParser(self.hw_model)
                if compiler_config.results_format.format is not None:
                    parser.results_format = compiler_config.results_format.format
                builder = parser.parse(fp.name)
            finally:
                os.remove(fp.name)
        return builder
