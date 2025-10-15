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
    get_optimizer_config,
)
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.transpiler import TranspilerError

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.frontend.parsers import CloudQasmParser as PydCloudQasmParser
from qat.frontend.parsers import Qasm3Parser as PydQasm3Parser
from qat.frontend.passes.analysis import InputAnalysisResult
from qat.integrations.tket import run_pyd_tket_optimizations
from qat.ir.builder_factory import BuilderFactory
from qat.ir.instruction_builder import (
    PydQuantumInstructionBuilder,
    QuantumInstructionBuilder,
)
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class InputOptimisation:
    def __init__(self, hardware: PydHardwareModel, *args, **kwargs):
        """Instantiate the pass with a hardware model.

        :param hardware: The hardware model is used in TKET optimisations.
        """
        self.hardware = hardware

    def run(
        self,
        program: str,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        """
        :param program: The program as a string (e.g. QASM or QIR), or filepath to the
            program.
        :param res_mgr: The results manager to look-up the :class:`InputAnalysisResults`.
        :param met_mgr: The metrics manager to save the optimised circuit.
        :param compiler_config: The compiler config should be provided by a keyword
            argument.
        """

        input_results = res_mgr.lookup_by_type(InputAnalysisResult)
        language = input_results.language
        program = input_results.raw_input
        if compiler_config.optimizations is None:
            compiler_config.optimizations = get_optimizer_config(language)
        if language in (Languages.Qasm2, Languages.Qasm3):
            program = self.run_qasm_optimisation(
                program, compiler_config.optimizations, met_mgr
            )
        return program

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

    def run_qiskit_optimization(self, qasm_string, level):
        if level is not None:
            try:
                optimized_circuits = transpile(
                    QuantumCircuit.from_qasm_str(qasm_string),
                    basis_gates=["u1", "u2", "u3", "cx"],
                    optimization_level=level,
                )
                qasm_string = qasm2.dumps(optimized_circuits)
            except TranspilerError as e:
                log.warning(f"Qiskit transpile pass failed. {str(e)}")

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
                parser = QIRParser(results_format=compiler_config.results_format.format)
                builder = BuilderFactory.create_builder(self.hw_model)
                builder = parser.parse(builder, fp.name)
            finally:
                os.remove(fp.name)
        return builder


class FlattenIR(TransformPass):
    """Flatten the IR by removing nested structures like InstructionBlocks."""

    def run(self, ir: PydQuantumInstructionBuilder, *args, **kwargs):
        return ir.flatten()


PydInputOptimisation = InputOptimisation
PydFlattenIR = FlattenIR
