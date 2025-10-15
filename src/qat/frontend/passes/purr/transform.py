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
from qat.frontend.passes.analysis import InputAnalysisResult
from qat.ir.builder_factory import BuilderFactory
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qasm import CloudQasmParser, Qasm3Parser
from qat.purr.integrations.tket import run_tket_optimizations_qasm


class FlattenIR(TransformPass):
    """Flatten the IR by removing nested structures like InstructionBlocks."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        ir._instructions = ir.instructions
        return ir


class InputOptimisation(TransformPass):
    """Run third party optimisation passes on the incoming QASM."""

    def __init__(self, hardware: QuantumHardwareModel, *args, **kwargs):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model is used in TKET optimisations.
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
            qasm_string = run_tket_optimizations_qasm(
                qasm_string, optimizations.tket_optimizations, self.hardware
            )

        # TODO: [QK] Spend time looking at qiskit optimization and seeing if it's
        #   worth keeping around.
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
        """
        TODO: [QK] Current setup is unlikely to provide much benefit, refine settings
            before using.
        """
        if level is not None:
            try:
                optimized_circuits = transpile(
                    QuantumCircuit.from_qasm_str(qasm_string),
                    basis_gates=["u1", "u2", "u3", "cx"],
                    optimization_level=level,
                )
                qasm_string = qasm2.dumps(optimized_circuits)
            except TranspilerError:
                pass
                # log.warning(f"Qiskit transpile pass failed. {str(ex)}")

        return qasm_string


class Parse(TransformPass):
    """Parses the QASM/QIR input into IR."""

    def __init__(self, hardware: QuantumHardwareModel):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model is required to create Qat IR.
        """
        self.hardware = hardware

    def run(
        self,
        program: str,
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        """
        :param program: The program as a string (e.g. QASM or QIR).
        :param res_mgr: The results manager to look-up the :class:`InputAnalysisResults`.
        :param compiler_config: The compiler config should be provided by a keyword
            argument.
        """
        input_results = res_mgr.lookup_by_type(InputAnalysisResult)
        language = input_results.language
        builder = self.hardware.create_builder()
        parser = None
        if language == Languages.QIR:
            builder = self.parse_qir(program, compiler_config)
        elif language == Languages.Qasm2:
            parser = CloudQasmParser()
        elif language == Languages.Qasm3:
            parser = Qasm3Parser()
        if parser is not None:
            if compiler_config.results_format.format is not None:
                parser.results_format = compiler_config.results_format.format
            builder = parser.parse(builder, program)

        return (
            self.hardware.create_builder()
            .repeat(
                compiler_config.repeats,
                repetition_period=compiler_config.repetition_period,
                passive_reset_time=compiler_config.passive_reset_time,
            )
            .add(builder)
        )

    def parse_qir(self, qir_string, compiler_config):
        """Extracted from QIRFrontend"""
        # TODO: Resolve circular import
        from qat.purr.integrations.qir import QIRParser

        # TODO: Remove need for saving to file before parsing
        suffix = ".bc" if isinstance(qir_string, bytes) else ".ll"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            if suffix == ".ll":
                fp.write(qir_string.encode())
            else:
                fp.write(qir_string)
            fp.close()
            try:
                builder = BuilderFactory.create_builder(self.hardware)
                parser = QIRParser(self.hardware, builder=builder)
                if compiler_config.results_format.format is not None:
                    parser.results_format = compiler_config.results_format.format
                quantum_builder = parser.parse(fp.name)
            finally:
                os.remove(fp.name)
        return quantum_builder
