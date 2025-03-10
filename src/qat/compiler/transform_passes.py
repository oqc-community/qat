# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import os
import tempfile
from numbers import Number
from typing import Dict, List

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

from qat.compiler.analysis_passes import InputAnalysisResult
from qat.frontend.qasm_parsers.qasm2_parser import CloudQasmParser as PydCloudQasmParser
from qat.frontend.qasm_parsers.qasm3_parser import Qasm3Parser as PydQasm3Parser
from qat.integrations.tket import run_pyd_tket_optimizations
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import PhaseReset as PydPhaseReset
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.waveforms import Pulse as PydPulse
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.passes.metrics_base import MetricsManager
from qat.passes.pass_base import TransformPass
from qat.passes.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    AcquireMode,
    CustomPulse,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
)
from qat.purr.integrations.qasm import CloudQasmParser, Qasm3Parser
from qat.purr.integrations.tket import run_tket_optimizations_qasm


class PhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.

    Extracted from :meth:`qat.purr.compiler.execution.QuantumExecutionEngine.optimize`.
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        :param met_mgr: The metrics manager to store the number of instructions after
            optimisation.
        """

        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in ir.instructions:
            if isinstance(instruction, PhaseShift) and isinstance(
                instruction.phase, Number
            ):
                if accum_phaseshift := accum_phaseshifts.get(instruction.channel, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.channel] = PhaseShift(
                        instruction.channel, instruction.phase
                    )
            elif isinstance(instruction, (Pulse, CustomPulse)):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, List):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if quantum_target in accum_phaseshifts:
                        optimized_instructions.append(accum_phaseshifts.pop(quantum_target))
                optimized_instructions.append(instruction)
            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    accum_phaseshifts.pop(channel, None)
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir


class PydPhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.

    Extracted from :meth:`qat.purr.compiler.execution.QuantumExecutionEngine.optimize`.
    """

    def run(
        self,
        ir: QuantumInstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        :param met_mgr: The metrics manager to store the number of instructions after
            optimisation.
        """

        accum_phaseshifts: dict[str, PydPhaseShift] = dict()
        optimized_instructions: list = []
        for instruction in ir:
            if isinstance(instruction, PydPhaseShift):
                if accum_phaseshift := accum_phaseshifts.get(instruction.target, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.target] = PydPhaseShift(
                        targets=instruction.target, phase=instruction.phase
                    )

            elif isinstance(instruction, PydPulse):
                if (target := instruction.target) in accum_phaseshifts:
                    optimized_instructions.append(accum_phaseshifts.pop(target))
                optimized_instructions.append(instruction)

            elif isinstance(instruction, PydPhaseReset):
                for target in instruction.targets:
                    accum_phaseshifts.pop(target, None)
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir


class PostProcessingSanitisation(TransformPass):
    """Checks that the :class:`PostProcessing` instructions that follow an acquisition are
    suitable for the acquisition mode, and removes them if not.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.optimize`.
    """

    def run(
        self,
        ir: InstructionBuilder,
        _: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param met_mgr: The metrics manager to store the number of instructions after
            optimisation.
        """

        pp_insts = [val for val in ir.instructions if isinstance(val, PostProcessing)]
        discarded = []
        for pp in pp_insts:
            if pp.acquire.mode == AcquireMode.SCOPE:
                if (
                    pp.process == PostProcessType.MEAN
                    and ProcessAxis.SEQUENCE in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)

            elif pp.acquire.mode == AcquireMode.INTEGRATOR:
                if (
                    pp.process == PostProcessType.DOWN_CONVERT
                    and ProcessAxis.TIME in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)
                if (
                    pp.process == PostProcessType.MEAN
                    and ProcessAxis.TIME in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)
        ir.instructions = [val for val in ir.instructions if val not in discarded]
        met_mgr.record_metric(MetricsType.OptimizedInstructionCount, len(ir.instructions))
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
        :param program: The program as a string (e.g. QASM or QIR), or filepath to the
            program.
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
            .repeat(compiler_config.repeats, compiler_config.repetition_period)
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
                parser = QIRParser(self.hardware)
                if compiler_config.results_format.format is not None:
                    parser.results_format = compiler_config.results_format.format
                quantum_builder = parser.parse(fp.name)
            finally:
                os.remove(fp.name)
        return quantum_builder


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
            .repeat(compiler_config.repeats, compiler_config.repetition_period)
            .__add__(builder)
        )

    def parse_qir(self, qir_string, compiler_config):
        """Extracted from QIRFrontend"""
        # TODO: Resolve circular import
        from qat.frontend.qir_parser import QIRParser

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
