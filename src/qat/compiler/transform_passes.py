import os
import tempfile
from numbers import Number
from typing import Dict, List, Optional

from attr import dataclass
from compiler_config.config import (
    CompilerConfig,
    Languages,
    Qiskit,
    QiskitOptimizations,
    Tket,
    TketOptimizations,
    get_optimizer_config,
)
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.transpiler import TranspilerError

from qat.compiler.analysis_passes import InputAnalysisResult
from qat.ir.pass_base import QatIR, TransformPass
from qat.ir.result_base import ResultInfoMixin, ResultManager
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
from qat.purr.integrations.tket import run_tket_optimizations


class PhaseOptimisation(TransformPass):
    """
    Extracted from QuantumExecutionEngine.optimize()
    """

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in builder.instructions:
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

        builder.instructions = optimized_instructions


class PostProcessingOptimisation(TransformPass):
    """
    Extracted from LiveDeviceEngine.optimize()
    Better pass name/id ?
    """

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        pp_insts = [val for val in builder.instructions if isinstance(val, PostProcessing)]
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
        builder.instructions = [val for val in builder.instructions if val not in discarded]


@dataclass
class InputOptimisationResult(ResultInfoMixin):
    optimised_circuit: str = None


class InputOptimisation(TransformPass):
    """Run third party optimisation passes on the incoming QASM."""

    def __init__(
        self,
        hardware: QuantumHardwareModel,
        compiler_config: Optional[CompilerConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hardware = hardware
        self.compiler_config = compiler_config or CompilerConfig()

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        optimisation_result = InputOptimisationResult()
        input_results = res_mgr.lookup_by_type(InputAnalysisResult)
        language = input_results.language
        program = input_results.raw_input
        if self.compiler_config.optimizations is None:
            self.compiler_config.optimizations = get_optimizer_config(language)
        if language in (Languages.Qasm2, Languages.Qasm3):
            program = self.run_qasm_optimisation(
                program, self.compiler_config.optimizations
            )
            optimisation_result.optimised_circuit = program
        ir.value = program
        res_mgr.add(optimisation_result)

    def run_qasm_optimisation(self, qasm_string, optimizations, *args, **kwargs):
        """Extracted from DefaultOptimizers.optimize_qasm"""

        if (
            isinstance(optimizations, Tket)
            and optimizations.tket_optimizations != TketOptimizations.Empty
        ):
            qasm_string = run_tket_optimizations(
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

        # self.record_metric(MetricsType.OptimizedCircuit, qasm_string)
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
    def __init__(
        self,
        hardware: QuantumHardwareModel,
        compiler_config: Optional[CompilerConfig] = None,
    ):
        self.hardware = hardware
        self.compiler_config = compiler_config or CompilerConfig()

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        input_results = res_mgr.lookup_by_type(InputAnalysisResult)
        language = input_results.language
        builder = self.hardware.create_builder()
        parser = None
        if language == Languages.QIR:
            builder = self.parse_qir(ir.value)
        elif language == Languages.Qasm2:
            parser = CloudQasmParser()
        elif language == Languages.Qasm3:
            parser = Qasm3Parser()
        if parser is not None:
            if self.compiler_config.results_format.format is not None:
                parser.results_format = self.compiler_config.results_format.format
            builder = parser.parse(builder, ir.value)
        ir.value = (
            self.hardware.create_builder()
            .repeat(self.compiler_config.repeats, self.compiler_config.repetition_period)
            .add(builder)
        )

    def parse_qir(self, qir_string):
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
                if self.compiler_config.results_format.format is not None:
                    parser.results_format = self.compiler_config.results_format.format
                quantum_builder = parser.parse(fp.name)
            finally:
                os.remove(fp.name)
        return quantum_builder
