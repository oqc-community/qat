# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from collections.abc import Iterable
from numbers import Number
from typing import List, Optional, TypeVar, Union

import numpy
from compiler_config.config import (
    CalibrationArguments,
    CompilerConfig,
    ErrorMitigationConfig,
    MetricsType,
    ResultsFormatting,
)

from qat.compiler.transform_passes import PhaseOptimisation, PostProcessingOptimisation
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.ir.pass_base import InvokerMixin, PassManager
from qat.ir.result_base import ResultManager
from qat.purr.compiler.builders import (
    FluidBuilderWrapper,
    InstructionBuilder,
    QuantumInstructionBuilder,
)
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.execution import (
    InstructionExecutionEngine,
    QuantumExecutionEngine,
    _binary,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel, get_cl2qu_index_mapping
from qat.purr.compiler.instructions import Instruction, is_generated_name
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.compiler.metrics import CompilationMetrics, MetricsMixin
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class RemoteCalibration:
    """
    Base class for any remote calibration executions. These are far more complicated
    blocks than purely a string of instructions and include nested executions and rely
    on classic Python code.
    """

    def run(
        self,
        model: QuantumHardwareModel,
        runtime: "QuantumRuntime",
        args: CalibrationArguments,
    ):
        raise ValueError("Calibration cannot be run at this time.")

    def arguments_type(self) -> type:
        """Returns the type of this calibrations arguments."""
        return CalibrationArguments


class QuantumExecutableBlock:
    """Generic executable block that can be run on a quantum runtime."""

    def run(self, runtime: "QuantumRuntime"):
        pass


class CalibrationWithArgs(QuantumExecutableBlock):
    """Wrapper for a calibration and argument combination."""

    def __init__(self, calibration: RemoteCalibration, args: CalibrationArguments = None):
        self.calibration = calibration
        self.args = args or CalibrationArguments()

    def run(self, runtime: "QuantumRuntime"):
        if self.calibration is None:
            raise ValueError("No calibration to run.")

        self.calibration.run(runtime.model, runtime, self.args)


AnyEngine = TypeVar("AnyEngine", bound=InstructionExecutionEngine, covariant=True)


class QuantumRuntime(MetricsMixin):
    def __init__(self, execution_engine: InstructionExecutionEngine, metrics=None):
        super().__init__()
        self.engine: AnyEngine = execution_engine
        self.compilation_metrics = metrics or CompilationMetrics()

    @property
    def model(self):
        return self.engine.model if self.engine is not None else None

    def _transform_results(
        self, results, format_flags: ResultsFormatting, repeats: Optional[int] = None
    ):
        """
        Transform the raw results into the format that we've been asked to provide. Look
        at individual transformation documentation for descriptions on what they do.
        """
        if len(results) == 0:
            return []

        # If we have no flags at all just infer structure simplification.
        if format_flags is None:
            format_flags = ResultsFormatting.DynamicStructureReturn

        if repeats is None:
            repeats = 1000

        def simplify_results(simplify_target):
            """
            To facilitate backwards compatability and being able to run low-level
            experiments alongside quantum programs we make some assumptions based upon
            form of the results.

            If all results have default variable names then the user didn't care about
            value assignment or this was a low-level experiment - in both cases, it
            means we can throw away the names and simply return the results in the order
            they were defined in the instructions.

            If we only have one result after this, just return that list directly
            instead, as it's probably just a single experiment.
            """
            if all([is_generated_name(k) for k in simplify_target.keys()]):
                if len(simplify_target) == 1:
                    return list(simplify_target.values())[0]
                else:
                    squashed_results = list(simplify_target.values())
                    if all(isinstance(val, numpy.ndarray) for val in squashed_results):
                        return numpy.array(squashed_results)
                    return squashed_results
            else:
                return simplify_target

        if ResultsFormatting.BinaryCount in format_flags:
            results = {key: _binary_count(val, repeats) for key, val in results.items()}

        def squash_binary(value):
            if isinstance(value, int):
                return str(value)
            elif all(isinstance(val, int) for val in value):
                return "".join([str(val) for val in value])

        if ResultsFormatting.SquashBinaryResultArrays in format_flags:
            results = {key: squash_binary(val) for key, val in results.items()}

        # Dynamic structure return is an ease-of-use flag to strip things that you know
        # your use-case won't use, such as variable names and nested lists.
        if ResultsFormatting.DynamicStructureReturn in format_flags:
            results = simplify_results(results)

        return results

    def _apply_error_mitigation(self, results, instructions, error_mitigation):
        if error_mitigation is None or error_mitigation == ErrorMitigationConfig.Empty:
            return results

        # TODO: add support for multiple registers
        # TODO: reconsider results length
        if len(results) > 1:
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error mitigation."
            )

        mapping = get_cl2qu_index_mapping(instructions, self.model)
        for mitigator in get_readout_mitigation(error_mitigation):
            new_result = mitigator.apply_error_mitigation(results, mapping, self.model)
            results[mitigator.name] = new_result
        return results  # TODO: new results object

    def run_calibration(
        self, calibrations: Union[CalibrationWithArgs, List[CalibrationWithArgs]]
    ):
        """Make 'calibration' distinct from 'quantum executable' for usabilities sake."""
        self.run_quantum_executable(calibrations)

    def run_quantum_executable(
        self, executables: Union[QuantumExecutableBlock, List[QuantumExecutableBlock]]
    ):
        if executables is None:
            return

        if not isinstance(executables, list):
            executables = [executables]

        for exe in executables:
            exe.run(self)

    def _common_execute(
        self,
        fexecute: callable,
        instructions,
        results_format=None,
        repeats=None,
        error_mitigation=None,
    ):
        """
        Executes these instructions against the current engine and returns the results.
        """
        if self.engine is None:
            raise ValueError("No execution engine available.")

        if isinstance(instructions, InstructionBuilder) or isinstance(
            instructions, FluidBuilderWrapper
        ):
            instructions = instructions.instructions

        if instructions is None or not any(instructions):
            raise ValueError(
                "No instructions passed to the process or stored for execution."
            )

        instructions = self.engine.optimize(instructions)
        self.engine.validate(instructions)
        self.record_metric(
            MetricsType.OptimizedInstructionCount, opt_inst_count := len(instructions)
        )
        log.info(f"Optimized instruction count: {opt_inst_count}")

        results = fexecute(instructions)
        results = self._transform_results(results, results_format, repeats)
        return self._apply_error_mitigation(results, instructions, error_mitigation)

    def _execute_with_interrupt(
        self,
        instructions,
        results_format=None,
        repeats=None,
        interrupt: Interrupt = NullInterrupt(),
        error_mitigation=None,
    ):
        """
        Executes these instructions against the current engine and returns the results.
        """

        def fexecute(instrs):
            return self.engine._execute_with_interrupt(instrs, interrupt)

        return self._common_execute(
            fexecute, instructions, results_format, repeats, error_mitigation
        )

    def execute(
        self, instructions, results_format=None, repeats=None, error_mitigation=None
    ):
        """
        Executes these instructions against the current engine and returns the results.
        """

        def fexecute(instrs):
            return self.engine.execute(instrs)

        return self._common_execute(
            fexecute, instructions, results_format, repeats, error_mitigation
        )


class NewQuantumRuntime(QuantumRuntime, InvokerMixin):
    """
    Uses the new pass infrastructure.

    Notice how polymorphic calls to XEngine.optimize() and XEngine.validate() are avoided. Instead, we have
    a flat structure of passes. This allows developers to focus on efficiently implementing a pass and easily test,
    demonstrate, and register passes without worrying too much about where it fits into the global compilation
    workflow.

    The NewQuantumRuntime deliberately recognises the builder as the only acceptable form of input "IR" and refuses
    to take in a bare list of instructions. This reduces the constant confusion of "builder" vs "instructions".

    The NewQuantumRuntime is also deliberately stripped out of any handling of compilation metrics. In fact, ideas
    similar to the new pass infrastructure can be applied to compilation metrics, that's why we're excluding them
    during this iteration partly because other pieces need to come together and partly because the current iteration
    needs to be kept light-weight and technically tractable.
    """

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingOptimisation()
            | InstructionValidation()
            | ReadoutValidation()
        )

    def _common_execute(
        self,
        fexecute: callable,
        builder: InstructionBuilder,
        results_format=None,
        repeats=None,
        error_mitigation=None,
    ):
        if self.engine is None:
            raise ValueError("No execution engine available.")

        if not isinstance(builder, InstructionBuilder):
            raise ValueError(
                f"Expected InstructionBuilder, but got {type(builder)} instead"
            )

        res_mgr = ResultManager()
        self.run_pass_pipeline(builder, res_mgr, self.model, self.engine)
        results = fexecute(builder)
        results = self._transform_results(results, results_format, repeats)
        return self._apply_error_mitigation(results, builder, error_mitigation)


def _binary_count(results_list, repeats):
    """
    Returns a dictionary of binary number: count. So for a two qubit register it'll return the various counts for
    ``00``, ``01``, ``10`` and ``11``.
    """

    def flatten(res):
        """
        Combine binary result from the QPU into composite key result.
        Aka '0110' or '0001'
        """
        if isinstance(res, Iterable):
            return "".join([flatten(val) for val in res])
        else:
            return str(res)

    def get_tuple(res, index):
        return [
            val[index] if isinstance(val, (List, numpy.ndarray)) else val for val in res
        ]

    binary_results = _binary(results_list)

    # If our results are a single qubit then pretend to be a register of one.
    if (
        isinstance(next(iter(binary_results), None), Number)
        and len(binary_results) == repeats
    ):
        binary_results = [binary_results]

    result_count = dict()
    for qubit_result in [list(get_tuple(binary_results, i)) for i in range(repeats)]:
        key = flatten(qubit_result)
        value = result_count.get(key, 0)
        result_count[key] = value + 1

    return result_count


def get_model(hardware: Union[QuantumExecutionEngine, QuantumHardwareModel]):
    if isinstance(hardware, QuantumExecutionEngine):
        return hardware.model
    return hardware


def get_runtime(hardware: Union[QuantumExecutionEngine, QuantumHardwareModel]):
    # We shouldn't really have an orphaned execution engine, but no harm in it.
    if isinstance(hardware, QuantumExecutionEngine):
        return hardware.model.create_runtime(hardware)
    elif isinstance(hardware, QuantumHardwareModel):
        return hardware.create_runtime()

    raise ValueError(
        f"{str(hardware)} is not a recognized hardware model or execution engine."
    )


def get_builder(
    model: Union[QuantumHardwareModel, InstructionExecutionEngine]
) -> QuantumInstructionBuilder:
    if isinstance(model, InstructionExecutionEngine):
        model = model.model
    return model.create_builder()


def execute_instructions(
    hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
    instructions: Union[List[Instruction], QuantumInstructionBuilder],
    config: CompilerConfig = None,
    executable_blocks: List[QuantumExecutableBlock] = None,
    repeats: Optional[int] = None,
    metrics: MetricsType = None,
    *args,
    **kwargs,
):
    if config is None:
        config = CompilerConfig()
    config.validate(hardware)

    active_runtime = get_runtime(hardware)

    active_runtime.run_quantum_executable(executable_blocks)
    results = active_runtime.execute(
        instructions,
        config.results_format,
        config.repeats,
        config.error_mitigation,
    )
    return results, active_runtime.compilation_metrics


def _execute_instructions_with_interrupt(
    hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
    instructions: Union[List[Instruction], QuantumInstructionBuilder],
    config: CompilerConfig = None,
    executable_blocks: List[QuantumExecutableBlock] = None,
    interrupt: Interrupt = NullInterrupt(),
    *args,
    **kwargs,
):
    if config is None:
        config = CompilerConfig()
    config.validate(hardware)

    active_runtime = get_runtime(hardware)
    active_runtime.run_quantum_executable(executable_blocks)

    results = active_runtime._execute_with_interrupt(
        instructions,
        config.results_format,
        config.repeats,
        interrupt,
        config.error_mitigation,
    )
    return results, active_runtime.compilation_metrics
