# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from typing import List, TypeVar, Union

from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.config import (
    CalibrationArguments,
    CompilerConfig,
    MetricsType,
    ResultsFormatting,
)
from qat.purr.compiler.execution import InstructionExecutionEngine, QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Repeat
from qat.purr.compiler.metrics import CompilationMetrics, MetricsMixin
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

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
        args: CalibrationArguments
    ):
        raise ValueError("Calibration cannot be run at this time.")

    def arguments_type(self) -> type:
        """ Returns the type of this calibrations arguments. """
        return CalibrationArguments


class QuantumExecutableBlock:
    """ Generic executable block that can be run on a quantum runtime. """
    def run(self, runtime: "QuantumRuntime"):
        pass


class CalibrationWithArgs(QuantumExecutableBlock):
    """ Wrapper for a calibration and argument combination. """
    def __init__(self, calibration: RemoteCalibration, args: CalibrationArguments = None):
        self.calibration = calibration
        self.args = args or CalibrationArguments()

    def run(self, runtime: "QuantumRuntime"):
        if self.calibration is None:
            raise ValueError("No calibration to run.")

        self.calibration.run(runtime.model, runtime, self.args)


AnyEngine = TypeVar('AnyEngine', bound=InstructionExecutionEngine, covariant=True)


class QuantumRuntime(MetricsMixin):
    def __init__(self, execution_engine: InstructionExecutionEngine, metrics=None):
        super().__init__()
        self.engine: AnyEngine = execution_engine
        self.compilation_metrics = metrics or CompilationMetrics()

    @property
    def model(self):
        return self.engine.model if self.engine is not None else None

    def run_calibration(
        self, calibrations: Union[CalibrationWithArgs, List[CalibrationWithArgs]]
    ):
        """ Make 'calibration' distinct from 'quantum executable' for usabilities sake. """
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

    def execute(
        self, builder: InstructionBuilder, results_format: ResultsFormatting = None
    ):
        """
        Executes these instructions against the current engine and returns the results.
        """
        if self.engine is None:
            raise ValueError("No execution engine available.")

        instructions = builder.instructions

        # TODO: Change to return default value, not exception.
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

        return self.engine.execute(instructions, results_format)


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
    model: Union[QuantumHardwareModel, QuantumExecutionEngine]
) -> QuantumInstructionBuilder:
    if isinstance(model, QuantumExecutionEngine):
        model = model.model

    return model.create_builder()


def execute_instructions_via_config(
    hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
    instructions: InstructionBuilder,
    config: CompilerConfig,
):
    # If we don't have a repeat, default it.
    if not any(val for val in instructions.instructions if isinstance(val, Repeat)):
        instructions.repeat(config.repeats, config.repetition_period)

    # TODO: Look up later how much of a runtime hit this is, I'd hope
    #  hoisted to global and don't have to re-import every time.
    from qat.purr.backends.calibrations.remote import find_calibration
    calibrations = [find_calibration(arg) for arg in config.active_calibrations]

    return execute_instructions(
        hardware, instructions, config.results_format, calibrations, config.metrics
    )


def execute_instructions(
    hardware: Union[QuantumExecutionEngine, QuantumHardwareModel],
    instructions: InstructionBuilder,
    results_format=None,
    executable_blocks: List[QuantumExecutableBlock] = None,
    metrics: MetricsType = None
):
    with log_duration("Execution completed, took {} seconds."):
        active_runtime = get_runtime(hardware)
        active_runtime.initialize_metrics(metrics)
        active_runtime.run_quantum_executable(executable_blocks)
        return active_runtime.execute(instructions, results_format), active_runtime.compilation_metrics
