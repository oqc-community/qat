# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.purr.backends.live import LiveDeviceEngine
from qat.purr.backends.qiskit_simulator import QiskitBuilder
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.execution import InstructionExecutionEngine
from qat.purr.compiler.instructions import Repeat
from qat.runtime import BaseRuntime
from qat.runtime.connection import ConnectionMode


class LegacyRuntime(BaseRuntime):
    """A runtime that provides a wrapper around legacy engines for compatibility with the
    new stack."""

    def __init__(
        self,
        engine: InstructionExecutionEngine,
        results_pipeline: PassManager | None = None,
        connection_mode: ConnectionMode = ConnectionMode.DEFAULT,
    ):
        """
        :param engine: The execution engine for a target machine.
        :param results_pipeline: Optionally provided a pipeline for results processing. If
            not provided, a default pipeline is provided.
        :param connection_mode: Specifies how the connection is maintained.
        """
        if not isinstance(engine, InstructionExecutionEngine):
            raise ValueError(
                f"The engine provided has type {type(engine)}, which is not a "
                "`qat.purr.complier.execution.InstructionExecutionEngine`. Please use the "
                "`SimpleRuntime` for the refactored engines."
            )
        self.engine = engine
        self.connection_mode = connection_mode
        self.connect_engine(ConnectionMode.CONNECT_AT_BEGINNING)

        if not results_pipeline:
            results_pipeline = PassManager()
        self.results_pipeline = results_pipeline

    def execute(
        self,
        package: InstructionBuilder,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        **kwargs,
    ):
        """Fully execute QatIR against the hardware using a legacy execution engines.

        :param package: The program as an instruction builder.
        :param res_mgr: Optionally provide a results manager to save pass information.
        :param met_mgr: Optionally provide a metric manager to save pass information.
        :returns: Execution results.
        """

        if res_mgr is None:
            res_mgr = ResultManager()
        if met_mgr is None:
            met_mgr = MetricsManager()

        self.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        acquisitions = self.engine.execute(package.instructions)
        self.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)

        package.shots = self._determine_shots(package, kwargs.get("compiler_config"))

        return self.results_pipeline.run(
            acquisitions, res_mgr, met_mgr, package=package, **kwargs
        )

    def _determine_shots(
        self, package: InstructionBuilder, compiler_config: CompilerConfig | None
    ) -> int:
        """Determine the number of shots to use for execution.

        :param package: The "package" that is being executed. For a
            :class:`LegacyRuntime` that is a Builder.
        :compiler_config: The config information for the task.
        """
        if hasattr(package, "shots") and package.shots is not None:
            return package.shots
        elif (
            isinstance(insts := package.instructions, QiskitBuilder)
            and insts.shot_count is not None
        ):
            return insts.shot_count
        elif (
            repeat_inst := next(filter(lambda x: isinstance(x, Repeat), insts), None)
        ) is not None and repeat_inst.repeat_count is not None:
            return repeat_inst.repeat_count
        elif compiler_config is not None and compiler_config.repeats is not None:
            return compiler_config.repeats
        else:
            return None

    def connect_engine(self, flag: ConnectionMode):
        """Connect the engine according to the connection mode."""
        if not isinstance(self.engine, LiveDeviceEngine):
            return

        if flag in self.connection_mode:
            self.engine.startup()

    def disconnect_engine(self, flag: ConnectionMode):
        """Disconnect the engine according to the connection mode."""
        if not isinstance(self.engine, LiveDeviceEngine):
            return

        if flag in self.connection_mode:
            self.engine.shutdown()
