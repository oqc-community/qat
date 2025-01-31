# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from typing import Optional

from qat.ir.metrics_base import MetricsManager
from qat.ir.pass_base import PassManager, QatIR
from qat.ir.result_base import ResultManager
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.runtime import BaseRuntime


class LegacyRuntime(BaseRuntime):
    """
    A runtime that provides a wrapper around legacy engines for compatibility with the new
    stack.
    """

    def __init__(
        self,
        engine: QuantumExecutionEngine,
        results_pipeline: Optional[PassManager] = None,
        startup_engine: bool = False,
    ):
        """
        :param QuantumExecutionEngine engine: The execution engine for a target backend.
        :param results_pipeline: Optionally provided a pipeline for results processing. If not
            provided, a default pipeline is provided.
        :type results_pipeline: PassManager, optional
        :param bool startup_engine: Instruct the engine to connect to the backend on startup?
        """
        if not isinstance(engine, QuantumExecutionEngine):
            raise ValueError(
                f"The engine provided has type {type(engine)}, which is not a "
                "`qat.purr.complier.execution.QuantumExecutionEngine`. Please use the "
                "`SimpleRuntime` for the refactored engines."
            )
        self.engine = engine

        if not results_pipeline:
            results_pipeline = PassManager()
        self.results_pipeline = results_pipeline

        if startup_engine:
            self.engine.startup()

    def execute(
        self,
        package: QatIR,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        **kwargs,
    ):
        """
        Fully execute QatIR against the hardware using a legacy execution engines.

        :param QatIR package: The program as QatIR.
        :param res_mgr: Optionally provide a results manager to save pass information.
        :type res_mgr: ResultManager, optional
        :param met_mgr: Optionally provide a metric manager to save pass information.
        :type met_mgr: MetricsManager, optional
        :returns: Execution results.
        """

        if res_mgr == None:
            res_mgr = ResultManager()
        if met_mgr == None:
            met_mgr = MetricsManager()

        results = self.engine.execute(package.value.instructions)

        # TODO: Remove QatIR with changes to pass manager
        results = QatIR(results)
        self.results_pipeline.run(results, res_mgr, met_mgr, package=package, **kwargs)
        return results.value
