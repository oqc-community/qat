# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.engines import NativeEngine
from qat.executables import Executable
from qat.runtime import BaseRuntime
from qat.runtime.aggregator import QBloxAggregator, ResultsAggregator
from qat.runtime.connection import ConnectionMode


class SimpleRuntime(BaseRuntime):
    """
    The entry point to the execution interface. Programs generated from the backend are wrapped
    in a :class:Executable object which further specifies high level metadata about acquisition
    restrictions such as the expected shape of any loop nest, the post-processing, and results
    formatting. The runtime provides the following services.

    - Execution batching: Some programs require memory specifications beyond that of the control
      hardware. It important to slice and batch-execute large programs. Equally important is
      necessity to aggregate results from different batches.
    - Postprocessing: Performs any required post-processing steps that haven't been carried out
      real-time on the FPGA.
    - Error mitigation: Adjusts results based on error mitigation strategies.

    These services are specified as a pipline of passes, see `results_pipeline` module.
    """

    def __init__(
        self,
        engine: NativeEngine,
        results_pipeline: PassManager | None = None,
        connection_mode: ConnectionMode = ConnectionMode.DEFAULT,
        aggregator: ResultsAggregator | QBloxAggregator | None = None,
    ):
        super().__init__(engine, results_pipeline, connection_mode)
        self.aggregator = aggregator or ResultsAggregator()

    def execute(
        self,
        executable: Executable,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        **kwargs,
    ):
        """Fully execute a package against the hardware with batching of shots and results
        post-processing.

        :param executable: The executable program.
        :param res_mgr: Optionally provide a results manager to save pass information.
        :param met_mgr: Optionally provide a metric manager to save pass information.
        :returns: Execution results.
        """

        if res_mgr is None:
            res_mgr = ResultManager()
        if met_mgr is None:
            met_mgr = MetricsManager()

        self.aggregator.clear()
        with self._hold_connection():
            for program in executable.programs:
                playback = self.engine.execute(program, met_mgr=met_mgr, **kwargs)
                self.aggregator.append(playback, executable.acquires)

        acquisitions = self.aggregator.finalise()

        return self.results_pipeline.run(
            acquisitions, res_mgr, met_mgr, package=executable, **kwargs
        )
