# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from typing import Optional

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.runtime import BaseRuntime
from qat.runtime.aggregator import ResultsCollection


class SimpleRuntime(BaseRuntime):
    """A Runtime for the complete execution of packages without sweeps.

    The :class:`SimpleRuntime` handles the complete execution for simple programs that are
    free from sweeps (with exceptions of sweeps that have been lowered to the hardware).
    This includes batching of shots, executing the program on the target machine, and any software
    post-processing that cannot be achieved in the target machine. The runtime must be provided
    with a :class:`NativeEngine` that is capable of executing the desired programs.
    """

    def execute(
        self,
        package: Executable,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        **kwargs,
    ):
        """Fully execute a package against the hardware with batching of shots and results
        post-processing.

        :param package: The executable program.
        :param res_mgr: Optionally provide a results manager to save pass information.
        :param met_mgr: Optionally provide a metric manager to save pass information.
        :returns: Execution results.
        """

        if res_mgr is None:
            res_mgr = ResultManager()
        if met_mgr is None:
            met_mgr = MetricsManager()

        aggregator = ResultsCollection(package.acquires)
        with self._hold_connection():
            for program in package.programs:
                results = self.engine.execute(program, met_mgr=met_mgr, **kwargs)
                aggregator.append(results)
        acquisitions = aggregator.results

        return self.results_pipeline.run(
            acquisitions, res_mgr, met_mgr, package=package, **kwargs
        )
