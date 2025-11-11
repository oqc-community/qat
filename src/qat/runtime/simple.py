# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.runtime import BaseRuntime
from qat.runtime.aggregator import ResultsCollection


class SimpleRuntime(BaseRuntime):
    """A Runtime for the complete execution of packages without sweeps.

    The :class:`SimpleRuntime` is designed to handle the execution of executables; it does
    not take on any classical-quantum hybrid capabilities, such as using the collective
    measurement results to inform later quantum execution (such as variational quantum
    algorithms). This does not mean quantum execution cannot have control flow conditioned
    on classical measurements.  The runtime must be provided with a :class:`NativeEngine`
    that is capable of executing the desired programs.
    """

    def execute(
        self,
        package: Executable,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
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
