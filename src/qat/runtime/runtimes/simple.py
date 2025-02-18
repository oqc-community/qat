# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import warnings
from typing import Optional

import numpy as np

from qat.passes.metrics_base import MetricsManager
from qat.passes.result_base import ResultManager
from qat.runtime import BaseRuntime, ResultsAggregator
from qat.runtime.executables import Executable


class SimpleRuntime(BaseRuntime):
    """A Runtime for the complete execution of packages without sweeps.

    The :class:`SimpleRuntime` handles the complete execution for simple programs that are
    free from sweeps (with exceptions of sweeps that have been lowered to the hardware).
    This includes batching of shots, executing the program on the backend, and any software
    post-processing that cannot be achieved in the backend. The runtime must be provided
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

        :param Executable package: The executable program.
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

        self.validate_max_shots(package.shots)
        if package.shots == 0:
            warnings.warn("Tried to execute a package with zero shots.")
            return {acquire.output_variable: np.array([]) for acquire in package.acquires}

        number_of_batches = self.number_of_batches(package.shots, package.compiled_shots)
        aggregator = ResultsAggregator(package.acquires)
        for _ in range(number_of_batches):
            batch_results = self.engine.execute(package)
            aggregator.update(batch_results)
        acquisitions = aggregator.results(package.shots)
        return self.results_pipeline.run(
            acquisitions, res_mgr, met_mgr, package=package, **kwargs
        )
