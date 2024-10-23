from time import perf_counter

from tests.benchmarking.qatbm.pipeline.passes import BenchmarkingPass
from tests.benchmarking.qatbm.pipeline.results import (
    BenchmarkingResult,
    ExecutionTimeResult,
)
from tests.benchmarking.qatbm.qatbm import QatCollection


class BenchmarkingWrapper:
    """
    Base class for wrapping a pass. Simply runs the pass with no extra action.
    """

    def create_result(self):
        return BenchmarkingResult()

    def run(self, pass_: BenchmarkingPass, qatcol: QatCollection):
        pass_.run(qatcol)
        return None


class ExecutionTimeWrapper(BenchmarkingWrapper):
    """
    Measures the execution time of a pass.
    """

    def run(self, pass_: BenchmarkingPass, qatcol: QatCollection):
        t = perf_counter()
        pass_.run(qatcol)
        return perf_counter() - t

    def create_result(self):
        return ExecutionTimeResult()
