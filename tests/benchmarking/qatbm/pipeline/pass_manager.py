from typing import Callable, Union

from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.benchmarking.qatbm.pipeline.passes import (
    BenchmarkingPass,
    CircuitBuilderPass,
    CreateTimelinePass,
    EmitterPass,
    ExecutionPass,
    OptimizationPass,
    QasmBuilderPass,
    ValidationPass,
)
from tests.benchmarking.qatbm.pipeline.results import BenchmarkingResult
from tests.benchmarking.qatbm.pipeline.wrapper import (
    BenchmarkingWrapper,
    ExecutionTimeWrapper,
)
from tests.benchmarking.qatbm.qatbm import QatCollection


class BenchmarkingPassManager:

    def __init__(
        self,
        model: Union[QuantumHardwareModel, QuantumExecutionEngine],
        wrapper: BenchmarkingWrapper = ExecutionTimeWrapper(),
    ):
        """
        Create an execution pipeline to benchmark.
        """
        # Initiate the builder and execution engine
        self.qatcol = QatCollection(model)
        self.wrapper = wrapper

        # Passes and results
        self.passes: list[BenchmarkingPass] = []
        self._results: list[BenchmarkingResult] = []

    def add(self, pass_: BenchmarkingPass):
        """
        Add a pass to the pipeline.
        """
        self.passes.append(pass_)
        return self

    def __or__(self, pass_: BenchmarkingPass):
        return self.add(pass_)

    def run(self, iters: int = 1):
        """
        Run each pass in the pipeline, measuring the execution time for each
        pass and saving it to the results.

        Optionally, run many times and save each result.
        """
        for _ in range(iters):
            self.qatcol.reset()
            result = self.wrapper.create_result()
            for pass_ in self.passes:
                result[pass_] = self.wrapper.run(pass_, self.qatcol)
            self._results.append(result)

    @property
    def results(self):
        return self._results

    @property
    def mean(self):
        """
        Return the average execution time of each pass.
        """
        result = self.wrapper.create_result()
        for res in self.results:
            for key, val in res.result.items():
                result[key] = result.result.get(key, 0) + val

        num_results = len(self.results)
        for key, val in result.result.items():
            result.result[key] = val / num_results

        return result

    @property
    def min(self):
        """
        Returns the minimum execution time of each pass.
        """
        result = self.wrapper.create_result()
        for res in self.results:
            for key, val in res.result.items():
                result[key] = min(result.result.get(key, 0), val)

        return result

    @property
    def max(self):
        """
        Returns the minimum execution time of each pass.
        """
        result = self.wrapper.create_result()
        for res in self.results:
            for key, val in res.result.items():
                result[key] = max(result.result.get(key, 0), val)

        return result


def default_benchmarking(
    model: Union[QuantumHardwareModel, QuantumExecutionEngine],
    input: Union[str, Callable],
    wrapper: BenchmarkingWrapper = ExecutionTimeWrapper(),
):
    """
    Work in progress to isolate "passes".
    Given a string for a QASM file, or a function that takes a hardware model
    and builder as arguments, return a BenchmarkingPassManager with a default
    execution pipeline. Optionally provide a wrapper.

    Build Circuit -> Optimize -> Validate -> Emit -> Create timeline.
    """

    return (
        BenchmarkingPassManager(model, wrapper)
        | (QasmBuilderPass(input) if isinstance(input, str) else CircuitBuilderPass(input))
        | OptimizationPass()
        | ValidationPass()
        | EmitterPass()
        | CreateTimelinePass()
    )


def execution_benchmarking(
    model: Union[QuantumHardwareModel, QuantumExecutionEngine],
    input: Union[str, Callable],
    wrapper: BenchmarkingWrapper = ExecutionTimeWrapper(),
):
    """
    Creates a pipeline to execute a QASM file of QAT circuit. Optionally provide a
    wrapper.

    Build Circuit -> Optimize -> Validate -> Execute
    """

    return (
        BenchmarkingPassManager(model, wrapper)
        | (QasmBuilderPass(input) if isinstance(input, str) else CircuitBuilderPass(input))
        | OptimizationPass()
        | ValidationPass()
        | ExecutionPass()
    )
