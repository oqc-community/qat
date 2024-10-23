from json import dump
from typing import Any

from tests.benchmarking.qatbm.pipeline.passes import BenchmarkingPass


class BenchmarkingResult:

    def __init__(self):
        self.result: dict[BenchmarkingPass, Any] = {}

    def __getitem__(self, idx):
        return self.result[idx]

    def __setitem__(self, idx, val):
        self.result[idx] = val

    def __repr__(self):
        return "\n".join([f"{key}: {val}" for key, val in self.result.items()])

    def items(self):
        return self.result.items()


class ExecutionTimeResult(BenchmarkingResult):

    def __init__(self):
        """
        Store the execution times for a benchmarking run.
        """

        self.result: dict[BenchmarkingPass, float] = {}

    def dump(self, f):
        """
        Save the execution times as a JSON blob.
        """

        results = {str(key): val for key, val in self.items()}
        with open(f, "w") as f:
            dump(results, f, indent=4)
