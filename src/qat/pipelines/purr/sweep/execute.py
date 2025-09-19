# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from math import prod

import numpy as np
from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.executables import BatchedExecutable
from qat.model.target_data import AbstractTargetData
from qat.pipelines.base import AbstractPipeline
from qat.pipelines.pipeline import ExecutePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.utils.logger import get_default_logger

logger = get_default_logger()


class ExecuteSweepPipeline(AbstractPipeline):
    """A pipeline that supports executing batches of executables.

    Its assumed that the :class:`BatchedExecutable` contains a number of the same programs,
    but parameterized by some variables that are "swept" over. This pipeline will handle
    merging the results of each execution so that the results format is consistent with the
    legacy behaviour.


    .. warning::

        Results are merged into a single dictionary under the assumption that the results
        format in the compiler config is raw. If this is not the case, this might result
        in unexpected behaviour.
    """

    def __init__(self, base_pipeline: AbstractPipeline):
        """
        :param base_pipeline: The underlying pipeline that is used to execute each
            instance of the batch.
        """
        if not base_pipeline.is_subtype_of(ExecutePipeline):
            raise TypeError("The base pipeline must be an ExecutePipeline.")
        self._base_pipeline = base_pipeline

    def is_subtype_of(self, cls):
        return isinstance(self, cls) or self._base_pipeline.is_subtype_of(cls)

    @property
    def name(self) -> str:
        return self._base_pipeline.name

    @property
    def model(self) -> QuantumHardwareModel:
        return self._base_pipeline.model

    @property
    def target_data(self) -> AbstractTargetData:
        return self._base_pipeline.target_data

    def execute(
        self, executable: BatchedExecutable, compiler_config: CompilerConfig | None = None
    ):
        """Execute a batched executable using the base pipeline.

        :param executable: A batched executable containing multiple instances to execute.
        :param compiler_config: Optional compiler configuration to use for this execute
            call. If not provided, the compiler configuration from the pipeline will be
            used.
        """
        if not isinstance(executable, BatchedExecutable):
            raise TypeError("ExecuteSweepPipeline expects a BatchedExecutable.")

        if (
            compiler_config is not None
            and not compiler_config.results_format == QuantumResultsFormat().raw()
        ):
            logger.warning(
                "The compiler_config.results_format is set to a format that is not raw."
                "This is not the expected use case, and may result in unexpected behavior."
            )

        results = []
        for instance in executable.executables:
            result, metrics = self._base_pipeline.execute(
                instance.executable, compiler_config
            )
            results.append(result)

        results = self._combine_results(executable.shape, results)
        return results, metrics

    def _combine_results(self, parameter_sizes: tuple[int], results: list):
        """Combines the results from each execution into a recursive list structure.

        For each key, the combined result will be a nested list that reflects the sweep
        structure. For example, if there are two sweep parameters, A with 3 points and B
        with 2 points, the result will be structured as:
        [[res, res], [res, res], [res, res]].

        :param parameters: The sweep parameters used for the execution.
        :param results: The list of results from each execution.
        :return: A dictionary where each key contains a recursive list of results.
        """
        if len(results) == 0:
            return {}

        if isinstance(results[0], list):
            raise TypeError("Expected a dict, but got a list. Cannot combine results.")

        keys = results[0].keys()
        combined_results = {key: [] for key in keys}

        def build_recursive_list(results, depth=0):
            """Helper function to build a recursive list structure."""
            if depth == len(parameter_sizes):
                return results

            num_points = prod(parameter_sizes[depth + 1 :])
            if num_points == 1:
                return build_recursive_list(results, depth + 1)
            return [
                build_recursive_list(
                    results[i * num_points : (i + 1) * num_points], depth + 1
                )
                for i in range(parameter_sizes[depth])
            ]

        for key in keys:
            key_results = [result[key] for result in results]
            combined_results[key] = np.asarray(build_recursive_list(key_results))

        return combined_results
