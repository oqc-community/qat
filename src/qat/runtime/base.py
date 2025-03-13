# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import abc
import warnings
from typing import Dict, List, Optional

import numpy as np

from qat.core.pass_base import PassManager
from qat.engines import ConnectionMixin, NativeEngine
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.qatconfig import qatconfig
from qat.runtime.connection import ConnectionMode
from qat.runtime.executables import AcquireDataStruct
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
)


class BaseRuntime(abc.ABC):
    """Provides a Base class to build on for runtimes of varying complexities.

    A runtime provides the means to execute quantum programs. It can take on various
    responsibilities, including interfacing the execution engine and post-processing of
    results. Runtimes are designed to fit a specific purpose. For example, the
    :class:`SimpleRuntime` provides the means to execute already compiled programs
    :class:`Executable`. In the future, there will be support for hybrid runtimes that take
    on both compilation and execution responsibilities.
    """

    def __init__(
        self,
        engine: NativeEngine,
        results_pipeline: Optional[PassManager] = None,
        connection_mode: ConnectionMode = ConnectionMode.DEFAULT,
    ):
        """
        :param engine: The execution engine for a target machine.
        :param results_pipeline: Optionally provided a pipeline for results processing. If
            not provided, a default pipeline is provided.
        :param connection_mode: Specifies how the connection is maintained.
        """
        self.engine = engine
        self.connection_mode = connection_mode
        self.connect_engine(ConnectionMode.CONNECT_AT_BEGINNING)

        if not results_pipeline:
            results_pipeline = self.default_pipeline()
        self.results_pipeline = results_pipeline

    @abc.abstractmethod
    def execute(package, *args): ...

    def default_pipeline(self):
        return (
            PassManager()
            | PostProcessingTransform()
            | InlineResultsProcessingTransform()
            | AssignResultsTransform()
        )

    @staticmethod
    def number_of_batches(total_shots: int, shots_per_batch: int):
        """Calculates the number of shot batches to execute.

        When the total number of shots exceeds the capabilities of the target machine, we can
        execute a number of batches with a subset of the shots. This number of shots should
        be calculated during compilation, and included in the executable. This method
        calculates number of batches to execute.

        In the instance that the total number of shots cannot be batched into a whole number
        of the compiled shots, the runtime will execute more shots than required. If results
        are returned from the hardware per shot, then we can simply trim the results down to
        the required amount of shots. However, this cannot be done if the post-processing
        over the shots is done on the hardware. In this case, the program will be executed
        for :code:`ceil(total_shots / shots_per_batch) * shots_per_batch` shots.

        If the compiled number of shots is zero, then it is assumed all shots can be
        achieved in a single batch.

        :param int total_shots: The total number of shots to execute.
        :param int shots_per_batch: The compiled number of shots that can be executed in a
            single batch.
        """

        if not isinstance(total_shots, int) and total_shots < 0:
            raise ValueError("The number of shots must be a non-negative integer.")

        if not isinstance(shots_per_batch, int) and (shots_per_batch < 0):
            raise ValueError(
                "The shots per batch must be a positive integer, or `0` to indicate to use "
                "the `total_shots`."
            )

        if shots_per_batch == 0:
            return 1

        number_of_batches = int(np.ceil(total_shots / shots_per_batch))

        if total_shots % shots_per_batch != 0:
            warnings.warn(
                f"Cannot batch {total_shots} into whole batches of {shots_per_batch}."
                "SCOPE acquisitions will be done with a total of "
                f"{shots_per_batch*number_of_batches}."
            )
        return number_of_batches

    @staticmethod
    def validate_max_shots(shots: int):
        # TODO: determine if this should be a pass.
        if shots > qatconfig.MAX_REPEATS_LIMIT:
            raise ValueError(
                f"Number of shots {shots} exceeds the maximum amount of "
                f"{qatconfig.MAX_REPEATS_LIMIT}."
            )

    def connect_engine(self, flag: ConnectionMode) -> bool | None:
        """Connect the engine according to the connection mode."""
        if not isinstance(self.engine, ConnectionMixin):
            return None

        if flag in self.connection_mode:
            if not self.engine.is_connected:
                self.engine.connect()
        return self.engine.is_connected

    def disconnect_engine(self, flag: ConnectionMode) -> bool | None:
        """Disconnect the engine according to the connection mode."""
        if not isinstance(self.engine, ConnectionMixin):
            return None

        if flag in self.connection_mode:
            if self.engine.is_connected:
                self.engine.disconnect()
        return self.engine.is_connected

    def __del__(self):
        self.disconnect_engine(ConnectionMode.DISCONNECT_AT_END)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.disconnect_engine(ConnectionMode.DISCONNECT_AT_END)


class ResultsAggregator:
    """Aggregates the acquisition results from batching of shots."""

    def __init__(self, acquires: List[AcquireDataStruct]):
        """Begin the aggregation of results for a list of acquisitions.

        :param acquires: List of acquires to be collected.
        :type acquires: List[AcquireDataStruct]
        """
        self._results = {}
        for acquire in acquires:
            match acquire.mode:
                case AcquireMode.RAW:
                    result = np.zeros((0, acquire.length))
                case AcquireMode.INTEGRATOR:
                    result = np.zeros(0)
                case AcquireMode.SCOPE:
                    result = np.zeros(acquire.length)

            self._results[acquire.output_variable] = {
                "mode": acquire.mode,
                "result": result,
                "batches": 0,
            }

    def update(self, new_results: Dict[str, np.ndarray]):
        """Add a batch of results.

        For :attr:`AcquireMode.RAW` and :attr:`AcquireMode.INTEGRATOR`, this means
        to append the new results. For :attr:`AcquireMode.SCOPE`, results are accumulated
        as a sum.
        """
        for output_variable, new_result in new_results.items():
            result = self._results[output_variable]
            result["batches"] += 1
            match result["mode"]:
                case AcquireMode.RAW:
                    result["result"] = np.concatenate(
                        (result["result"], new_result), axis=0
                    )
                case AcquireMode.INTEGRATOR:
                    result["result"] = np.concatenate((result["result"], new_result))
                case AcquireMode.SCOPE:
                    result["result"] += new_result

    def results(self, max_shots: Optional[int] = None):
        """Returns the results normalized in an appropiate way.

        For :attr:`AcquireMode.SCOPE`, results are averaged over batches. For
        :attr:`AcquireMode.RAW` or :attr:`AcquireMode.INTEGRATOR`, a truncated
        amount of :code:`max_shots` shots are returned. If :code:`None`, then returns all
        shots.
        """

        normalized_results = {}
        for output_variable, result in self._results.items():
            match result["mode"]:
                case AcquireMode.RAW:
                    normalized_result = result["result"][0:max_shots, :]
                case AcquireMode.INTEGRATOR:
                    normalized_result = result["result"][0:max_shots]
                case AcquireMode.SCOPE:
                    normalized_result = result["result"] / result["batches"]
            normalized_results[output_variable] = normalized_result
        return normalized_results
