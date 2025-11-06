# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from math import prod

import numpy as np

from qat.executables import AcquireData
from qat.purr.compiler.instructions import AcquireMode


class ResultsAggregator:
    """Aggregates acquisition results from multiple programs.

    For now, we assume that the acquisitions always appear in some nested format, with
    iterations over variables being the outer-most levels in the nest. All of the different
    acquisition modes are supported:

    * RAW: The results for each iteration are saved as (number_shots, readout_length, )
    * SCOPE: The results for each iteration are saved as (readout_length, )
    * INTEGRATOR: The results for each iteration are saved as (number_shots, )

    The results for each iteration are the inner-most levels in the nest.

    Results are aggregated as a numpy array, and can be returned as either a numpy array or
    nested lists, with numpy arrays being the default. The nested lists is here if we need
    to replicated legacy behaviour.

    .. warning::

        The ResultsAggregator does not support batching of shots for SCOPE acquire modes.
    """

    def __init__(
        self, mode: AcquireMode, shape: tuple[int, ...], return_as_list: bool = False
    ):
        self._readout_signal = mode == AcquireMode.RAW or mode == AcquireMode.SCOPE
        if self._readout_signal:
            self._readout_length = shape[-1]
        self._shape = shape
        self._results = np.zeros(self._flatten_results_shape(shape), dtype=np.complex128)
        self._index = 0
        self._return_as_list = return_as_list

    def append(self, results: np.ndarray):
        """Appends the results from the batch to the next free index within the
        aggregator."""

        results = results.reshape(self._flatten_results_shape(results.shape))
        if results.shape[0] + self._index > self._results.shape[0]:
            raise ValueError(
                "Attempting to append more results than allocated in the aggregator."
            )

        if self._readout_signal:
            if results.shape[1] != self._readout_length:
                raise ValueError(
                    f"Expected readout length of {self._readout_length}, got "
                    f"{results.shape[1]}."
                )
            self._results[self._index : self._index + results.shape[0], :] = results
        else:
            self._results[self._index : self._index + results.shape[0]] = results
        self._index += results.shape[0]

    @property
    def results(self) -> np.ndarray | list:
        """Returns the aggregated results, reshaped to the expected shape."""
        arr = np.reshape(self._results, self._shape)
        if self._return_as_list:
            return arr.tolist()
        return arr

    def _flatten_results_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """The outer dimensions of the results are treated as linear, allowing us to
        aggregate the results by just adding them to the single dimension. This is reshaped
        at the end to the correct shape."""

        if self._readout_signal:
            return (prod(shape[:-1]), shape[-1])
        return (prod(shape),)


class ResultsCollection:
    """Used to aggregate the results over multiple acquisitions.

    Handles the iteration over acquisitions, abstracting the aggregation details to the
    :class:`ResultsAggregator` class for a given acquisition.
    """

    def __init__(self, acquires: dict[str, AcquireData]):
        self._aggregators = {
            name: ResultsAggregator(acquire.mode, acquire.shape)
            for name, acquire in acquires.items()
        }

    def append(self, results: dict[str, np.ndarray]):
        """Appends the results from a single acquisition to the relevant aggregators."""

        for name, data in results.items():
            self._aggregators[name].append(data)

    @property
    def results(self) -> dict[str, np.ndarray]:
        """Returns the aggregated results for all acquisitions."""

        return {name: aggregator.results for name, aggregator in self._aggregators.items()}
