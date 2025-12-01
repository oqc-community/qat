# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections import defaultdict
from functools import reduce
from itertools import groupby
from math import prod

import numpy as np

from qat.backend.qblox.acquisition import Acquisition
from qat.executables import AcquireData
from qat.purr.compiler.instructions import AcquireMode


class ResultsAggregator:
    """Aggregates acquisition results from multiple acquisitions and programs.

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

    def __init__(self):
        self._modes = {}
        self._shapes = {}
        self._results = {}
        self._indices = {}

    def _flatten_results_shape(
        self, mode: AcquireMode, shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """The outer dimensions of the results are treated as linear, allowing us to
        aggregate the results by just adding them to the single dimension. This is reshaped
        at the end to the correct shape."""

        if mode == AcquireMode.RAW or mode == AcquireMode.SCOPE:
            return (prod(shape[:-1]), shape[-1])
        return (prod(shape),)

    def clear(self):
        self._modes.clear()
        self._shapes.clear()
        self._results.clear()
        self._indices.clear()

    def append(self, playback: dict[str, np.ndarray], acquires: dict[str, AcquireData]):
        for name, data in playback.items():
            mode = self._modes.setdefault(name, acquires[name].mode)
            shape = self._shapes.setdefault(name, acquires[name].shape)
            result = self._results.setdefault(
                name,
                np.zeros(self._flatten_results_shape(mode, shape), dtype=np.complex128),
            )
            index = self._indices.setdefault(name, 0)

            data = data.reshape(self._flatten_results_shape(mode, data.shape))
            if data.shape[0] + index > result.shape[0]:
                raise ValueError(
                    "Attempting to append more results than allocated in the aggregator."
                )

            if mode == AcquireMode.RAW or mode == AcquireMode.SCOPE:
                if data.shape[1] != shape[-1]:
                    raise ValueError(
                        f"Expected readout length of {shape[-1]}, got {data.shape[1]}."
                    )
                result[index : index + data.shape[0], :] = data
            else:
                result[index : index + data.shape[0]] = data
            self._indices[name] = index + data.shape[0]

    def finalise(self, as_list: bool = False):
        results = {}
        for name, result in self._results.items():
            shape = self._shapes[name]
            finalised = np.reshape(result, shape)
            if as_list:
                finalised = finalised.tolist()
            results[name] = finalised
        return results


class QBloxAggregator:
    """
    Combines acquisition objects from multiple acquire instructions in multiple readout targets.
    Notice that :meth:`groupby` preserves (original) relative order, which makes it honour
    the (sequential) lexicographical order of the loop nest:

    playback[target]["acq_0"] contains (potentially) a list of acquisitions collected in the same
    order as the order in which the packages were sent to the FPGA.

    Although acquisition names are enough for unicity in practice, the playback's structure
    distinguishes different (multiple) acquisitions per readout target, thus making it more robust.
    """

    def __init__(self):
        self.playbacks: dict[str, list[Acquisition]] = defaultdict(list)

    def clear(self):
        self.playbacks.clear()

    def append(
        self, playback: dict[str, list[Acquisition]], acquires: dict[str, AcquireData]
    ):
        for pulse_channel_id, acquisitions in playback.items():
            self.playbacks[pulse_channel_id] += acquisitions

    def finalise(self) -> dict[str, dict[str, Acquisition]]:
        playback: dict[str, dict[str, Acquisition]] = {}
        for pulse_channel_id, acquisitions in self.playbacks.items():
            acquisitions.sort(key=lambda acquisition: acquisition.name)
            groups_by_name = groupby(acquisitions, lambda acquisition: acquisition.name)
            playback[pulse_channel_id] = {
                name: reduce(
                    lambda acq1, acq2: acq1 + acq2,
                    acqs,
                    Acquisition(),
                )
                for name, acqs in groups_by_name
            }

        return playback
