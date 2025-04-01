# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


import numpy as np

from qat.runtime.executables import AcquireData, ChannelData, ChannelExecutable
from qat.utils.pydantic import ComplexNDArray


class WaveformV1ChannelData(ChannelData):
    """
    Contains the channel data for a :class:`WaveformV1Executable`.

    Stores the waveforms and acqusitions needed for execution. No control flow is possible.

    :param buffer: The waveform to be sent at each sample.
    :type buffer: list[complex]
    :param baseband_frequency: The frequency to be set for the baseband.
    :type baseband_frequency: float | None
    :param acquires: Acquire information needed for readouts.
    :type acquires: list[AcquireData]
    """

    buffer: ComplexNDArray = np.ndarray([])
    baseband_frequency: float | None = None
    acquires: list[AcquireData] = []

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.acquires != other.acquires:
            return False
        if self.baseband_frequency != other.baseband_frequency:
            return False
        if not np.all(self.buffer == other.buffer):
            return False
        return True


class WaveformV1Executable(ChannelExecutable):
    """
    A :class:`WaveformV1Executable` is an executable used in early iterations of QAT.

    This type of executable is composed of explicit waveforms that are compiled for each
    pulse channel, and summed to provide the waveform "buffer" for each physical channel.
    This early iteration can only instruct waveforms to be executed at a particular time;
    control flow is not possible. However, simple post-processing of results after execution
    is provided.

    :param channel_data: Stores the data required by the control hardware for each pulse
        channel.
    :type channel_data: dict[str, WaveformV1ChannelData]
    :param int shots: The number of times the program is executed.
    :param float repetition_time: The amount of time to wait between shots for the QPU to
        reset.
    :param post_processing: Contains the post-processing information for each acquisition.
    :type post_processing: dict[str, list[PostProcessing]]
    :param results_processing: Contains the information for how results should be formatted.
    :type results_processing: dict[str, InlineResultsProcessing]
    :param assigns: Assigns results to given variables.
    :type assigns: list[Assign]
    :param returns: Which acqusitions/variables should be returned.
    :type returns: list[str]
    """

    channel_data: dict[str, WaveformV1ChannelData]
    repetition_time: float = 100e-6
