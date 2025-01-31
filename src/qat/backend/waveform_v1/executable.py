# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import List, Optional

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from qat.runtime.executables import AcquireDataStruct, ChannelData, Executable


class WaveformV1ChannelData(ChannelData):
    """
    Contains the channel data for a :class:`WaveformV1Executable`.

    Stores the waveforms and acqusitions needed for execution. No control flow is possible.

    :param np.ndarray buffer: The waveform to be sent at each sample.
    :param baseband_frequency: The frequency to be set for the baseband.
    :type baseband_frequency: Optional[float]
    :param acquires: Acquire information needed for readouts.
    :type acquires: List[AcquireDataStruct]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: np.ndarray = np.ndarray([])
    baseband_frequency: Optional[float] = None
    acquires: List[AcquireDataStruct] = []

    @field_serializer("buffer", when_used="json")
    def _serialize_nparray_as_bytes(self, lst):
        """
        Lists of complex numbers can be expensive to serialise: by serializing type
        information and its value as a hex, we can have more performant serialization.
        """
        lst = np.array(lst)
        return {"dtype": lst.dtype.name, "shape": lst.shape, "value": lst.tobytes().hex()}

    @field_validator("buffer", mode="before")
    @classmethod
    def _deserialize_bytes_to_nparray(cls, lst):
        """
        Reverts the hex value and type information into a numpy array.
        """
        if isinstance(lst, dict):
            arr = np.frombuffer(
                bytearray.fromhex(lst["value"]), dtype=np.dtype(lst["dtype"])
            )
            return arr.reshape(lst["shape"])

        # validators are run for any instantiation, so if its not a dict, we don't need any
        # special validation.
        return lst

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


class WaveformV1Executable(Executable):
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
    :type post_processing: Dict[str, List[PostProcessing]]
    :param results_processing: Contains the information for how results should be formatted.
    :type results_processing: Dict[str, InlineResultsProcessing]
    :param assigns: Assigns results to given variables.
    :type assigns: List[Assign]
    :param returns: Which acqusitions/variables should be returned.
    :type returns: List[str]
    """

    channel_data: dict[str, WaveformV1ChannelData]
    repetition_time: float = 100e-6
