# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import Dict, List, Set, Union

from compiler_config.config import InlineResultsProcessing
from pydantic import BaseModel
from pydantic_core import from_json

from qat.ir.instructions import Assign
from qat.ir.measure import PostProcessing
from qat.purr.compiler.instructions import AcquireMode


class AcquireDataStruct(BaseModel):
    """
    Contains all the information needed for to record a measurement / readout.

    Depending on the target machine, not all information might be needed.

    :param int length: A readout is performed for some given time which translates to a
        number of discrete samples.
    :param int position: The sample which a readout starts.
    :param AcquireMode mode: The acqusition mode used by the hardware to carry out the readout.
    :param str output_variable: The name of the variable to save the result to.
    """

    length: int
    position: int
    mode: AcquireMode
    output_variable: str


class ChannelData(BaseModel):
    """
    Contains the instructions required by a particular channel in the control stack to execute
    a program.

    For a given target, this should be expanded on to include any instructions / data
    required for execution.

    :param acquires: Specifies the acquire (or a list of acquires) for a channel.
    :type acqurie: Union[AcquireDataStruct, List[AcquireDataStruct]
    """

    acquires: Union[AcquireDataStruct, List[AcquireDataStruct]] = []


class Executable(BaseModel):
    """
    :class:`Executables` are instruction packages that will be sent to a target (either
    physical control hardware or a simulator).

    This class serves as a base model that can be inherited to define a complete instruction
    package tailored to a particular target.

    :param channel_data: Stores the data required by the control hardware for each pulse
        channel.
    :type channel_data: dict[str, ChannelData]
    :param int shots: The number of times the program is executed.
    :param int compiled_shots: When the required number of shots exceeds the allowed amount by
        the target machine, shots can be batched into groups. This states how many shots to do
        in each batch.
    :param post_processing: Contains the post-processing information for each acquisition.
    :type post_processing: dict[str, List[PostProcessing]]
    :param results_processing: Contains the information for how results should be formatted.
    :type results_processing: dict[str, InlineResultsProcessing]
    :param assigns: Assigns results to given variables.
    :type assigns: list[Assign]
    :param returns: Which acqusitions/variables should be returned.
    :type returns: list[str]
    """

    # TODO: The way we separate post processing, results processing and assigns implies a
    # fixed flow: post processing -> results processing -> assign -> return. It also implies
    # that it doesn't matter what order we process the variables. While this might be true
    # today, this might change in the future.

    shots: int = 1000
    compiled_shots: int = 0
    channel_data: Dict[str, ChannelData] = dict()
    post_processing: Dict[str, List[PostProcessing]] = dict()
    results_processing: Dict[str, InlineResultsProcessing] = dict()
    assigns: List[Assign] = []
    returns: Set[str] = set()

    def serialize(self, indent: int = 4) -> str:
        """
        Serializes the executable as a JSON blob.
        """
        return self.model_dump_json(indent=indent, exclude_none=True)

    @classmethod
    def deserialize(cls, blob: str):
        """
        Instantiates a executable from a JSON blob.
        """
        return cls(**from_json(blob))

    @property
    def acquires(self) -> List[AcquireDataStruct]:
        """
        Retrieves all assigns from each channel.
        """
        acquires = []
        for channel in self.channel_data.values():
            # Redundancy measure for executables that only store and allow a single acquire
            # per chanel
            if isinstance(channel.acquires, List):
                acquires.extend(channel.acquires)
            else:
                acquires.append(channel.acquires)
        return acquires
