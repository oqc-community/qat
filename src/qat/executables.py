# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import abstractmethod
from typing import Generic, TypeVar

from compiler_config.config import InlineResultsProcessing
from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from pydantic_core import from_json

from qat.ir.instructions import Assign
from qat.ir.measure import PostProcessing
from qat.purr.compiler.instructions import AcquireMode
from qat.utils.pydantic import RehydratableModel


class BaseExecutable(RehydratableModel):
    """Base object for executables."""

    calibration_id: str = ""

    def serialize(self, indent: int = 4) -> str:
        """Serializes the executable as a JSON blob."""
        return self.model_dump_json(indent=indent, exclude_none=True)

    @classmethod
    def deserialize(cls, blob: str):
        """Instantiates a executable from a JSON blob."""
        return cls._rehydrate_object(from_json(blob))


class AcquireData(BaseModel):
    """Contains acquisition information for a measurement / readout that is needed at
    runtime.

    :param NonNegativeInt length: A readout is performed for some given time which
        translates to a number of discrete samples.
    :param NonNegativeInt position: The sample which a readout starts.
    :param AcquireMode mode: The acqusition mode used by the hardware to carry out the
        readout.
    :param str output_variable: The name of the variable to save the result to.
    """

    length: NonNegativeInt
    position: NonNegativeInt
    mode: AcquireMode
    output_variable: str


class Executable(BaseExecutable):
    """
    :class:`Executables` are instruction packages that will be sent to a target (either
    physical control hardware or a simulator).

    This class serves as a base model that can be inherited to define a complete instruction
    package tailored to a particular target.

    :param channel_data: Stores the data required by the control hardware for each pulse
        channel.
    :type channel_data: dict[str, ChannelData]
    :param PositiveInt shots: The number of times the program is executed.
    :param PositiveInt compiled_shots: When the required number of shots exceeds the
        allowed amount by the target machine, shots can be batched into groups. This states
        how many shots to do in each batch.
    :param post_processing: Contains the post-processing information for each acquisition.
    :type post_processing: dict[str, List[PostProcessing]]
    :param results_processing: Contains the information for how results should be formatted.
    :type results_processing: dict[str, InlineResultsProcessing]
    :param assigns: Assigns results to given variables.
    :type assigns: list[Assign]
    :param returns: Which acqusitions/variables should be returned.
    :type returns: list[str]
    :param calibration_id: The (unique) calibration ID of the underlying hardware model.
    :type calibration_id: str
    """

    shots: PositiveInt = 1000
    compiled_shots: NonNegativeInt = 0
    post_processing: dict[str, list[PostProcessing]] = dict()
    results_processing: dict[str, InlineResultsProcessing] = dict()
    assigns: list[Assign] = []
    returns: set[str] = set()

    @property
    @abstractmethod
    def acquires(self) -> list[AcquireData]:
        """Retrieves all acquires for the program.

        This abstract property should be specified for an executable that is coupled to a
        specific backend and target device. It is expected that this returns a list of all
        acquisitions for the program, which will be required by the Runtime."""
        ...


class ChannelData(BaseModel):
    """Base class for "channel data". This can be used for exectuables which target devices
    that seperate instructions over multiple "channels".

    It is expected to contain any target-specific information, along with a property that
    retrieves all acquistion data for that channel.
    """

    acquires: list[AcquireData] | AcquireData = []


class ChannelExecutable(Executable):
    """A base Executable object used for compiled programs that target hardware with many
    "channels", e.g., live hardware. Defines the acquire property to fetch acquisitions from
    the channel data."""

    channel_data: dict[str, ChannelData] = {}

    @property
    def acquires(self) -> list[AcquireData]:
        """Retrieves all acquires from each channel."""

        acquires = []
        for channel in self.channel_data.values():
            if isinstance(channel.acquires, list):
                acquires.extend(channel.acquires)
            else:
                acquires.append(channel.acquires)
        return acquires


T = TypeVar("T", bound=BaseExecutable)


class ParameterizedExecutable(BaseModel, Generic[T]):
    """Contains an executable and the metadata surrounding its paramerization when used in
    an iterator.

    :param params: The parameters used to instantiate the executable.
    :param executable: The executable instantiated with the given parameters.
    """

    def __init__(self, **data):
        executable = data.pop("executable")
        if isinstance(executable, dict):
            executable = BaseExecutable._rehydrate_object(executable)
        super().__init__(executable=executable, **data)

    params: dict[str, complex | float] = {}
    executable: T


class BatchedExecutable(BaseExecutable):
    """Contains a batch of executables to be run sequentially.

    :param executables: The list of executables to be run in sequence.
    """

    executables: list[ParameterizedExecutable] = []

    @model_validator(mode="after")
    def _check_executable_types(self):
        types_ = {type(pkg.executable) for pkg in self.executables}
        if len(types_) > 1:
            raise ValueError(
                "All executables in a BatchedExecutable must be of the same type."
            )
        return self
