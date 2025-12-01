# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
from pydantic import BaseModel, Field

from qat.executables import AbstractProgram
from qat.ir.measure import AcquireMode
from qat.utils.pydantic import ComplexNDArray


class PositionalAcquireData(BaseModel):
    """Contains the position to acquire from the readout signal, and the length of the
    readout."""

    output_variable: str
    position: int
    length: int
    mode: AcquireMode


class WaveformChannelData(BaseModel):
    """
    Contains the channel data for a :class:`WaveformV1Program`.

    Stores the waveforms and acqusitions needed for execution. No control flow is possible.

    :param buffer: The waveform to be sent at each sample.
    :type buffer: list[complex]
    :param baseband_frequency: The frequency to be set for the baseband.
    :type baseband_frequency: float | None
    :param acquires: Acquire information needed for readouts.
    :type acquires: list[AcquireData]
    """

    buffer: ComplexNDArray = Field(default_factory=lambda: ComplexNDArray([]))
    baseband_frequency: float | None = None
    acquires: list[PositionalAcquireData] = Field(default_factory=list)

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


class WaveformProgram(AbstractProgram):
    """Contains the information to execute a task using the waveform buffer channel hardware.

    Contains the buffers and acquire information for each channel, the repetition period for
    each shot and the number of shots to execute.

    :param channel_data: Contains the waveform buffers, acquisitions and baseband
        frequencies for each physical channel.
    :param repetition_time: The time each shot takes to execute.
    :param shots: The number of shots to be executed as part of the program.
    """

    channel_data: dict[str, WaveformChannelData]
    repetition_time: float
    shots: int

    @property
    def acquires(self) -> list[PositionalAcquireData]:
        acquires: list[PositionalAcquireData] = []
        for channel in self.channel_data.values():
            acquires.extend(channel.acquires)
        return acquires

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        acquire_shapes = {}
        for data in self.channel_data.values():
            for acquire in data.acquires:
                if acquire.mode == AcquireMode.SCOPE:
                    acquire_shapes[acquire.output_variable] = (acquire.length,)
                elif acquire.mode == AcquireMode.RAW:
                    acquire_shapes[acquire.output_variable] = (
                        self.shots,
                        acquire.length,
                    )
                else:
                    acquire_shapes[acquire.output_variable] = (self.shots,)
        return acquire_shapes
