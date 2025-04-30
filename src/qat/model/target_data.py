# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from pathlib import Path

import piny
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt

from qat.utils.piny import VeryStrictMatcher
from qat.utils.pydantic import NoExtraFieldsFrozenModel


class DeviceDescription(NoExtraFieldsFrozenModel):
    """
    Device-related target description.

    :param clock_cycle_f: The clock cycle (in frequency units).
    :param instruction_memory_size: The max. allowed number of instructions.
    :param weveform_memory_size: The max. memory that can be used for waveforms.
    :param pulse_duration_min: The minimal pulse duration for all pulse channels.
    :param pulse_duration_max: The maximal pulse duration for all pulse channels.
    :pulse_channel_lo_freq_min: The minimal LO frequency for a pulse channel.
    :pulse_channel_lo_freq_max: The maximal LO frequency for a pulse channel.
    :pulse_channel_if_freq_min: The minimal intermediate frequency for a pulse channel.
    :pulse_channel_if_freq_max: The maximal intermediate frequency for a pulse channel.
    """

    clock_cycle_f: PositiveInt
    instruction_memory_size: PositiveInt
    waveform_memory_size: PositiveInt
    pulse_duration_min: PositiveFloat
    pulse_duration_max: PositiveFloat
    pulse_channel_lo_freq_min: PositiveInt | PositiveFloat
    pulse_channel_lo_freq_max: PositiveInt | PositiveFloat
    pulse_channel_if_freq_min: PositiveInt | PositiveFloat
    pulse_channel_if_freq_max: PositiveInt | PositiveFloat

    @property
    def clock_cycle_t(self):
        return 1 / self.clock_cycle_f


class QubitDescription(DeviceDescription):
    """
    Qubit-related target description.

    :param passive_reset_time: The amount of time after each shot where the qubit is idle.
    """

    passive_reset_time: NonNegativeFloat


class ResonatorDescription(DeviceDescription):
    """
    Resonator-related target description.
    """

    ...


class TargetData(NoExtraFieldsFrozenModel):
    """
    Data related to a general target machine.

    :param max_shots: The maximum amount of shots possible on this target.
    :param default_shots: The default amount of shots on this target if none specified through the instructions.
    :param QUBIT_DATA: Qubit-related target description.
    :param RESONATOR_DATA: Resonator-related target description.
    """

    max_shots: PositiveInt
    default_shots: PositiveInt = 1

    QUBIT_DATA: QubitDescription
    RESONATOR_DATA: ResonatorDescription

    @classmethod
    def from_yaml(cls, path: str | Path):
        blob = piny.YamlLoader(path=str(path), matcher=VeryStrictMatcher).load()
        return cls(**blob)
