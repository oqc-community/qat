# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random
from functools import cached_property
from pathlib import Path

import piny
from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from qat.utils.piny import VeryStrictMatcher
from qat.utils.pydantic import NoExtraFieldsFrozenModel


class DeviceDescription(NoExtraFieldsFrozenModel):
    """
    Device-related target description.

    :param sample_time: The rate at which the pulse is sampled (in s).
    :param samples_per_clock_cycle: The number of samples per clock cycle.
    :param instruction_memory_size: The max. allowed number of instructions.
    :param waveform_memory_size: The max. memory that can be used for waveforms, in clock cycles.
    :param pulse_duration_min: The minimal pulse duration for all pulse channels.
    :param pulse_duration_max: The maximal pulse duration for all pulse channels.
    :param pulse_channel_lo_freq_min: The minimal LO frequency for a pulse channel.
    :param pulse_channel_lo_freq_max: The maximal LO frequency for a pulse channel.
    :param pulse_channel_if_freq_min: The minimal intermediate frequency for a pulse channel.
    :param pulse_channel_if_freq_max: The maximal intermediate frequency for a pulse channel.
    """

    sample_time: PositiveFloat
    samples_per_clock_cycle: PositiveInt
    instruction_memory_size: PositiveInt
    waveform_memory_size: PositiveInt
    pulse_duration_min: PositiveFloat
    pulse_duration_max: PositiveFloat
    pulse_channel_lo_freq_min: NonNegativeInt
    pulse_channel_lo_freq_max: NonNegativeInt
    pulse_channel_if_freq_min: NonNegativeInt
    pulse_channel_if_freq_max: NonNegativeInt

    @model_validator(mode="after")
    def validate_durations(self):
        if self.pulse_duration_min > self.pulse_duration_max:
            raise ValueError(
                "Min. pulse duration cannot be larger than max. pulse duration."
            )
        return self

    @classmethod
    def random(cls, seed=42):
        return cls(
            **{
                "sample_time": random.Random(seed).choice([1e-09, 2e-09]),
                "samples_per_clock_cycle": random.Random(seed).randint(5, 10),
                "instruction_memory_size": random.Random(seed).randint(200, 8000),
                "waveform_memory_size": random.Random(seed).randint(1000, 2000),
                "pulse_duration_min": random.Random(seed).uniform(1e-09, 1e-07),
                "pulse_duration_max": random.Random(seed).uniform(1e-06, 1e-03),
                "pulse_channel_lo_freq_min": random.Random(seed).randint(1_000, 10_000),
                "pulse_channel_lo_freq_max": random.Random(seed).randint(
                    1_000_000, 100_000_000
                ),
                "pulse_channel_if_freq_min": random.Random(seed).randint(1_000, 10_000),
                "pulse_channel_if_freq_max": random.Random(seed).randint(
                    1_000_000, 100_000_000
                ),
            }
        )

    @cached_property
    def clock_cycle(self):
        return self.samples_per_clock_cycle * self.sample_time


class QubitDescription(DeviceDescription):
    """
    Qubit-related target description.

    :param passive_reset_time: The amount of time after each shot where the qubit is idle.
    """

    passive_reset_time: NonNegativeFloat

    @classmethod
    def random(cls, seed=42):
        device_descr = DeviceDescription.random(seed).model_dump()
        device_descr.update(
            {"passive_reset_time": random.Random(seed).uniform(1e-06, 1e-04)}
        )
        return cls(**device_descr)


class ResonatorDescription(DeviceDescription):
    """
    Resonator-related target description.
    """

    ...


class AbstractTargetData(NoExtraFieldsFrozenModel):
    """
    Data related to a general target machine.

    :param max_shots: The maximum amount of shots possible on this target.
    :param default_shots: The default amount of shots on this target if none specified through the instructions.
    """

    max_shots: PositiveInt
    default_shots: PositiveInt = 1

    def __getattribute__(self, name: str):
        if name in ["QUBIT_DATA", "RESONATOR_DATA"]:
            try:
                return super().__getattribute__(name)
            except AttributeError:
                raise AttributeError(
                    f"Tried to get '{name}' from {self.__class__.__name__}, which does not exist. Please use a child class of `AbstractTargetData` that has '{name}'."
                )
        return super().__getattribute__(name)

    @classmethod
    def from_yaml(cls, path: str | Path):
        blob = piny.YamlLoader(path=str(path), matcher=VeryStrictMatcher).load()
        return cls(**blob)


class TargetData(AbstractTargetData):
    """
    Data related to a general target machine.

    :param max_shots: The maximum amount of shots possible on this target.
    :param default_shots: The default amount of shots on this target if none specified through the instructions.
    :param QUBIT_DATA: Qubit-related target description.
    :param RESONATOR_DATA: Resonator-related target description.
    """

    QUBIT_DATA: QubitDescription
    RESONATOR_DATA: ResonatorDescription

    @cached_property
    def clock_cycle(self):
        return max(self.QUBIT_DATA.clock_cycle, self.RESONATOR_DATA.clock_cycle)

    @cached_property
    def instruction_memory_size(self):
        return max(
            self.QUBIT_DATA.instruction_memory_size,
            self.RESONATOR_DATA.instruction_memory_size,
        )

    @model_validator(mode="after")
    def validate_clock_cycles(self):
        if self.QUBIT_DATA.clock_cycle != self.RESONATOR_DATA.clock_cycle:
            raise ValueError(
                "Different clock cycles for qubit and resonator are currently not supported."
            )
        return self

    @model_validator(mode="after")
    def validate_instruction_memory_size(self):
        if (
            self.QUBIT_DATA.instruction_memory_size
            != self.RESONATOR_DATA.instruction_memory_size
        ):
            raise ValueError(
                "Different instruction memory sizes for qubit and resonator are currently not supported."
            )
        return self

    @classmethod
    def random(cls, seed: int = 42) -> "TargetData":
        return cls(
            max_shots=random.Random(seed).randint(200, 1000),
            default_shots=random.Random(seed).randint(1, 100),
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )

    @classmethod
    def create_with(
        cls,
        max_shots: int = 10_000,
        default_shots: int = 1_000,
        passive_reset_time: float = 1e-03,
    ) -> "TargetData":
        """Returns a default TargetData instance, allowing configuration over user-level
        parameters.

        :param max_shots: The maximum amount of shots possible on this target.
        :param default_shots: The default amount of shots on this target if none specified
            through the instructions or compiler_config.
        :param passive_reset_time: The amount of time to wait after each shot to allow the
            qubits to passively reset.
        """
        return TargetData(
            max_shots=max_shots,
            default_shots=default_shots,
            QUBIT_DATA=QubitDescription(
                passive_reset_time=passive_reset_time,
                sample_time=1e-09,
                samples_per_clock_cycle=1,
                instruction_memory_size=50_000,
                waveform_memory_size=1_500,
                pulse_duration_min=64e-09,
                pulse_duration_max=1e-03,
                pulse_channel_lo_freq_min=1e06,
                pulse_channel_lo_freq_max=1e10,
                pulse_channel_if_freq_min=0,
                pulse_channel_if_freq_max=1e10,
            ),
            RESONATOR_DATA=ResonatorDescription(
                sample_time=1e-09,
                samples_per_clock_cycle=1,
                instruction_memory_size=50_000,
                waveform_memory_size=1_500,
                pulse_duration_min=64e-09,
                pulse_duration_max=1e-03,
                pulse_channel_lo_freq_min=1e06,
                pulse_channel_lo_freq_max=1e10,
                pulse_channel_if_freq_min=0,
                pulse_channel_if_freq_max=1e10,
            ),
        )

    @classmethod
    def default(cls) -> "TargetData":
        """Returns a default TargetData instance."""
        return cls.create_with()


def DefaultTargetData() -> TargetData:
    """Returns a default TargetData instance."""
    return TargetData.default()


def CustomTargetData(
    max_shots: int = 10_000, default_shots: int = 1_000, passive_reset_time: float = 1e-03
) -> TargetData:
    """Returns a default TargetData instance, allowing configuration over user-level
    parameters.

    :param max_shots: The maximum amount of shots possible on this target.
    :param default_shots: The default amount of shots on this target if none specified
        through the instructions or compiler_config.
    :param passive_reset_time: The amount of time to wait after each shot to allow the
        qubits to passively reset.
    """
    return TargetData.create_with(
        max_shots=max_shots,
        default_shots=default_shots,
        passive_reset_time=passive_reset_time,
    )
