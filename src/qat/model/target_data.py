# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random
from functools import cached_property
from pathlib import Path
from warnings import warn

import piny
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from qat.utils.piny import VeryStrictMatcher
from qat.utils.pydantic import NoExtraFieldsFrozenModel
from qat.utils.uuid import SeedType


class DeviceDescription(NoExtraFieldsFrozenModel):
    """Device-related target description.

    :param sample_time: The rate at which the pulse is sampled (in s).
    :param samples_per_clock_cycle: The number of samples per clock cycle.
    :param instruction_memory_size: The max. allowed number of instructions.
    :param waveform_memory_size: The max. memory that can be used for waveforms, in clock
        cycles.
    :param pulse_duration_min: The minimal pulse duration for all pulse channels.
    :param pulse_duration_max: The maximal pulse duration for all pulse channels.
    :param pulse_channel_lo_freq_min: The minimal LO frequency for a pulse channel.
    :param pulse_channel_lo_freq_max: The maximal LO frequency for a pulse channel.
    :param pulse_channel_if_freq_min: The minimal intermediate frequency for a pulse
        channel.
    :param pulse_channel_if_freq_max: The maximal intermediate frequency for a pulse
        channel.
    """

    sample_time: PositiveFloat = 1e-09
    samples_per_clock_cycle: PositiveInt = 1
    instruction_memory_size: PositiveInt = 50_000
    waveform_memory_size: PositiveInt = 1_500
    pulse_duration_min: PositiveFloat = 64e-09
    pulse_duration_max: PositiveFloat = 1e-03
    pulse_channel_lo_freq_min: NonNegativeInt = 1_000_000
    pulse_channel_lo_freq_max: NonNegativeInt = 10_000_000_000
    pulse_channel_if_freq_min: NonNegativeInt = 0
    pulse_channel_if_freq_max: NonNegativeInt = 10_000_000_000

    @model_validator(mode="after")
    def validate_durations(self):
        if self.pulse_duration_min > self.pulse_duration_max:
            raise ValueError(
                "Min. pulse duration cannot be larger than max. pulse duration."
            )
        return self

    @classmethod
    def random(cls, seed: SeedType | None = None):
        """Build a randomized device description for testing and demos.

        :param seed: Optional seed used to make the generated values reproducible.
        :return: A :class:`DeviceDescription` with randomized, valid hardware limits.
        """
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
    """Qubit-related target description.

    :param passive_reset_time: The amount of time after each shot where the qubit is idle.
    """

    passive_reset_time: NonNegativeFloat = 1e-03

    @classmethod
    def random(cls, seed: SeedType | None = None):
        """Build a randomized qubit description for testing and demos.

        This method starts from :meth:`DeviceDescription.random` and adds a
        randomized ``passive_reset_time``.

        :param seed: Optional seed used to make the generated values reproducible.
        :return: A :class:`QubitDescription` instance with randomized valid fields.
        """
        device_descr = DeviceDescription.random(seed).model_dump()
        device_descr.update(
            {"passive_reset_time": random.Random(seed).uniform(1e-06, 1e-04)}
        )
        return cls(**device_descr)


class ResonatorDescription(DeviceDescription):
    """Resonator-related target description."""

    ...


class AbstractTargetData(NoExtraFieldsFrozenModel):
    """Data related to a general target machine.

    :param max_acquisitions: The maximum amount of acquisitions possible on this target.
    :param max_shots: (deprecated) Use `max_acquisitions` instead.
    :param default_shots: The default amount of shots on this target if none specified through the instructions.
    """

    max_acquisitions: PositiveInt = Field(
        10_000,
        alias="max_shots",
        description=(
            "Max acquisitions per memory channel. "
            "(max_shots is deprecated, use this instead.)"
        ),
    )
    default_shots: PositiveInt = 1

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    @model_validator(mode="before")
    def warn_on_max_shots(cls, values):
        if "max_shots" in values:
            warn(
                "`max_shots` is deprecated; use `max_acquisitions` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return values

    @property
    def max_shots(self) -> PositiveInt:
        warn(
            "`max_shots` is deprecated; use `max_acquisitions` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.max_acquisitions

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
    """Data related to a general target machine.

    :param max_acquisitions: The maximum amount of acquisitions per channel possible on this
        target.
    :param default_shots: The default amount of shots on this target if none specified
        through the instructions.
    :param QUBIT_DATA: Qubit-related target description.
    :param RESONATOR_DATA: Resonator-related target description.
    """

    default_shots: PositiveInt = 1_000
    QUBIT_DATA: QubitDescription = QubitDescription()
    RESONATOR_DATA: ResonatorDescription = ResonatorDescription()

    @cached_property
    def clock_cycle(self):
        return max(self.QUBIT_DATA.clock_cycle, self.RESONATOR_DATA.clock_cycle)

    @cached_property
    def instruction_memory_size(self):
        """
        .. deprecated:: 3.3.0
            This property will be removed in version 4.0.0.
            Use :attr:`TargetData.QUBIT_DATA` (then
            ``TargetData().QUBIT_DATA.instruction_memory_size``) or
            :attr:`TargetData.RESONATOR_DATA` (then
            ``TargetData().RESONATOR_DATA.instruction_memory_size``) instead.
        """
        warn(
            f"`{type(self).__name__}.instruction_memory_size` is deprecated; use "
            "`QUBIT_DATA.instruction_memory_size` or "
            "`RESONATOR_DATA.instruction_memory_size` instead. "
            "This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=4,
        )
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

    @classmethod
    def random(cls, seed: SeedType | None = None) -> "TargetData":
        """Build randomized target data for testing and synthetic examples.

        When ``seed`` is ``None``, a seed is generated and reused for both qubit
        and resonator descriptions so they remain clock-cycle compatible.

        :param seed: Optional seed used to make the generated values reproducible.
        :return: A randomized :class:`TargetData` instance.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        return cls(
            max_acquisitions=random.Random(seed).randint(200, 1000),
            default_shots=random.Random(seed).randint(1, 100),
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )

    @classmethod
    def create_with(
        cls,
        passive_reset_time: NonNegativeFloat | None = None,
        **kwargs,
    ) -> "TargetData":
        """Returns a default TargetData instance, allowing configuration over user-level
        parameters.

        .. deprecated:: 3.3.0
            This method will be removed in version 4.0.0.
            Use :func:`TargetData` with appropriate keyword arguments instead.

        :param max_shots: The maximum amount of shots possible on this target.
        :param default_shots: The default amount of shots on this target if none specified
            through the instructions or compiler_config.
        :param passive_reset_time: The amount of time to wait after each shot to allow the
            qubits to passively reset.
        """
        warn(
            f"`{cls.__name__}.create_with()` is deprecated; use `{cls.__name__}()` with "
            "appropriate keyword arguments instead. "
            "This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if passive_reset_time is not None:
            kwargs["QUBIT_DATA"] = kwargs.get("QUBIT_DATA", {}) | {
                "passive_reset_time": passive_reset_time
            }
        return cls(**kwargs)

    @classmethod
    def default(cls) -> "TargetData":
        """Returns a default TargetData instance.

        .. deprecated:: 3.3.0
            This method will be removed in version 4.0.0.
            Use :func:`TargetData` instead.
        """
        warn(
            f"`{cls.__name__}.default()` is deprecated; use `{cls.__name__}()` instead. "
            "This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls()


def DefaultTargetData() -> TargetData:
    """Returns a default TargetData instance.

    .. deprecated:: 3.3.0
        This will be removed in version 4.0.0.
        Use :func:`TargetData` instead.
    """
    warn(
        "`DefaultTargetData()` is deprecated; use `TargetData()` instead. "
        "This will be removed in v4.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TargetData()


def CustomTargetData(
    max_shots: int = 10_000,
    default_shots: int = 1_000,
    passive_reset_time: float = 1e-03,
) -> TargetData:
    """Returns a default TargetData instance, allowing configuration over user-level
    parameters.

    .. deprecated:: 3.3.0
        This will be removed in version 4.0.0.
        Use :func:`TargetData` with appropriate keyword arguments instead.

    :param max_shots: The maximum amount of shots possible on this target.
    :param default_shots: The default amount of shots on this target if none specified
        through the instructions or compiler_config.
    :param passive_reset_time: The amount of time to wait after each shot to allow the
        qubits to passively reset.
    """
    warn(
        "`CustomTargetData()` is deprecated; use `TargetData()` with appropriate keyword "
        "arguments instead. "
        "This will be removed in v4.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TargetData(
        max_shots=max_shots,
        default_shots=default_shots,
        QUBIT_DATA=QubitDescription(passive_reset_time=passive_reset_time),
    )
