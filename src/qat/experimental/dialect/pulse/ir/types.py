# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from typing import TypeVar

from xdsl.ir import TypeAttribute
from xdsl.irdl import ParametrizedAttribute, irdl_attr_definition


@irdl_attr_definition
class FrequencyType(ParametrizedAttribute, TypeAttribute):
    """A type representing a frequency value, used for expressing frequencies of pulse
    channels.

    The value is expected to be a floating-point number representing the frequency in Hz.
    """

    name = "pulse.frequency"


@irdl_attr_definition
class AmplitudeType(ParametrizedAttribute, TypeAttribute):
    """A type representing an amplitude value, used for expressing the amplitude of pulse
    channels.

    The value is expected to be a floating-point number representing the amplitude in
    arbitrary units.
    """

    name = "pulse.amplitude"


@irdl_attr_definition
class PhaseType(ParametrizedAttribute, TypeAttribute):
    """A type representing a phase value, typically used for phase manipulations.

    The value is expected to be a floating-point number representing the angle in radians.
    """

    name = "pulse.phase"


@irdl_attr_definition
class TimeType(ParametrizedAttribute, TypeAttribute):
    """Represents a time value, used for expressing durations of operations on frames."""

    name = "pulse.time"


@irdl_attr_definition
class FrameType(ParametrizedAttribute, TypeAttribute):
    """Represents a reference frame for a quantum system, encoding a frequency, and tracks
    phase and time evolution relative to that frequency.

    Used with the intent of manipulating a quantum component.
    """

    name = "pulse.frame"


@irdl_attr_definition
class WaveformType(ParametrizedAttribute, TypeAttribute):
    """Represents a waveform type."""

    name = "pulse.waveform"


PULSE_VAR_TYPE = TypeVar(
    "PULSE_VAR_TYPE",
    bound=FrequencyType | PhaseType | TimeType | AmplitudeType | FrameType | WaveformType,
)
