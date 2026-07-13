# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from typing import TypeVar

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import TypeAttribute
from xdsl.irdl import ParametrizedAttribute, irdl_attr_definition, param_def


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

    :ivar port_kind: A target-resolved token used to identify the port class for this frame,
        without encoding hardware object details directly in the IR. This name can take any
        string, but is expected to match a meaningful port class in the context of the
        system data. The default is "output", which for example, is the standard port kind
        for a qubit drive frame.
    """

    name = "pulse.frame"
    port_kind: StringAttr = param_def()

    def __init__(self, port_kind: StringAttr | str = "output"):
        if isinstance(port_kind, str):
            port_kind = StringAttr(port_kind)
        return super().__init__(port_kind)


@irdl_attr_definition
class WaveformType(ParametrizedAttribute, TypeAttribute):
    """Represents a waveform type."""

    name = "pulse.waveform"


@irdl_attr_definition
class AcquisitionType(ParametrizedAttribute, TypeAttribute):
    """Represents an acquisition type.

    Abstractly this represents the acquisition signal result for a given duration from an
    acquire operation, which we can do further processing on, such as integration to map it
    to an IQ value, or addition with other acquisition signals to achieve shot-averaged
    time-series data.
    """

    name = "pulse.acquisition"


@irdl_attr_definition
class IQResultType(ParametrizedAttribute, TypeAttribute):
    """Represents an IQ result type.

    This type exists instead of using the builtin complex type to allow for a more explicit
    representation of IQ values acquired from a quantum system, and to avoid hardware-
    specific typing details leaking into the IR. In practice, this can be treated as a
    complex number.
    """

    name = "pulse.iq_result"


PULSE_VAR_TYPE = TypeVar(
    "PULSE_VAR_TYPE",
    bound=FrequencyType
    | PhaseType
    | TimeType
    | AmplitudeType
    | WaveformType
    | AcquisitionType,
)
