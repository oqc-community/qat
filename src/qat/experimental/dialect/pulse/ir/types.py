# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from typing import TypeVar

from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.ir import TypeAttribute
from xdsl.irdl import ParametrizedAttribute, irdl_attr_definition, param_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


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

    :ivar port: A target-resolved token used to identify the port for this frame, without
        encoding hardware object details directly in the IR. This name can take any string,
        but is expected to match a meaningful port in the context of the system data.
    """

    name = "pulse.frame"
    port: StringAttr = param_def()

    def __init__(self, port: StringAttr | str):
        if isinstance(port, str):
            port = StringAttr(port)
        return super().__init__(port)


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


@irdl_attr_definition
class StateKeyType(ParametrizedAttribute, TypeAttribute):
    """Represents a discriminated state key type with integer labels between inclusive
    bounds.

    This type is the result of state discrimination, which maps IQ values from qubit
    readout to integer state identifiers. The bounds encode the valid range of state labels
    that can be produced by a discrimination policy. For example:

    - A real-threshold discriminator produces states {0, 1}
    - A maximum-likelihood discriminator produces states {-1, 0, ..., n-1}, where -1
      represents an unmapped/ambiguous state

    The parametrized bounds enable compile-time verification that downstream operations
    (like state mapping) cover all possible state outcomes.

    :ivar min_state: The smallest allowed integer state label (inclusive).
    :ivar max_state: The largest allowed integer state label (inclusive).
    """

    name = "pulse.state"
    min_state: IntAttr = param_def()
    max_state: IntAttr = param_def()

    def __init__(self, min_state: int, max_state: int):
        return super().__init__(IntAttr(min_state), IntAttr(max_state))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[IntAttr]:
        with parser.in_angle_brackets():
            min_state = parser.parse_integer(allow_negative=True)
            parser.parse_punctuation(",", " between minimum and maximum state bounds")
            max_state = parser.parse_integer(allow_negative=True)

        return [IntAttr(min_state), IntAttr(max_state)]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(f"{self.min_state.data}, {self.max_state.data}")

    def verify(self) -> None:
        if self.min_state.data > self.max_state.data:
            raise VerifyException(
                f"StateKeyType bounds invalid: minimum state ({self.min_state.data}) "
                f"must be less than or equal to maximum state ({self.max_state.data})."
            )

    @property
    def state_range(self) -> tuple[int, int]:
        """Returns the valid range of state labels as a (min, max) tuple.

        All integers in this range (inclusive) are valid state identifiers produced by the
        associated discrimination policy. Used for verification that state mapping
        operations cover all possible outcomes.
        """

        return (self.min_state.data, self.max_state.data)
