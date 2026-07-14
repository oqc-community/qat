# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from numbers import Number
from typing import Generic

import numpy as np
from xdsl.dialects.builtin import ComplexType, FloatData, IntAttr, f64
from xdsl.dialects.complex import ComplexNumberAttr
from xdsl.ir import Data
from xdsl.irdl import ParametrizedAttribute, irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.pulse.units import (
    FREQUENCY_UNIT_EXPONENTS,
    TIME_UNIT_EXPONENTS,
    FrequencyUnits,
    TimeUnits,
)

from .types import (
    PULSE_VAR_TYPE,
    AmplitudeType,
    FrequencyType,
    PhaseType,
    TimeType,
    WaveformType,
)


class PulseNumericTypedAttr(ParametrizedAttribute, Generic[PULSE_VAR_TYPE], ABC):
    """Base class for attributes in the pulse dialect that have a type associated with them.

    This is used to group together attributes that represent typed values, such as
    frequencies, phases, times and amplitudes.
    """

    @property
    @abstractmethod
    def literal_value(self) -> Number:
        """Converts the attribute to a literal value, which is returned as a numeric value,
        and specified by the concrete subclass."""
        pass

    @property
    @abstractmethod
    def associated_type(self) -> type[PULSE_VAR_TYPE]:
        """Returns the type that is associated with this attribute, which is specified by
        the concrete subclass."""
        pass


@irdl_attr_definition
class TimeUnitsData(Data[TimeUnits]):
    """Data attribute for representing time units in the pulse dialect."""

    name = "pulse.time_units"

    def print_parameter(self, printer: Printer) -> None:
        """Prints the parameters of the attribute, which are expected to be a string
        representing the time units."""
        printer.print_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> TimeUnits:
        """Parses the parameters of the attribute, which are expected to be a string
        representing the time units."""

        unit_str = parser.parse_str_literal()

        try:
            return TimeUnits(unit_str)
        except ValueError as e:
            raise ValueError(f"Unable to resolve time units {unit_str}.") from e


@irdl_attr_definition
class TimeAttr(PulseNumericTypedAttr[TimeType]):
    """An attribute that represents a compile-time constant time.

    This attribute intentionally does not specify the precision of the time value, and uses
    the standard Python precision for its respective type. The representation of the value
    will be set by the target.

    :ivar value: The time value, which can be a float or an integer.
    :ivar unit: The time units, which is an instance of the :class:`TimeUnits` enum.
    """

    name = "pulse.time_attr"
    value: FloatData | IntAttr
    unit: TimeUnitsData

    def __init__(self, value: float | int, unit: TimeUnits = TimeUnits.SECOND):
        """
        :param value: The time value in seconds, which can be a float or an integer.
        :param unit: The time units, which is an instance of the :class:`TimeUnits` enum
            and defaults to seconds if not provided.
        """

        value = IntAttr(value) if isinstance(value, int) else FloatData(value)
        return super().__init__(value, TimeUnitsData(unit))

    @property
    def literal_value(self) -> float | int:
        """Returns the time value in seconds."""
        return self.value.data * 10 ** TIME_UNIT_EXPONENTS[self.unit.data]

    @property
    def associated_type(self) -> type[TimeType]:
        """Returns the associated dialect type."""
        return TimeType

    @classmethod
    def from_literal_value(
        cls, value: float | int, unit: TimeUnits = TimeUnits.SECOND
    ) -> "TimeAttr":
        """Creates a time attribute from a canonical value in seconds.

        :param value: The time value in seconds.
        :param unit: The unit used to store the value.
        :returns: A time attribute storing ``value`` in the requested unit.
        """

        value_in_unit = value / 10 ** TIME_UNIT_EXPONENTS[unit]
        if value_in_unit.is_integer():
            value_in_unit = int(value_in_unit)

        return cls(value_in_unit, unit)


@irdl_attr_definition
class FrequencyUnitsData(Data[FrequencyUnits]):
    """Data attribute for representing frequency units in the pulse dialect."""

    name = "pulse.frequency_units"

    def print_parameter(self, printer: Printer) -> None:
        """Prints the parameters of the attribute, which are expected to be a string
        representing the frequency units."""
        printer.print_string_literal(self.data.value)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> FrequencyUnits:
        """Parses the parameters of the attribute, which are expected to be a string
        representing the frequency units."""

        unit_str = parser.parse_str_literal()
        try:
            return FrequencyUnits(unit_str)
        except ValueError as e:
            raise ValueError(f"Unable to resolve frequency units {unit_str}.") from e


@irdl_attr_definition
class FrequencyAttr(PulseNumericTypedAttr[FrequencyType]):
    """An attribute that represents a compile-time constant frequency.

    :ivar value: The frequency value, which can be a float or an integer.
    :ivar unit: The frequency units, which is an instance of the :class:`FrequencyUnits`
        enum.
    """

    name = "pulse.frequency_attr"
    value: FloatData | IntAttr
    unit: FrequencyUnitsData

    def __init__(self, value: float | int, unit: FrequencyUnits = FrequencyUnits.HERTZ):
        """
        :param value: The frequency value in Hz, which can be a float or an integer.
        :param unit: The frequency units, which is an instance of the
            :class:`FrequencyUnits`, enum and defaults to hertz if not provided.
        """

        value = IntAttr(value) if isinstance(value, int) else FloatData(value)
        return super().__init__(value, FrequencyUnitsData(unit))

    @property
    def literal_value(self) -> float | int:
        """Returns the frequency value in Hertz."""
        return self.value.data * 10 ** FREQUENCY_UNIT_EXPONENTS[self.unit.data]

    @property
    def associated_type(self) -> type[FrequencyType]:
        """Returns the associated dialect type."""
        return FrequencyType

    @classmethod
    def from_literal_value(
        cls, value: float | int, unit: FrequencyUnits = FrequencyUnits.HERTZ
    ) -> "FrequencyAttr":
        """Creates a frequency attribute from a canonical value in Hertz.

        :param value: The frequency value in Hertz.
        :param unit: The unit used to store the value.
        :returns: A frequency attribute storing ``value`` in the requested unit.
        """

        value_in_unit = value / 10 ** FREQUENCY_UNIT_EXPONENTS[unit]
        if value_in_unit.is_integer():
            value_in_unit = int(value_in_unit)

        return cls(value_in_unit, unit)


@irdl_attr_definition
class PhaseAttr(PulseNumericTypedAttr[PhaseType]):
    """An attribute that represents a compile-time constant phase. Phases are represented by
    radians.

    :ivar value: The phase value, which is expected to be a float representing the phase in
        radians.
    """

    name = "pulse.phase_attr"
    value: FloatData

    def __init__(self, value: float):
        """
        :param value: The phase value in radians, represented as a float.
        """

        return super().__init__(FloatData(value))

    @property
    def literal_value(self) -> float:
        """Returns the phase value."""
        return self.value.data

    @property
    def associated_type(self) -> type[PhaseType]:
        """Returns the associated dialect type."""
        return PhaseType


@irdl_attr_definition
class AmplitudeAttr(PulseNumericTypedAttr[AmplitudeType]):
    """An attribute that represents a compile-time constant amplitude.

    :ivar real: The real part of the amplitude, which is expected to be a float representing
        the amplitude in arbitrary units.
    :ivar imag: The imaginary part of the amplitude, which is expected to be a float
        representing the amplitude in arbitrary units, and defaults to 0.0 if not provided.
    """

    name = "pulse.amplitude_attr"
    real: FloatData
    imag: FloatData

    def __init__(self, value: complex | float):
        """
        :param value: The amplitude value, which can be a complex number or a float. If a
            float is provided, the imaginary part is set to 0.0.
        """

        real = value.real if isinstance(value, complex) else value
        imag = value.imag if isinstance(value, complex) else 0.0
        return super().__init__(FloatData(real), FloatData(imag))

    @property
    def literal_value(self) -> complex:
        """Returns the amplitude value as a complex number."""
        return complex(self.real.data, self.imag.data)

    @property
    def associated_type(self) -> type[AmplitudeType]:
        """Returns the associated dialect type."""
        return AmplitudeType


@irdl_attr_definition
class NumericArrayData(Data[np.ndarray[np.complexfloating]]):
    """Stores numeric arrays for use in attributes.

    Manipulations of sampled waveforms and weight vectors are processed using numpy due to
    its performance, and for this reason, we store this data as a numpy array.

    To be future thinking, and compatible with integrations with MLIR in the future, we
    print and parse the data as a list of builtin attributes, with the conversion handled in
    the interface.
    """

    name = "pulse.numeric_array_data"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> np.ndarray[np.complex128]:
        """Parses the parameters of the attribute, which are expected to be a list of
        builtin complex attributes representing elements within the array."""
        elements = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_attribute
        )

        samples: list[np.complex128] = []
        for element in elements:
            if not isinstance(element, ComplexNumberAttr):
                raise ValueError(
                    "Expected numeric array elements to be builtin complex attributes."
                )
            samples.append(np.complex128(complex(element.real.data, element.imag.data)))
        return np.asarray(samples, dtype=np.complex128)

    def print_parameter(self, printer: Printer) -> None:
        """Prints the parameters as textual MLIR, which are provided as complex  attributes
        in a list."""
        with printer.in_square_brackets():
            printer.print_list(
                self.data.tolist(),
                lambda x: printer.print_attribute(
                    ComplexNumberAttr(x.real, x.imag, ComplexType(f64))
                ),
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NumericArrayData):
            return False
        return np.array_equal(self.data, other.data, equal_nan=True)

    def __hash__(self) -> int:
        return hash(self.data.tobytes())


@irdl_attr_definition
class SampledWaveformAttr(PulseNumericTypedAttr[WaveformType]):
    """An attribute that represents a sampled waveform, which is represented by a real or
    complex numpy array."""

    name = "pulse.sampled_waveform"
    samples: NumericArrayData
    width: TimeAttr
    sample_time: TimeAttr

    def __init__(
        self,
        samples: np.ndarray[complex] | list[complex],
        width: TimeAttr,
        sample_time: TimeAttr,
    ):
        """
        :param samples: The samples of the waveform, represented as a numpy array or a list
            of floats or complex numbers.
        :param width: The total width of the waveform, represented as a TimeAttr.
        :param sample_time: The time between samples, represented as a TimeAttr.
        """
        samples = np.asarray(samples, dtype=np.complex128)
        return super().__init__(NumericArrayData(samples), width, sample_time)

    @property
    def literal_value(self) -> np.ndarray:
        """Returns the samples of the waveform as a numpy array."""
        return self.samples.data

    @property
    def associated_type(self) -> type[WaveformType]:
        """Returns the associated dialect type."""
        return WaveformType

    def verify(self) -> None:
        super().verify()

        if self.samples.data.ndim != 1:
            raise VerifyException(
                "Sampled waveform samples must be a one-dimensional array."
            )

        expected_width = len(self.samples.data) * self.sample_time.literal_value
        actual_width = self.width.literal_value
        if not np.isclose(actual_width, expected_width):
            raise VerifyException(
                "Sampled waveform width must equal number of samples multiplied by "
                "sample_time."
            )


@irdl_attr_definition
class WeightsAttr(ParametrizedAttribute):
    """An attribute that represents a set of weights that can be used in demodulation.

    This is expected to be optionally attached to acquire operations.

    :ivar weights: The weights, represented as a numpy array of complex values.
    """

    name = "pulse.weights"
    weights: NumericArrayData

    def __init__(self, weights: np.ndarray[np.complexfloating] | list[complex | float]):
        """
        :param weights: The weights, represented as a numpy array or a list of complex
            values.
        """
        weights = np.asarray(weights, dtype=np.complex128)
        return super().__init__(NumericArrayData(weights))

    def verify(self) -> None:
        super().verify()
        if self.weights.data.ndim != 1:
            raise VerifyException("Weights must be a one-dimensional array.")
