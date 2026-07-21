# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from math import isclose, isnan
from numbers import Number
from typing import ClassVar, Generic

import numpy as np
from immutabledict import immutabledict
from xdsl.dialects.builtin import ArrayAttr, ComplexType, FloatData, IntAttr, f64
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


@irdl_attr_definition
class ComplexData(Data[complex]):
    """Store a Pythonic complex data type.

    Comparatively to the ``complex.ComplexNumberAttr`` type, this data object can
    store a Pythonic complex number without having to specify bitwidth (with its counterpart
    also breaking it up into two real components). This is analogous to the FloatData, which
    allows us to specify floats without specifying bitwidth. For the high level of
    abstraction targeted by the pulse dialect, we do not need to specify details like
    bitwidth, as this is specified when lowering to lower levels, be it a classical CPU, or
    quantum control system.

    .. note::

        This data type is specified in the absence of an existing one in xDSL. If that
        changes in the future, this type would be marked for deprecation.
    """

    name = "pulse.complex_data"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> complex:
        """Parses the parameter represented as a string into a complex number."""

        elements = parser.parse_comma_separated_list(
            parser.Delimiter.ANGLE, parser.parse_float
        )
        if (num_el := len(elements)) != 2:
            parser.raise_error(
                f"Expected two elements when parsing ``pulse.ComplexData``. Found {num_el}"
            )
        return complex(elements[0], elements[1])

    def print_parameter(self, printer: Printer):
        """Prints the complex parameter, representing the real and imaginary components as a
        two-element tuple."""
        with printer.in_angle_brackets():
            printer.print_string(f"{self.data.real},{self.data.imag}")

    def __eq__(self, other: object):
        # avoid triggering `float('nan') != float('nan')` inequality
        if not isinstance(other, ComplexData):
            return False

        real_equality = (
            isnan(self.data.real) and isnan(other.data.real)
        ) or self.data.real == other.data.real
        imag_equality = (
            isnan(self.data.imag) and isnan(other.data.imag)
        ) or self.data.imag == other.data.imag
        return real_equality and imag_equality

    def __hash__(self):
        return hash(self.data)


@irdl_attr_definition
class EqualiseAttr(ParametrizedAttribute):
    """An attribute that represents an affine transformation, used as part of the post-
    processing pipeline for acquired signals.

    In a complex space, an affine transformation of a complex number z can be represented
    as:

    .. math:: z' = a * z + b * conj(z) + c

    where:

    * :math:`a` is the (complex) linear coefficient,
    * :math:`b` is the (complex) conjugate coefficient,
    * :math:`c` is the (complex) translation.

    If we consider this from a two-dimensional real space perspective, we can represent the
    affine transformation as

    .. math:: z' = A * z + C

    where ``A`` is an arbitrary 2x2 real matrix representing the linear transformation, and
    ``C`` is an arbitrary 2D real vector representing the translation.

    This attribute stores the transformation using the three complex numbers to be
    consistent with the complex representation of the IQ space. However, it offers utilities
    to expose the transformation in a real space representation, and also build from it.
    This allows a compact representation that is consistent with legacy PuRR
    ``linear_map_complex_to_real``, and also the more generalised affine transformations
    defined in pydantic pipelines.
    """

    name = "pulse.affine_transform"

    linear_coefficient: ComplexData
    conjugate_coefficient: ComplexData
    translation: ComplexData

    def __init__(
        self,
        linear_coefficient: ComplexData | complex,
        conjugate_coefficient: ComplexData | complex,
        translation: ComplexData | complex,
    ):
        """
        :param linear_coefficient: The (complex) linear coefficient.
        :param conjugate_coefficient: The (complex) conjugate coefficient.
        :param translation: The (complex) translation.
        """
        if not isinstance(linear_coefficient, ComplexData):
            linear_coefficient = ComplexData(complex(linear_coefficient))
        if not isinstance(conjugate_coefficient, ComplexData):
            conjugate_coefficient = ComplexData(complex(conjugate_coefficient))
        if not isinstance(translation, ComplexData):
            translation = ComplexData(complex(translation))
        return super().__init__(linear_coefficient, conjugate_coefficient, translation)

    @property
    def linear_matrix(self) -> np.ndarray[np.float64]:
        """Returns the linear transformation matrix in real space representation."""
        a = self.linear_coefficient
        b = self.conjugate_coefficient
        return np.array(
            [
                [a.data.real + b.data.real, -a.data.imag + b.data.imag],
                [a.data.imag + b.data.imag, a.data.real - b.data.real],
            ],
            dtype=np.float64,
        )

    @property
    def translation_vector(self) -> np.ndarray[np.float64]:
        """Returns the translation vector in real space representation."""
        c = self.translation
        return np.array([c.data.real, c.data.imag], dtype=np.float64)

    @classmethod
    def from_real_space(
        cls,
        linear_matrix: np.ndarray[np.float64],
        translation_vector: np.ndarray[np.float64],
    ) -> "EqualiseAttr":
        """Creates an EqualiseAttr from a real space representation.

        :param linear_matrix: A 2x2 real matrix representing the linear transformation.
        :param translation_vector: A 2D real vector representing the translation.
        :returns: An instance of EqualiseAttr.
        """
        if linear_matrix.shape != (2, 2):
            raise ValueError("Linear matrix must be 2x2.")
        if translation_vector.shape != (2,):
            raise ValueError("Translation vector must be 2D.")

        a_real = (linear_matrix[0, 0] + linear_matrix[1, 1]) / 2
        a_imag = (linear_matrix[1, 0] - linear_matrix[0, 1]) / 2
        b_real = (linear_matrix[0, 0] - linear_matrix[1, 1]) / 2
        b_imag = (linear_matrix[1, 0] + linear_matrix[0, 1]) / 2

        a = complex(a_real, a_imag)
        b = complex(b_real, b_imag)
        c = complex(translation_vector[0], translation_vector[1])
        return cls(a, b, c)


@irdl_attr_definition
class StateMapDictAttr(Data[immutabledict[int, IntAttr]]):
    """An attribute that represents a mapping from integer state labels to integer data,
    which for example, could be another state label, or a binary value.

    The expected use cases are for mapping state labels from state discrimination onto
    binary outputs.
    """

    name = "pulse.state_map_dict"

    # The only current need for this is a mapping to binary data, but we can expand this to
    # store arbitrary (generic'd) attributes if needed

    def __init__(self, data: Mapping[int, IntAttr | int]):
        """Initializes the state map dictionary attribute.

        :param data: A mapping from integer state labels to integer attributes associated
            with those states.
        """
        return super().__init__(
            immutabledict(
                {k: v if isinstance(v, IntAttr) else IntAttr(v) for k, v in data.items()}
            )
        )

    def print_parameter(self, printer: Printer):
        """Prints the parameters of the attribute, which are expected to be a dictionary
        representing the state mapping."""
        printer.print_attr_dict({str(k): v for k, v in self.data.items()})

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> immutabledict[int, IntAttr]:
        """Parses the parameters of the attribute, which are expected to be a dictionary
        representing the state mapping."""
        state_map_dict = parser.parse_optional_dictionary_attr_dict()
        parsed_dict = {}
        for key, value in state_map_dict.items():
            if not isinstance(value, IntAttr):
                parser.raise_error(
                    "Expected state map values to be builtin integer attributes."
                )
            try:
                parsed_key = int(key)
            except (ValueError, TypeError):
                parser.raise_error(
                    f"Expected state map keys to be integer keys, got '{key}'."
                )
            parsed_dict[parsed_key] = value
        return immutabledict(parsed_dict)


class DiscriminatorPolicyAttr(ParametrizedAttribute, ABC):
    """Parent class for all discriminator policies, which are used to determine how to
    process IQ values.

    State labels are non-negative integers, with the exception of -1, which is reserved for
    representing an unmapped state, such as IQ values which cannot be confidently mapped to
    any state.
    """

    name = "pulse.discriminator_policy"
    POLICY_NAME: ClassVar[str]

    @property
    @abstractmethod
    def state_range(self) -> tuple[int, int]:
        """Returns the range of state labels that the discriminator policy can discriminate
        between."""
        ...


@irdl_attr_definition
class RealThresholdPolicyAttr(DiscriminatorPolicyAttr):
    """A discriminator policy that discriminates between two states based on a real
    threshold.

    The discriminator works by mapping the IQ values to states ``{0, 1}`` according to the
    classifier

    .. math::

        \\text{state}(z) = \\begin{cases}
            0 & \\text{if } \\Re(z) < \\text{threshold} \\\\
            1 & \\text{if } \\Re(z) \\geq \\text{threshold}
        \\end{cases}

    :ivar threshold: The threshold value, which is used to determine the state of the IQ
        value.
    """

    name = "pulse.real_threshold_policy"
    POLICY_NAME: ClassVar[str] = "real_threshold"

    threshold: FloatData

    def __init__(self, threshold: float = 0.0):
        """
        :param threshold: The threshold value, which is used to determine the state of the
            IQ value.
        """
        return super().__init__(FloatData(threshold))

    @property
    def state_range(self) -> tuple[int, int]:
        """The range of state labels that the discriminator policy can discriminate
        between."""
        return (0, 1)


@irdl_attr_definition
class MaximumLikelihoodPolicyAttr(DiscriminatorPolicyAttr):
    """A discriminator policy that discriminates between multiple states based on a maximum
    likelihood approach.

    State labels are non-negative integers, each mapping to a center represented by a
    complex number. Every IQ point can be assigned to the state label ``k`` with the highest
    normalised likelihood:

    .. math::

        \\tilde{p}_k(z) = \\frac{L_k(z)}{\\sum_j L_j(z)},
        \\quad L_k(z) = \\exp\\!\\left(-\\frac{|z - \\mathrm{loc}_k|^2}{2\\,\\nu}\\right)

    where :math:`\\nu` is the noise power (variance) ``noise_est``.

    If the maximum likelihood is below a threshold ``p_min``, the IQ point is assigned to
    the state ``-1`` (unmapped).

    :ivar state_centers: A list of the different state centers which map respectively to
        their state labels, which are ordered from 0 to ``len(state_centers) - 1``. Each
        center is represented by a complex number.
    :ivar noise_estimate: The noise power (variance) of the IQ values.
    :ivar p_min: The minimum normalised likelihood required to assign an IQ point to a
        state label. If the maximum normalised likelihood is below this threshold, the IQ
        point is assigned to the state ``-1`` (unmapped).
    """

    name = "pulse.maximum_likelihood_policy"
    POLICY_NAME: ClassVar[str] = "maximum_likelihood"

    state_centers: ArrayAttr[ComplexData]
    noise_estimate: FloatData
    p_min: FloatData

    def __init__(
        self,
        state_centers: Iterable[complex | ComplexData],
        noise_estimate: float = 1.0,
        p_min: float = 0.0,
    ):
        """
        :param state_centers: A list of the different state centers which map respectively
            to their state labels, which are ordered from 0 to ``len(state_centers) - 1``.
            Each center is represented by a complex number.
        :param noise_estimate: The noise power (variance) of the IQ values. Defaults to 1.0
            if not provided. Noise estimate is only important if ``p_min`` is set to a value
            greater than 0.0, as it is used to calculate the normalised likelihoods of the
            IQ points. If ``p_min`` is set to 0.0, the noise estimate is ignored.
        :param p_min: The minimum normalised likelihood required to assign an IQ point to a
            state label. If the maximum normalised likelihood is below this threshold, the
            IQ point is assigned to the state ``-1`` (unmapped).
        """
        state_centers_data = ArrayAttr(
            ComplexData(c) if not isinstance(c, ComplexData) else c for c in state_centers
        )
        return super().__init__(
            state_centers_data, FloatData(noise_estimate), FloatData(p_min)
        )

    def verify(self):
        """Validates the properties of the maximum likelihood policy attribute."""

        super().verify()

        if len(self.state_centers) == 0:
            raise VerifyException("state_centers must contain at least one mapped state.")

        p_min_value = self.p_min.data
        if not (
            isclose(p_min_value, 0.0)
            or isclose(p_min_value, 1.0)
            or (0.0 < p_min_value < 1.0)
        ):
            raise VerifyException(
                f"p_min must be in the range [0, 1], but got {p_min_value}."
            )

    @property
    def state_range(self) -> tuple[int, int]:
        """The range of state labels that the discriminator policy can discriminate
        between."""
        return (-1, len(self.state_centers) - 1)


class ExternalPolicyAttr(DiscriminatorPolicyAttr):
    """Stub for a future extension that will allow custom (off-chip) state discrimination to
    be hooked into the runtime post-processing facilities."""

    ...
