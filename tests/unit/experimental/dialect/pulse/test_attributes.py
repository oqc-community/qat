# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import re
from collections.abc import Mapping
from copy import deepcopy
from math import isclose, nan

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, FloatData, IntAttr
from xdsl.dialects.complex import Complex
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, VerifyException

from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    AmplitudeType,
    ComplexData,
    EqualiseAttr,
    FrequencyAttr,
    FrequencyType,
    MaximumLikelihoodPolicyAttr,
    PhaseAttr,
    PhaseType,
    Pulse,
    SampledWaveformAttr,
    TimeAttr,
    TimeType,
    WaveformType,
    WeightsAttr,
)
from qat.experimental.dialect.pulse.ir.attributes import (
    RealThresholdPolicyAttr,
    StateMapDictAttr,
)
from qat.experimental.dialect.pulse.units import FrequencyUnits, TimeUnits


class TestTimeAttr:
    @pytest.mark.parametrize(
        "value, units, type_, value_in_seconds",
        [
            (80e-9, TimeUnits.SECOND, float, 80e-9),
            (800, TimeUnits.NANOSECOND, int, 800e-9),
            (4, TimeUnits.MICROSECOND, int, 4e-6),
            (8e-5, TimeUnits.MILLISECOND, float, 8e-8),
            (0, TimeUnits.SECOND, int, 0.0),
            (0.0, TimeUnits.SECOND, float, 0.0),
        ],
    )
    def test_properties(self, value, units, type_, value_in_seconds):
        attr = TimeAttr(value, units)
        assert attr.value.data == value
        assert attr.unit.data == units
        assert isinstance(attr.value.data, type_)
        assert np.isclose(attr.literal_value, value_in_seconds)
        assert attr.associated_type is TimeType
        attr.verify()  # should succeed

    @pytest.mark.parametrize(
        "value, units, type_",
        [
            (0.04, TimeUnits.SECOND, "float_data"),
            (8, TimeUnits.NANOSECOND, "int"),
        ],
    )
    def test_print_and_parse_roundtrip(self, value, units, type_, io_stream):
        attr = TimeAttr(value, units)
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()

        pattern = r"#pulse\.time_attr<\s*#builtin\.(.*?)<(.*?)>,\s*#pulse\.time_units\"(.*?)\"\s*>"
        match = re.search(pattern, output)
        assert match is not None
        type_string, value_str, unit_str = match.groups()
        assert type_string == type_
        assert np.isclose(float(value_str), value)
        assert unit_str == units.value

        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, output)
        parsed_attr = parser.parse_attribute()
        assert parsed_attr.value.data == value
        assert parsed_attr.unit.data == units
        assert parsed_attr == attr

    def test_parse_with_unknown_units_raises_parse_exception(self):
        attr_str = '#pulse.time_attr<#builtin.float_data<0.1>, #pulse.time_units"GHz">'
        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, attr_str)
        with pytest.raises(ValueError, match=r"Unable to resolve time units GHz."):
            parser.parse_attribute()

    def test_equality(self):
        attr1 = TimeAttr(80e-9, TimeUnits.SECOND)
        attr2 = TimeAttr(80e-9, TimeUnits.SECOND)
        assert attr1 == attr2

    def test_inequality_with_value(self):
        attr1 = TimeAttr(80e-9, TimeUnits.SECOND)
        attr2 = TimeAttr(90e-9, TimeUnits.SECOND)
        assert attr1 != attr2

    def test_inequality_with_units(self):
        attr1 = TimeAttr(80e-9, TimeUnits.SECOND)
        attr2 = TimeAttr(80e-9, TimeUnits.NANOSECOND)
        assert attr1 != attr2

    def test_inequality_with_different_value_type(self):
        """Documents how xDSL equality works, and is not tested for each individual
        attribute."""
        assert 80 == 80.0
        attr1 = TimeAttr(80.0, TimeUnits.SECOND)
        attr2 = TimeAttr(80, TimeUnits.SECOND)
        assert attr1 != attr2

    @pytest.mark.parametrize(
        "value, units, stored_value, type_",
        [
            (8e-9, TimeUnits.NANOSECOND, 8, int),
            (0.5e-9, TimeUnits.NANOSECOND, 0.5, float),
        ],
    )
    def test_from_literal_value_preserves_integer_values(
        self, value, units, stored_value, type_
    ):
        attr = TimeAttr.from_literal_value(value, units)

        assert attr.value.data == stored_value
        assert isinstance(attr.value.data, type_)
        assert attr.unit.data == units
        assert np.isclose(attr.literal_value, value)


class TestFrequencyAttr:
    @pytest.mark.parametrize(
        "value, units, type_, value_in_hz",
        [
            (5e9, FrequencyUnits.HERTZ, float, 5e9),
            (5000, FrequencyUnits.KILOHERTZ, int, 5e6),
            (5, FrequencyUnits.MEGAHERTZ, int, 5e6),
            (0.005, FrequencyUnits.GIGAHERTZ, float, 5e6),
            (0, FrequencyUnits.HERTZ, int, 0.0),
            (0.0, FrequencyUnits.HERTZ, float, 0.0),
        ],
    )
    def test_properties(self, value, units, type_, value_in_hz):
        attr = FrequencyAttr(value, units)
        assert attr.value.data == value
        assert attr.unit.data == units
        assert isinstance(attr.value.data, type_)
        assert np.isclose(attr.literal_value, value_in_hz)
        assert attr.associated_type is FrequencyType
        attr.verify()  # should succeed

    @pytest.mark.parametrize(
        "value, units, type_",
        [
            (5.0, FrequencyUnits.HERTZ, "float_data"),
            (5, FrequencyUnits.KILOHERTZ, "int"),
        ],
    )
    def test_print_and_parse_roundtrip(self, value, units, type_, io_stream):
        attr = FrequencyAttr(value, units)
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()

        pattern = r"#pulse\.frequency_attr<\s*#builtin\.(.*?)<(.*?)>,\s*#pulse\.frequency_units\"(.*?)\"\s*>"
        match = re.search(pattern, output)
        assert match is not None
        type_string, value_str, unit_str = match.groups()
        assert type_string == type_
        assert np.isclose(float(value_str), value)
        assert unit_str == units.value

        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, output)
        parsed_attr = parser.parse_attribute()
        assert parsed_attr.value.data == value
        assert parsed_attr.unit.data == units
        assert parsed_attr == attr

    def test_parse_with_unknown_units_raises_parse_exception(self):
        attr_str = (
            '#pulse.frequency_attr<#builtin.float_data<5.0>, #pulse.frequency_units"ns">'
        )
        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, attr_str)
        with pytest.raises(ValueError, match=r"Unable to resolve frequency units ns."):
            parser.parse_attribute()

    def test_equality(self):
        attr1 = FrequencyAttr(5e9, FrequencyUnits.HERTZ)
        attr2 = FrequencyAttr(5e9, FrequencyUnits.HERTZ)
        assert attr1 == attr2

    def test_inequality_with_value(self):
        attr1 = FrequencyAttr(5e9, FrequencyUnits.HERTZ)
        attr2 = FrequencyAttr(6e9, FrequencyUnits.HERTZ)
        assert attr1 != attr2

    def test_inequality_with_units(self):
        attr1 = FrequencyAttr(5e9, FrequencyUnits.HERTZ)
        attr2 = FrequencyAttr(5e9, FrequencyUnits.KILOHERTZ)
        assert attr1 != attr2

    @pytest.mark.parametrize(
        "value, units, stored_value, type_",
        [
            (5e6, FrequencyUnits.MEGAHERTZ, 5, int),
            (5.5e6, FrequencyUnits.MEGAHERTZ, 5.5, float),
        ],
    )
    def test_from_literal_value_preserves_integer_values(
        self, value, units, stored_value, type_
    ):
        attr = FrequencyAttr.from_literal_value(value, units)

        assert attr.value.data == stored_value
        assert isinstance(attr.value.data, type_)
        assert attr.unit.data == units
        assert np.isclose(attr.literal_value, value)


class TestPhaseAttr:
    @pytest.mark.parametrize("value", [0.0, -np.pi / 2, 3.14, 2])
    def test_properties(self, value):
        attr = PhaseAttr(value)
        assert attr.value.data == value
        assert np.isclose(attr.literal_value, value)
        assert attr.associated_type is PhaseType
        attr.verify()  # should succeed

    def test_equality(self):
        attr1 = PhaseAttr(3.14)
        attr2 = PhaseAttr(3.14)
        assert attr1 == attr2

    def test_inequality(self):
        attr1 = PhaseAttr(3.14)
        attr2 = PhaseAttr(2.0)
        assert attr1 != attr2


class TestAmplitudeAttr:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, -0.5, 0.3 - 0.254j, 1.0 + 1.0j])
    def test_properties(self, value):
        attr = AmplitudeAttr(value)
        assert attr.real.data == value.real
        assert attr.imag.data == value.imag
        assert np.isclose(attr.literal_value, value)
        assert attr.associated_type is AmplitudeType
        attr.verify()  # should succeed

    def test_equality(self):
        attr1 = AmplitudeAttr(0.3 - 0.254j)
        attr2 = AmplitudeAttr(0.3 - 0.254j)
        assert attr1 == attr2

    def test_inequality_with_different_real_value(self):
        attr1 = AmplitudeAttr(0.3 - 0.254j)
        attr2 = AmplitudeAttr(0.4 - 0.254j)
        assert attr1 != attr2

    def test_inequality_with_different_imaginary_value(self):
        attr1 = AmplitudeAttr(0.3 - 0.254j)
        attr2 = AmplitudeAttr(0.3 - 0.255j)
        assert attr1 != attr2


class TestSampledWaveformAttr:
    @pytest.mark.parametrize(
        "waveform, time, sample_time",
        [
            ([0.0, 1.0, 0.5], 3e9, 1e9),
            ([1.0 + 1.0j, 0.5 - 0.5j], 2e6, 1e6),
        ],
    )
    def test_properties(self, waveform, time, sample_time):
        attr = SampledWaveformAttr(waveform, TimeAttr(time), TimeAttr(sample_time))
        assert np.array_equal(attr.samples.data, np.array(waveform, dtype=np.complex128))
        assert np.isclose(attr.width.value.data, time)
        assert np.isclose(attr.sample_time.value.data, sample_time)
        assert np.array_equal(attr.literal_value, np.array(waveform, dtype=np.complex128))
        assert attr.associated_type is WaveformType
        attr.verify()  # should succeed

    @pytest.mark.parametrize(
        "waveform",
        [
            [0.0, 1.0, 0.5],
            [1.0 + 1.0j, 0.5 - 0.5j, -1 / 3],
            [float("nan"), complex(float("inf"), float("nan")), float("-inf")],
        ],
    )
    def test_equality(self, waveform):
        attr = SampledWaveformAttr(waveform, TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr(waveform, TimeAttr(3e9), TimeAttr(1e9))
        assert attr == attr2

    def test_inequality_with_different_waveform(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.6], TimeAttr(3e9), TimeAttr(1e9))
        assert attr1 != attr2

    def test_inequality_with_different_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr(
            [0.0, 1.0, 0.5],
            TimeAttr(3e18, TimeUnits.NANOSECOND),
            TimeAttr(1e9),
        )
        assert attr1 != attr2

    def test_inequality_with_different_sample_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(6e9), TimeAttr(2e9))
        assert attr1 != attr2

    def test_print_and_parse_roundtrip(self, io_stream):
        waveform = [0.0, 1.0, 0.5, -1 / 3]
        time = TimeAttr(3.0, TimeUnits.NANOSECOND)
        sample_time = TimeAttr(1.0, TimeUnits.NANOSECOND)
        attr = SampledWaveformAttr(waveform, time, sample_time)
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()

        # Match the full output
        pattern = r"#pulse\.sampled_waveform<\s*#pulse\.numeric_array_data\[(.*?)\],\s*#pulse\.time_attr<(.*?)>,\s*#pulse\.time_attr<(.*?)>\s*>"
        match = re.search(pattern, output)
        assert match is not None
        waveform_contents, time_str, sample_time_str = match.groups()

        # Match the waveform elements
        element_pattern = r"#complex\.number<:f64 ([^,]+), ([^>]+)> : complex<f64>"
        elements = re.findall(element_pattern, waveform_contents)
        assert len(elements) == len(waveform)
        for i, (real, imag) in enumerate(elements):
            assert np.isclose(waveform[i], complex(float(real), float(imag)))

        # Match the time attribute contents
        time_pattern = r"#builtin\.(.*?)<(.*?)>,\s*#pulse\.time_units\"(.*?)\""
        time_match = re.search(time_pattern, time_str)
        assert time_match is not None

        time_type_str, time_value_str, time_unit_str = time_match.groups()
        assert time_type_str == "float_data"
        assert time_unit_str == TimeUnits.NANOSECOND.value
        assert np.isclose(float(time_value_str), time.value.data)

        sample_time_match = re.search(time_pattern, sample_time_str)
        assert sample_time_match is not None
        sample_time_type_str, sample_time_value_str, sample_time_unit_str = (
            sample_time_match.groups()
        )
        assert sample_time_type_str == "float_data"
        assert sample_time_unit_str == TimeUnits.NANOSECOND.value
        assert np.isclose(float(sample_time_value_str), sample_time.value.data)

        # Match the contents to the expectation
        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        context.load_dialect(Complex)
        parser = Parser(context, output)
        parsed_attr = parser.parse_attribute()
        assert parsed_attr == attr

    def test_parse_with_non_complex_type(self):
        attr_str = (
            "#pulse.sampled_waveform<#pulse.numeric_array_data[#builtin.float_data<0.0>, "
            "#builtin.float_data<1.0>, #builtin.float_data<0.5>], "
            '#pulse.time_attr<#builtin.float_data<3.0>, #pulse.time_units"ns">, '
            '#pulse.time_attr<#builtin.float_data<1.0>, #pulse.time_units"ns">>'
        )
        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, attr_str)
        with pytest.raises(
            ValueError, match="numeric array elements to be builtin complex attributes."
        ):
            parser.parse_attribute()

    def test_verify_fails_for_mismatched_width(self):
        with pytest.raises(
            VerifyException,
            match=(
                "Sampled waveform width must equal number of samples multiplied "
                "by sample_time."
            ),
        ):
            SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(4e9), TimeAttr(1e9))

    def test_hash_equality(self):
        waveform = [0.0, 1.0, 0.5, -1 / 3]
        attr1 = SampledWaveformAttr(waveform, TimeAttr(4e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr(deepcopy(waveform), TimeAttr(4e9), TimeAttr(1e9))
        assert hash(attr1) == hash(attr2)

    def test_hash_inequality_with_different_waveform(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.6], TimeAttr(3e9), TimeAttr(1e9))
        assert hash(attr1) != hash(attr2)

    def test_hash_inequality_with_different_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr(
            [0.0, 1.0, 0.5],
            TimeAttr(3e18, TimeUnits.NANOSECOND),
            TimeAttr(1e9),
        )
        assert hash(attr1) != hash(attr2)

    def test_hash_inequality_with_different_sample_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(6e9), TimeAttr(2e9))
        assert hash(attr1) != hash(attr2)

    def test_verify_fails_for_non_1d_array(self):
        """Sampled waveform samples can only be a one-dimensional array.

        This test ensures that the verify method raises an exception when a multi-
        dimensional array is provided.
        """
        with pytest.raises(
            VerifyException,
            match="Sampled waveform samples must be a one-dimensional array.",
        ):
            SampledWaveformAttr(
                [[0.0, 1.0], [0.5, 0.6]], TimeAttr(3e9), TimeAttr(1e9)
            ).verify()


class TestWeightsAttr:
    @pytest.mark.parametrize(
        "weights",
        [
            [0.0, 1.0, 0.5],
            [1.0 + 1.0j, 0.5 - 0.5j, -1 / 3],
        ],
    )
    def test_properties(self, weights):
        attr = WeightsAttr(weights)
        assert np.array_equal(attr.weights.data, np.array(weights, dtype=np.complex128))

    @pytest.mark.parametrize(
        "weights",
        [
            [0.0, 1.0, 0.5],
            [1.0 + 1.0j, 0.5 - 0.5j, -1 / 3],
            [float("nan"), complex(float("inf"), float("nan")), float("-inf")],
        ],
    )
    def test_equality(self, weights):
        attr1 = WeightsAttr(weights)
        attr2 = WeightsAttr(weights)
        assert attr1 == attr2

    def test_inequality_with_different_weights(self):
        attr1 = WeightsAttr([0.0, 1.0 + 0.1j, 0.5])
        attr2 = WeightsAttr([0.0, 1.0 + 0.2j, 0.5])
        assert attr1 != attr2

    def test_print_and_parse_roundtrip(self, io_stream):
        weights = [0.0, 1.0 + 1.0j, 0.5 - 0.5j, -1 / 3]
        attr = WeightsAttr(weights)
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()

        pattern = r"#pulse\.weights<\s*#pulse\.numeric_array_data\[(.*?)\]\s*>"
        match = re.search(pattern, output)
        assert match is not None
        weights_contents = match.group(1)

        element_pattern = r"#complex\.number<:f64 ([^,]+), ([^>]+)> : complex<f64>"
        elements = re.findall(element_pattern, weights_contents)
        assert len(elements) == len(weights)
        for i, (real, imag) in enumerate(elements):
            assert np.isclose(weights[i], complex(float(real), float(imag)))

        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        context.load_dialect(Complex)
        parser = Parser(context, output)
        parsed_attr = parser.parse_attribute()
        assert parsed_attr == attr

    def test_parse_with_non_complex_type(self):
        attr_str = (
            "#pulse.weights<#pulse.numeric_array_data[#builtin.float_data<0.0>, "
            "#builtin.float_data<1.0>, #builtin.float_data<0.5>]>"
        )
        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, attr_str)
        with pytest.raises(
            ValueError,
            match="Expected numeric array elements to be builtin complex attributes.",
        ):
            parser.parse_attribute()

    def test_hash_equality(self):
        weights = [0.0, 1.0 + 1.0j, 0.5 - 0.5j, -1 / 3]
        attr1 = WeightsAttr(weights)
        attr2 = WeightsAttr(deepcopy(weights))
        assert hash(attr1) == hash(attr2)

    def test_hash_inequality_with_different_weights(self):
        attr1 = WeightsAttr([0.0, 1.0 + 0.1j, 0.5])
        attr2 = WeightsAttr([0.0, 1.0 + 0.2j, 0.5])
        assert hash(attr1) != hash(attr2)

    def test_verify_fails_for_non_1d_array(self):
        """Weights can only be a one-dimensional array.

        This test ensures that the verify method raises an exception when a multi-
        dimensional array is provided.
        """
        with pytest.raises(
            VerifyException,
            match="Weights must be a one-dimensional array.",
        ):
            WeightsAttr([[0.0, 1.0], [0.5, 0.6]]).verify()


class TestComplexData:
    """Tests the complex data type, including its printing and parsing, equality checks and
    hashing."""

    def test_instantiates_from_complex(self):
        """Tests the type instantiates from a complex number."""

        data = ComplexData(1.0 + 2.54j)
        data.verify()
        assert data.data == 1.0 + 2.54j

    @pytest.mark.parametrize(
        "val1, val2",
        [
            (complex(2.54, 1.0), complex(2.54, 1.0)),
            (complex(nan, 1.1), complex(nan, 1.1)),
            (complex(1.1, nan), complex(1.1, nan)),
            (complex(nan, nan), complex(nan, nan)),
        ],
    )
    def test_equality_returns_true(self, val1, val2):
        """Tests the equality implementation, and specifically, nan handling."""

        assert ComplexData(val1) == ComplexData(val2)

    @pytest.mark.parametrize(
        "val1, val2",
        [
            (complex(2.54, 1.0), complex(2.53, 1.0)),
            (complex(2.54, 1.0), complex(2.54, 0.99)),
            (complex(nan, 1.1), complex(2.54, 1.1)),
            (complex(1.1, nan), complex(1.1, 2.54)),
        ],
    )
    def test_inequality_returns_true(self, val1, val2):
        """Tests the inequality implementation, and specifically, nan handling."""

        assert ComplexData(val1) != ComplexData(val2)

    def test_print_gives_expected_string(self, io_stream):
        """Tests that printing a complex number data type yields the expected string."""

        data = ComplexData(1.0 + 2.54j)
        printer = Printer(stream=io_stream)
        printer.print_attribute(data)
        output = io_stream.getvalue()
        assert output == "#pulse.complex_data<1.0,2.54>"

    def test_parse_gives_expected_complex_data(self):
        """Tests that parsing a complex data type yields the expected complex number."""

        attr_str = "#pulse.complex_data<1.0,2.54>"
        context = Context()
        context.load_dialect(Pulse)
        parser = Parser(context, attr_str)
        parsed_attr = parser.parse_attribute()
        assert isinstance(parsed_attr, ComplexData)
        assert parsed_attr.data == complex(1.0, 2.54)

    @pytest.mark.parametrize(
        "attr_string",
        [
            "#pulse.complex_data<>",
            "#pulse.complex_data<1.0>",
            "#pulse.complex_data<1.0,2.54,3.0>",
        ],
        ids=["zero_elements", "one_element", "three_elements"],
    )
    def test_parse_with_incorrect_amount_of_elements_raises(self, attr_string):
        """Tests that parsing a complex data type with an incorrect amount of elements
        raises an exception.

        Tests with zero, one and three elements, which should all raise an exception.
        """

        context = Context()
        context.load_dialect(Pulse)
        parser = Parser(context, attr_string)
        with pytest.raises(
            ParseError,
            match="Expected two elements when parsing ``pulse.ComplexData``.",
        ):
            parser.parse_attribute()

    def test_print_parse_roundtrip_yields_same_attribute(self, io_stream):
        """Tests that printing and parsing a complex data type yields the same attribute."""

        data = ComplexData(1.0 + 2.54j)
        context = Context()
        context.load_dialect(Pulse)
        printer = Printer(stream=io_stream)
        printer.print_attribute(data)
        output = io_stream.getvalue()
        parser = Parser(context, output)
        parsed_attr = parser.parse_attribute()
        assert parsed_attr == data


class TestEqualiseAttr:
    """Tests that EqualiseAttr can be instantiated as expected and validates correctly."""

    def test_instantiation_from_complex_numbers_raises_no_errors(self):
        """Tests that EqualiseAttr can be instantiated with valid parameters."""

        attr = EqualiseAttr(
            linear_coefficient=complex(1.0, 0.0),
            conjugate_coefficient=complex(0.0, 1.0),
            translation=complex(0.5, -0.5),
        )
        attr.verify()
        assert isinstance(attr.linear_coefficient, ComplexData)
        assert isinstance(attr.conjugate_coefficient, ComplexData)
        assert isinstance(attr.translation, ComplexData)

    def test_instantiation_directly_from_complex_data_type_raises_no_error(self):
        """Tests validity when instantiated from complex data types directly."""

        attr = EqualiseAttr(
            linear_coefficient=ComplexData(complex(1.0, 0.0)),
            conjugate_coefficient=ComplexData(complex(0.0, 1.0)),
            translation=ComplexData(complex(0.5, -0.5)),
        )
        attr.verify()
        assert isinstance(attr.linear_coefficient, ComplexData)
        assert isinstance(attr.conjugate_coefficient, ComplexData)
        assert isinstance(attr.translation, ComplexData)

    def test_instantiation_from_floats_enforces_complex_types(self):
        """If float values are provided, tests the attribute enforces they are type casted
        into complex numbers."""

        attr = EqualiseAttr(
            linear_coefficient=1.0,
            conjugate_coefficient=2.0,
            translation=3.0,
        )
        attr.verify()
        assert isinstance(attr.linear_coefficient, ComplexData)
        assert isinstance(attr.conjugate_coefficient, ComplexData)
        assert isinstance(attr.translation, ComplexData)

        assert isinstance(attr.linear_coefficient.data, complex)
        assert isinstance(attr.conjugate_coefficient.data, complex)
        assert isinstance(attr.translation.data, complex)

    def test_linear_matrix_property_returns_expected_results(self):
        """Tests by multiplying the matrix by an IQ point and checking the results are
        consistent with standard complex multiplication."""

        linear = 4.64 - 2.5j
        conjugate = -3.21 + 0.454j
        complex_number = -9.87 - 5.67j

        attr = EqualiseAttr(linear, conjugate, 0.0)
        matrix = attr.linear_matrix
        assert isinstance(matrix, np.ndarray)
        vec_result = matrix @ np.asarray([complex_number.real, complex_number.imag])

        complex_result = linear * complex_number + conjugate * complex_number.conjugate()
        assert isclose(complex_result.real, vec_result[0])
        assert isclose(complex_result.imag, vec_result[1])

    def test_translation_vector_returns_expected_result(self):
        """Tests the translation complex number is vectorised correctly."""

        translation = 1.0 - 2.0j
        attr = EqualiseAttr(0.0, 0.0, translation)
        vector = attr.translation_vector
        assert isinstance(vector, np.ndarray)
        assert isclose(vector[0], translation.real)
        assert isclose(vector[1], translation.imag)

    def test_creation_from_real_space_yields_correct_complex_numbers(self):
        """Tests that EqualiseAttr.from_real_space yields expected complex coefficients.

        The test verifies both representations produce identical transformed IQ values.
        """

        matrix = np.asarray([[0.564, -1.23], [0.987, 0.654]])
        translation = np.asarray([1.67, -0.02])

        attr = EqualiseAttr.from_real_space(matrix, translation)
        assert isinstance(attr.linear_coefficient, ComplexData)
        assert isinstance(attr.conjugate_coefficient, ComplexData)
        assert isinstance(attr.translation, ComplexData)

        iq_complex = 0.254 - 0.454j
        iq_vector = np.asarray([iq_complex.real, iq_complex.imag])

        transformed_vector = attr.linear_matrix @ iq_vector + attr.translation_vector
        transformed_complex = (
            attr.linear_coefficient.data * iq_complex
            + attr.conjugate_coefficient.data * iq_complex.conjugate()
            + attr.translation.data
        )
        assert isclose(transformed_vector[0], transformed_complex.real)
        assert isclose(transformed_vector[1], transformed_complex.imag)

    def test_roundtrip_creation_yields_equivalent_attributes(self):
        """Tests that creating an EqualiseAttr from real space and converting back yields
        the same results."""

        linear = 4.64 - 2.5j
        conjugate = -3.21 + 0.454j
        translation = 1.0 - 2.0j

        attr = EqualiseAttr(linear, conjugate, translation)
        matrix = attr.linear_matrix
        vector = attr.translation_vector

        new_attr = EqualiseAttr.from_real_space(matrix, vector)
        assert isclose(new_attr.linear_coefficient.data.real, linear.real)
        assert isclose(new_attr.linear_coefficient.data.imag, linear.imag)
        assert isclose(new_attr.conjugate_coefficient.data.real, conjugate.real)
        assert isclose(new_attr.conjugate_coefficient.data.imag, conjugate.imag)
        assert isclose(new_attr.translation.data.real, translation.real)
        assert isclose(new_attr.translation.data.imag, translation.imag)


class TestStateMapDictAttr:
    """This attribute allows mappings from integer state keys to generic attributes, and
    these tests check it works with expected use cases."""

    def test_instantiation_with_integer_keys_and_integer_attributes_is_valid(self):
        """Tests that the attribute can be instantiated with integer keys and integer
        attributes."""

        attr = StateMapDictAttr({0: IntAttr(1), 1: IntAttr(2), 2: IntAttr(3)})
        attr.verify()
        assert isinstance(attr.data, Mapping)
        for key, value in attr.data.items():
            assert isinstance(key, int)
            assert isinstance(value, IntAttr)

    def test_attempt_to_mutate_data_raises_exception(self):
        """Tests that the attribute is immutable and cannot be mutated after creation."""

        attr = StateMapDictAttr({0: IntAttr(1), 1: IntAttr(2)})
        with pytest.raises(TypeError):
            attr.data[0] = IntAttr(3)

    def test_print_and_parse_roundtrip_yields_same_attribute(self, io_stream):
        """Tests that printing and parsing a StateMapDictAttr yields the same attribute."""

        attr = StateMapDictAttr({0: IntAttr(1), 1: IntAttr(2)})
        printer = Printer(stream=io_stream)
        printer.print_attribute(attr)
        output = io_stream.getvalue()

        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, output)
        parsed_attr = parser.parse_attribute()
        assert parsed_attr == attr

    def test_parse_with_non_integer_value_raises_parse_error(self):
        """Tests that parsing rejects non-integer attribute values in the state map."""

        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, '#pulse.state_map_dict{"0" = "bad"}')

        with pytest.raises(
            ParseError,
            match="Expected state map values to be builtin integer attributes.",
        ):
            parser.parse_attribute()


class TestRealThresholdPolicyAttr:
    """Tests the RealThresholdPolicyAttr attribute, which is used to define a threshold
    policy for real-valued measurements."""

    def test_instantiation_with_valid_parameters(self):
        """Tests that the attribute can be instantiated with valid parameters."""

        attr = RealThresholdPolicyAttr(threshold=0.5)
        attr.verify()
        assert isinstance(attr.threshold, FloatData)

    def test_class_has_policy_name_class_var(self):
        """This class variable is used to hook in the implementation, so we want to contract
        test that."""

        assert hasattr(RealThresholdPolicyAttr, "POLICY_NAME")
        assert RealThresholdPolicyAttr.POLICY_NAME == "real_threshold"
        attr = RealThresholdPolicyAttr(threshold=0.5)
        assert hasattr(attr, "POLICY_NAME")
        assert attr.POLICY_NAME == "real_threshold"

    def test_discriminated_states_are_binary(self):
        """Tests that the discriminated states are binary, as expected for a real threshold
        policy."""

        attr = RealThresholdPolicyAttr(threshold=0.5)
        assert attr.state_range == (0, 1)


class TestMaximumLikelihoodPolicyAttr:
    """Tests the MaximumLikelihoodPolicyAttr attribute, which is used to define a maximum
    likelihood policy for measurements."""

    def test_instantiation_with_valid_centers_and_default_args(self):
        """Tests that the attribute can be instantiated with valid parameters."""

        attr = MaximumLikelihoodPolicyAttr(state_centers=(0.0 + 0.5j, 1.0 + 0.0j))
        attr.verify()

        centers = attr.state_centers.data
        assert len(centers) == 2
        assert all(isinstance(center, ComplexData) for center in centers)
        assert isinstance(attr.noise_estimate, FloatData)
        assert attr.noise_estimate.data == 1.0
        assert isinstance(attr.p_min, FloatData)
        assert attr.p_min.data == 0.0
        assert attr.state_range == (-1, 1)

    def test_instantiation_with_valid_centers_and_custom_args(self):
        """Tests that the attribute can be instantiated with valid parameters and custom
        arguments."""

        attr = MaximumLikelihoodPolicyAttr(
            state_centers=(0.0 + 0.5j, 1.0 + 0.0j, -1.0 - 0.5j),
            noise_estimate=0.1,
            p_min=0.05,
        )
        attr.verify()

        centers = attr.state_centers.data
        assert len(centers) == 3
        assert all(isinstance(center, ComplexData) for center in centers)
        assert isinstance(attr.noise_estimate, FloatData)
        assert attr.noise_estimate.data == 0.1
        assert isinstance(attr.p_min, FloatData)
        assert attr.p_min.data == 0.05
        assert attr.state_range == (-1, 2)

    def test_verify_raises_verification_error_for_empty_state_centers(self):
        """Tests that the attribute rejects an empty set of state centers."""

        with pytest.raises(VerifyException, match="state_centers must contain"):
            MaximumLikelihoodPolicyAttr(state_centers=()).verify()

    @pytest.mark.parametrize("p_min", [-0.1, 1.1])
    def test_verify_raises_verification_error_for_invalid_p_min(self, p_min):
        """Tests that the attribute raises a verification error when p_min is out of the
        range of [0, 1]."""

        with pytest.raises(VerifyException, match="p_min must be in the range"):
            MaximumLikelihoodPolicyAttr(
                state_centers=(0.0 + 0.5j, 1.0 + 0.0j), noise_estimate=0.1, p_min=p_min
            ).verify()
