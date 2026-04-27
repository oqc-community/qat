# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import re
from copy import deepcopy

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.complex import Complex
from xdsl.parser import Parser
from xdsl.printer import Printer

from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    AmplitudeType,
    FrequencyAttr,
    FrequencyType,
    PhaseAttr,
    PhaseType,
    Pulse,
    SampledWaveformAttr,
    TimeAttr,
    TimeType,
    WaveformType,
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
            ([1.0 + 1.0j, 0.5 - 0.5j], 2e-6, 1e6),
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
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(4e9), TimeAttr(1e9))
        assert attr1 != attr2

    def test_inequality_with_different_sample_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(2e9))
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
        pattern = r"#pulse\.sampled_waveform<\s*#pulse\.waveform_data\[(.*?)\],\s*#pulse\.time_attr<(.*?)>,\s*#pulse\.time_attr<(.*?)>\s*>"
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
            "#pulse.sampled_waveform<#pulse.waveform_data[#builtin.float_data<0.0>, "
            "#builtin.float_data<1.0>, #builtin.float_data<0.5>], "
            '#pulse.time_attr<#builtin.float_data<3.0>, #pulse.time_units"ns">, '
            '#pulse.time_attr<#builtin.float_data<1.0>, #pulse.time_units"ns">>'
        )
        context = Context()
        context.load_dialect(Pulse)
        context.load_dialect(Builtin)
        parser = Parser(context, attr_str)
        with pytest.raises(ValueError, match="samples to be builtin complex attributes."):
            parser.parse_attribute()

    def test_hash_equality(self):
        waveform = [0.0, 1.0, 0.5, -1 / 3]
        attr1 = SampledWaveformAttr(waveform, TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr(deepcopy(waveform), TimeAttr(3e9), TimeAttr(1e9))
        assert hash(attr1) == hash(attr2)

    def test_hash_inequality_with_different_waveform(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.6], TimeAttr(3e9), TimeAttr(1e9))
        assert hash(attr1) != hash(attr2)

    def test_hash_inequality_with_different_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(4e9), TimeAttr(1e9))
        assert hash(attr1) != hash(attr2)

    def test_hash_inequality_with_different_sample_time(self):
        attr1 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(1e9))
        attr2 = SampledWaveformAttr([0.0, 1.0, 0.5], TimeAttr(3e9), TimeAttr(2e9))
        assert hash(attr1) != hash(attr2)
