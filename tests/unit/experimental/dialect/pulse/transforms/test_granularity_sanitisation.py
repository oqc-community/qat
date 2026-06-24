# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import BoolAttr, StringAttr, i1
from xdsl.dialects.scf import IfOp
from xdsl.ir import Block, Region

from qat.experimental.dialect.pulse.ir.attributes import (
    FrequencyAttr,
    SampledWaveformAttr,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.ops import ConstantOp, CreateFrameOp, PulseOp
from qat.experimental.dialect.pulse.ir.types import TimeType, WaveformType
from qat.experimental.dialect.pulse.transforms.granularity_sanitisation import (
    ApplyGranularitySanitisation,
)
from qat.experimental.dialect.pulse.units import TimeUnits

from tests.unit.utils.ir import build_module_from_ops


def create_drive_pulse(target: ConstantOp) -> tuple[CreateFrameOp, PulseOp]:
    """Creates a drive frame and pulse for the given pulse target.

    :param target: The pulse duration or waveform constant to pulse.
    :returns: A frame and pulse operation using the provided target.
    """

    frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
    pulse = PulseOp(frame, target)
    return frame, pulse


class TestGranularitySanitisation:
    """Tests for the :class:`GranularitySanitisation` pass."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_waveform_constant_sanitisation(self, seed: int) -> None:
        rng = np.random.default_rng(seed)

        granularity = rng.integers(2, 10) * 1e-9
        supersampling = rng.integers(3, 10)
        sampled_time = granularity / supersampling

        num_samples = rng.integers(1, 100) * supersampling
        extra_samples = rng.integers(1, supersampling - 1)
        waveform_array = np.ones(num_samples + extra_samples, dtype=complex)
        time_width = len(waveform_array) * sampled_time

        const_waveform = ConstantOp(
            SampledWaveformAttr(
                waveform_array,
                TimeAttr(time_width),
                TimeAttr(sampled_time),
            ),
            WaveformType(),
        )

        assert (const_waveform.fold()[0].literal_value == waveform_array).all()
        assert len(const_waveform.fold()[0].literal_value) == num_samples + extra_samples

        frame, pulse = create_drive_pulse(const_waveform)

        module = build_module_from_ops([const_waveform, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == WaveformType()
        ]

        assert len(constant_ops) == 1
        assert np.isclose(
            constant_ops[0].fold()[0].width.literal_value,
            np.ceil(time_width / granularity) * granularity,
        )
        assert len(constant_ops[0].fold()[0].literal_value) == (num_samples + supersampling)

    def test_valid_time_constant_sanitisation_no_change_with_units(self) -> None:
        granularity = 2e-9
        time_val_ns = 16
        time_val = time_val_ns * 1e-9
        time = TimeAttr(time_val_ns, TimeUnits.NANOSECOND)
        const_time = ConstantOp(time, TimeType())

        frame, pulse = create_drive_pulse(const_time)
        module = build_module_from_ops([const_time, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        _, new_module = pass_instance.apply_to_clone(Context(), module)

        assert module.is_structurally_equivalent(new_module)

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == TimeType()
        ]

        assert len(constant_ops) == 1
        assert np.isclose(constant_ops[0].fold()[0].literal_value, time_val)

    def test_invalid_time_constant_sanitisation_changed_with_units(self) -> None:
        granularity = 2e-9
        time_val_ns = 17
        new_time_val_ns = 16
        new_time_val = new_time_val_ns * 1e-9
        time = TimeAttr(time_val_ns, TimeUnits.NANOSECOND)
        const_time = ConstantOp(time, TimeType())

        frame, pulse = create_drive_pulse(const_time)
        module = build_module_from_ops([const_time, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        _, new_module = pass_instance.apply_to_clone(Context(), module)

        assert not module.is_structurally_equivalent(new_module)

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == TimeType()
        ]

        assert len(constant_ops) == 1
        assert np.isclose(constant_ops[0].fold()[0].literal_value, new_time_val)

    def test_time_constant_sanitisation_preserves_units(self) -> None:
        granularity = TimeAttr(2, TimeUnits.NANOSECOND)
        time = TimeAttr(17, TimeUnits.NANOSECOND)
        const_time = ConstantOp(time, TimeType())

        frame, pulse = create_drive_pulse(const_time)
        module = build_module_from_ops([const_time, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(granularity)
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == TimeType()
        ]

        assert len(constant_ops) == 1

        sanitised_time = constant_ops[0].fold()[0]

        assert sanitised_time.unit.data == TimeUnits.NANOSECOND
        assert sanitised_time.value.data == 18
        assert np.isclose(sanitised_time.literal_value, 18e-9)

    def test_valid_waveform_constant_sanitisation_no_change(self) -> None:
        granularity = 2e-9
        sampled_time = 1e-9
        time_width = 16e-9
        waveform_array = np.ones(16, dtype=complex)

        const_waveform = ConstantOp(
            SampledWaveformAttr(
                waveform_array,
                TimeAttr(time_width),
                TimeAttr(sampled_time),
            ),
            WaveformType(),
        )

        frame, pulse = create_drive_pulse(const_waveform)
        module = build_module_from_ops([const_waveform, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        _, new_module = pass_instance.apply_to_clone(Context(), module)

        assert module.is_structurally_equivalent(new_module)

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == WaveformType()
        ]

        assert len(constant_ops) == 1

        sanitised_waveform = constant_ops[0].fold()[0]

        assert np.isclose(sanitised_waveform.width.literal_value, time_width)
        assert np.isclose(sanitised_waveform.sample_time.literal_value, sampled_time)
        np.testing.assert_array_equal(sanitised_waveform.literal_value, waveform_array)

    def test_waveform_constant_sanitisation_preserves_units(self) -> None:
        granularity = TimeAttr(8, TimeUnits.NANOSECOND)
        sampled_time = TimeAttr(1, TimeUnits.NANOSECOND)
        time_width = TimeAttr(17, TimeUnits.NANOSECOND)
        waveform_array = np.ones(17, dtype=complex)

        const_waveform = ConstantOp(
            SampledWaveformAttr(
                waveform_array,
                time_width,
                sampled_time,
            ),
            WaveformType(),
        )

        frame, pulse = create_drive_pulse(const_waveform)
        module = build_module_from_ops([const_waveform, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(granularity)
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == WaveformType()
        ]

        assert len(constant_ops) == 1

        sanitised_waveform = constant_ops[0].fold()[0]

        assert sanitised_waveform.width.unit.data == TimeUnits.NANOSECOND
        assert sanitised_waveform.width.value.data == 24
        assert np.isclose(sanitised_waveform.width.literal_value, 24e-9)
        assert sanitised_waveform.sample_time.unit.data == TimeUnits.NANOSECOND
        assert sanitised_waveform.sample_time.value.data == 1
        assert np.isclose(sanitised_waveform.sample_time.literal_value, 1e-9)
        assert len(sanitised_waveform.literal_value) == 24

    def test_waveform_constant_sanitisation_skips_non_integral_sample_count(
        self,
    ) -> None:
        granularity = TimeAttr(8, TimeUnits.NANOSECOND)
        sampled_time = TimeAttr(2, TimeUnits.NANOSECOND)
        time_width = TimeAttr(17, TimeUnits.NANOSECOND)
        waveform_array = np.ones(17, dtype=complex)

        const_waveform = ConstantOp(
            SampledWaveformAttr(
                waveform_array,
                time_width,
                sampled_time,
            ),
            WaveformType(),
        )

        frame, pulse = create_drive_pulse(const_waveform)
        module = build_module_from_ops([const_waveform, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(granularity)
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == WaveformType()
        ]

        assert len(constant_ops) == 1

        sanitised_waveform = constant_ops[0].fold()[0]

        assert sanitised_waveform.width.unit.data == TimeUnits.NANOSECOND
        assert sanitised_waveform.width.value.data == 17
        assert np.isclose(sanitised_waveform.width.literal_value, 17e-9)
        assert sanitised_waveform.sample_time.unit.data == TimeUnits.NANOSECOND
        assert sanitised_waveform.sample_time.value.data == 2
        assert np.isclose(sanitised_waveform.sample_time.literal_value, 2e-9)
        np.testing.assert_array_equal(sanitised_waveform.literal_value, waveform_array)

    def test_time_constant_sanitisation_in_nested_region(self) -> None:
        granularity = TimeAttr(2, TimeUnits.NANOSECOND)
        condition = ArithConstantOp(BoolAttr(False, value_type=1), i1)
        const_time = ConstantOp(TimeAttr(17, TimeUnits.NANOSECOND), TimeType())
        frame, pulse = create_drive_pulse(const_time)
        true_block = Block([const_time, frame, pulse])
        true_region = Region(true_block)
        if_op = IfOp(condition, [], true_region)
        module = build_module_from_ops([condition, if_op])

        pass_instance = ApplyGranularitySanitisation(granularity)
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == TimeType()
        ]

        assert len(constant_ops) == 1

        sanitised_time = constant_ops[0].fold()[0]

        assert sanitised_time.unit.data == TimeUnits.NANOSECOND
        assert sanitised_time.value.data == 18
        assert np.isclose(sanitised_time.literal_value, 18e-9)

    @pytest.mark.parametrize("time_val_ns", [3, 5, 15, 17, 99])
    def test_time_constant_sanitisation_modified_time_data_upon_invalid_time(
        self, time_val_ns: int
    ) -> None:
        granularity = 2e-9
        time_val = time_val_ns * 1e-9
        time = TimeAttr(time_val)
        const_time = ConstantOp(time, TimeType())

        frame, pulse = create_drive_pulse(const_time)
        module = build_module_from_ops([const_time, frame, pulse])

        pass_instance = ApplyGranularitySanitisation(TimeAttr(granularity))
        pass_instance.apply(Context(), module)

        constant_ops = [
            node
            for node in module.walk()
            if isinstance(node, ConstantOp) and node.result.type == TimeType()
        ]

        assert len(constant_ops) == 1
        assert constant_ops[0].result.type == TimeType()

        sanitised_time = constant_ops[0].fold()[0]
        expected_time = np.ceil(time_val / granularity) * granularity

        assert np.isclose(sanitised_time.literal_value, expected_time)
        assert sanitised_time.literal_value > time_val

    @pytest.mark.parametrize("granularity", [0.0, -1e-9])
    def test_invalid_granularity_raises(self, granularity: float) -> None:
        with pytest.raises(ValueError, match="granularity must be greater than zero"):
            ApplyGranularitySanitisation(TimeAttr(granularity))
