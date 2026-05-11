# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import ComplexType, FloatAttr, IntegerAttr, ModuleOp, f64, i64
from xdsl.dialects.complex import ComplexNumberAttr, ConstantOp as ComplexConstantOp
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import ConstantLike, Pure
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    AddOp,
    AmplitudeAttr,
    AmplitudeType,
    BinaryOp,
    ConstantOp,
    FrequencyAttr,
    FrequencyType,
    ModulateOp,
    ModuloOp,
    PhaseAttr,
    PhaseType,
    PulseTypesCanonicalizationPatternsTrait,
    SampledWaveformAttr,
    ScaleOp,
    SubOp,
    TimeAttr,
    TimeType,
    WaveformType,
)
from qat.experimental.dialect.pulse.units import FrequencyUnits, TimeUnits


@irdl_op_definition
class _DummyOp(IRDLOperation):
    """Serves as an operation that "uses" an operand so we can test canonicalization without
    it deleting it."""

    name = "dummy"

    arg = operand_def(ParametrizedAttribute)

    def __init__(self, arg):
        super().__init__(operands=[arg])


_SAMPLE_TIME_ATTR = TimeAttr(0.5, TimeUnits.NANOSECOND)
_WIDTH_ATTR = TimeAttr(80, TimeUnits.NANOSECOND)


class TestConstantFoldingOnOps:
    """Tests cases where constant folding should be applied."""

    @pytest.mark.parametrize(
        "lhs, rhs, expected, type_",
        [
            (PhaseAttr(1.23), PhaseAttr(2.54), PhaseAttr(3.77), PhaseType()),
            (
                AmplitudeAttr(2.54),
                AmplitudeAttr(4.54),
                AmplitudeAttr(7.08),
                AmplitudeType(),
            ),
            (
                AmplitudeAttr(2.54 + 3.14j),
                AmplitudeAttr(4.54 + 1.59j),
                AmplitudeAttr(7.08 + 4.73j),
                AmplitudeType(),
            ),
            (
                TimeAttr(400, TimeUnits.NANOSECOND),
                TimeAttr(200, TimeUnits.NANOSECOND),
                TimeAttr(600, TimeUnits.NANOSECOND),
                TimeType(),
            ),
            (
                TimeAttr(200, TimeUnits.NANOSECOND),
                TimeAttr(0.4, TimeUnits.MICROSECOND),
                TimeAttr(600.0, TimeUnits.NANOSECOND),
                TimeType(),
            ),
            (
                FrequencyAttr(5.5, FrequencyUnits.GIGAHERTZ),
                FrequencyAttr(0.1, FrequencyUnits.GIGAHERTZ),
                FrequencyAttr(5.6, FrequencyUnits.GIGAHERTZ),
                FrequencyType(),
            ),
            (
                FrequencyAttr(5.5, FrequencyUnits.GIGAHERTZ),
                FrequencyAttr(100, FrequencyUnits.MEGAHERTZ),
                FrequencyAttr(5600.0, FrequencyUnits.MEGAHERTZ),
                FrequencyType(),
            ),
            (
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(2 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(3 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                WaveformType(),
            ),
        ],
    )
    def test_constant_folding_on_add(self, lhs, rhs, expected, type_):
        constant1 = ConstantOp(lhs)
        constant2 = ConstantOp(rhs)
        add_op = AddOp(constant1, constant2, type_)
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 2
        op = module.body.ops.first
        assert isinstance(op, ConstantOp)
        assert op.value == expected
        assert op.result_types[0] == type_

    @pytest.mark.parametrize(
        "lhs, rhs, expected, type_",
        [
            (PhaseAttr(1.23), PhaseAttr(2.54), PhaseAttr(-1.31), PhaseType()),
            (
                AmplitudeAttr(2.54),
                AmplitudeAttr(4.54),
                AmplitudeAttr(-2.0),
                AmplitudeType(),
            ),
            (
                AmplitudeAttr(4.54 + 1.59j),
                AmplitudeAttr(2.54 + 3.14j),
                AmplitudeAttr(2.0 - 1.55j),
                AmplitudeType(),
            ),
            (
                TimeAttr(500, TimeUnits.NANOSECOND),
                TimeAttr(200, TimeUnits.NANOSECOND),
                TimeAttr(300, TimeUnits.NANOSECOND),
                TimeType(),
            ),
            (
                TimeAttr(0.5, TimeUnits.MICROSECOND),
                TimeAttr(200, TimeUnits.NANOSECOND),
                TimeAttr(300.0, TimeUnits.NANOSECOND),
                TimeType(),
            ),
            (
                FrequencyAttr(5.5, FrequencyUnits.GIGAHERTZ),
                FrequencyAttr(0.1, FrequencyUnits.GIGAHERTZ),
                FrequencyAttr(5.4, FrequencyUnits.GIGAHERTZ),
                FrequencyType(),
            ),
            (
                FrequencyAttr(5.5, FrequencyUnits.GIGAHERTZ),
                FrequencyAttr(100, FrequencyUnits.MEGAHERTZ),
                FrequencyAttr(5400.0, FrequencyUnits.MEGAHERTZ),
                FrequencyType(),
            ),
            (
                SampledWaveformAttr(3 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(2 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                WaveformType(),
            ),
        ],
    )
    def test_constant_folding_on_sub(self, lhs, rhs, expected, type_):
        constant1 = ConstantOp(lhs)
        constant2 = ConstantOp(rhs)
        sub_op = SubOp(constant1, constant2, type_)
        dummy_op = _DummyOp(sub_op.result)
        module = ModuleOp(ops=[constant1, constant2, sub_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 2
        op = module.body.ops.first
        assert isinstance(op, ConstantOp)
        assert op.value == expected
        assert op.result_types[0] == type_

    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (PhaseAttr(2.54), PhaseAttr(np.pi), PhaseAttr(2.54)),
            (PhaseAttr(1.1 + np.pi), PhaseAttr(np.pi), PhaseAttr(1.1)),
            (PhaseAttr(-1.1), PhaseAttr(np.pi), PhaseAttr(-1.1 + np.pi)),
        ],
    )
    def test_constant_folding_on_modulo(self, lhs, rhs, expected):
        constant1 = ConstantOp(lhs)
        constant2 = ConstantOp(rhs)
        modulo_op = ModuloOp(constant1, constant2, PhaseType())
        dummy_op = _DummyOp(modulo_op.result)
        module = ModuleOp(ops=[constant1, constant2, modulo_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 2
        op = module.body.ops.first
        assert isinstance(op, ConstantOp)
        assert np.isclose(op.value.literal_value, expected.literal_value)
        assert op.result_types[0] == PhaseType()

    @pytest.mark.parametrize(
        "lhs, rhs, expected, type_",
        [
            (FloatAttr(2.0, f64), PhaseAttr(1.23), PhaseAttr(2.46), PhaseType()),
            (IntegerAttr(2, i64), PhaseAttr(1.23), PhaseAttr(2.46), PhaseType()),
            (
                FloatAttr(2.0, f64),
                AmplitudeAttr(2.54),
                AmplitudeAttr(5.08),
                AmplitudeType(),
            ),
            (
                IntegerAttr(2, i64),
                AmplitudeAttr(2.54 + 1.1j),
                AmplitudeAttr(5.08 + 2.2j),
                AmplitudeType(),
            ),
            (
                ComplexNumberAttr(2.0, 3.0, ComplexType(f64)),
                AmplitudeAttr(2.54 + 1.1j),
                AmplitudeAttr((2.0 + 3.0j) * (2.54 + 1.1j)),
                AmplitudeType(),
            ),
            (
                FloatAttr(2.0, f64),
                TimeAttr(2, TimeUnits.MILLISECOND),
                TimeAttr(4.0, TimeUnits.MILLISECOND),
                TimeType(),
            ),
            (IntegerAttr(3, i64), TimeAttr(2), TimeAttr(6), TimeType()),
            (
                FloatAttr(2.0, f64),
                FrequencyAttr(5.5, FrequencyUnits.KILOHERTZ),
                FrequencyAttr(11.0, FrequencyUnits.KILOHERTZ),
                FrequencyType(),
            ),
            (IntegerAttr(3, i64), FrequencyAttr(4), FrequencyAttr(12), FrequencyType()),
            (
                FloatAttr(2.0, f64),
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(2 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                WaveformType(),
            ),
            (
                ComplexNumberAttr(2.0, 3.0, ComplexType(f64)),
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(
                    (2 + 3j) * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR
                ),
                WaveformType(),
            ),
        ],
    )
    def test_constant_folding_on_scale(self, lhs, rhs, expected, type_):
        if isinstance(lhs, ComplexNumberAttr):
            constant1 = ComplexConstantOp(lhs, ComplexType(f64))
        elif isinstance(lhs, FloatAttr):
            constant1 = ArithConstantOp(lhs, f64)
        else:
            constant1 = ArithConstantOp(lhs, i64)

        constant2 = ConstantOp(rhs)
        scale_op = ScaleOp(constant1, constant2, type_)
        dummy_op = _DummyOp(scale_op.result)
        module = ModuleOp(ops=[constant1, constant2, scale_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 2
        op = module.body.ops.first
        assert isinstance(op, ConstantOp)
        assert op.value == expected
        assert op.result_types[0] == type_

    def test_constant_folding_on_modulate(self):
        lhs_waveform = SampledWaveformAttr(
            np.linspace(0.0, 1.0, 160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR
        )
        rhs_waveform = SampledWaveformAttr(2 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR)
        expected_waveform = SampledWaveformAttr(
            np.linspace(0.0, 2.0, 160),
            _WIDTH_ATTR,
            _SAMPLE_TIME_ATTR,
        )

        constant1 = ConstantOp(lhs_waveform)
        constant2 = ConstantOp(rhs_waveform)

        modulate_op = ModulateOp(constant1, constant2)
        dummy_op = _DummyOp(modulate_op.result)
        module = ModuleOp(ops=[constant1, constant2, modulate_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 2
        op = module.body.ops.first
        assert isinstance(op, ConstantOp)
        assert np.allclose(op.value.literal_value, expected_waveform.literal_value)
        assert op.result_types[0] == WaveformType()


class TestConstantFoldingIsUnsuccessful:
    @irdl_op_definition
    class _ConstantOpThatIsNotDeclaredAsConstantLike(IRDLOperation):
        """A constant op that is not declared as constant-like by the canonicalizer, so we
        can test that folding is correctly skipped when an operand is not marked as
        constant-like."""

        name = "mock.non_constant_like_constant"
        value = prop_def()
        result = result_def()

        def __init__(self, value, result_type):
            super().__init__(result_types=[result_type], properties={"value": value})

    @irdl_op_definition
    class _ConstantOpThatIsDeclaredAsConstantLike(IRDLOperation, HasFolderInterface):
        """A constant op that is declared as constant-like by the canonicalizer, so we can
        test that folding is correctly applied when an operand is marked as constant-
        like."""

        name = "mock.constant_like_constant"
        value = prop_def()
        result = result_def()
        traits = traits_def(ConstantLike())

        def __init__(self, value, result_type):
            super().__init__(result_types=[result_type], properties={"value": value})

        def fold(self):
            return tuple()

    @irdl_op_definition
    class _ConstantOp(IRDLOperation, HasFolderInterface):
        """A constant op that is declared as constant-like by the canonicalizer, and
        implements folding to return a value, so we can test that folding is correctly
        applied when an operand is marked as constant-like and returns a value from
        fold()."""

        name = "mock.constant_like_constant_that_folds"
        value = prop_def()
        result = result_def()
        traits = traits_def(ConstantLike(), Pure())

        def __init__(self, value, result_type):
            super().__init__(result_types=[result_type], properties={"value": value})

        def fold(self):
            return (self.value,)

    @irdl_attr_definition
    class _MockAttribute(ParametrizedAttribute):
        """A mock attribute that we can use for testing folding with an operand that is
        constant-like but not one of the pulse attributes."""

        name = "mock.attribute"

    @irdl_op_definition
    class _MockBinaryOp(BinaryOp):
        """A mock binary operation that we can use for testing folding with an operand that
        is constant-like but not one of the pulse attributes."""

        name = "mock.binary_op"
        lhs = operand_def(ParametrizedAttribute)
        rhs = operand_def(ParametrizedAttribute)
        result = result_def()
        traits = traits_def(Pure(), PulseTypesCanonicalizationPatternsTrait())

        def __init__(self, lhs, rhs, result_type):
            super().__init__(operands=[lhs, rhs], result_types=[result_type])

    @pytest.mark.parametrize(
        "attr1, attr2, type_",
        [
            (PhaseAttr(1.23), PhaseAttr(2.54), PhaseType()),
            (
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(2 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                WaveformType(),
            ),
        ],
    )
    def test_folding_ignored_when_operand_not_marked_as_constant_like(
        self, attr1, attr2, type_
    ):
        """Tests that the pass does not do canonicalization for something not marked as
        constant-like."""
        constant1 = self._ConstantOpThatIsNotDeclaredAsConstantLike(attr1, type_)
        constant2 = ConstantOp(attr2)
        add_op = AddOp(constant1, constant2, type_)
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 4
        assert module.body.ops.first is constant1
        assert module.body.ops.first.next_op is constant2
        assert module.body.ops.first.next_op.next_op is add_op

    @pytest.mark.parametrize("side", ["lhs", "rhs"])
    @pytest.mark.parametrize(
        "lhs, rhs, type_",
        [
            (PhaseAttr(1.23), PhaseAttr(2.54), PhaseType()),
            (
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                SampledWaveformAttr(2 * np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                WaveformType(),
            ),
        ],
    )
    def test_folding_ignored_when_operand_marked_as_constant_like_but_not(
        self, side, lhs, rhs, type_
    ):
        """Tests that the pass does not apply any canonicalization if folding is not
        implemented."""
        if side == "lhs":
            constant1 = self._ConstantOpThatIsDeclaredAsConstantLike(lhs, type_)
            constant2 = ConstantOp(rhs)
        else:
            constant1 = ConstantOp(lhs)
            constant2 = self._ConstantOpThatIsDeclaredAsConstantLike(rhs, type_)
        add_op = AddOp(constant1, constant2, type_)
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 4
        assert module.body.ops.first is constant1
        assert module.body.ops.first.next_op is constant2
        assert module.body.ops.first.next_op.next_op is add_op

    def test_folding_with_supported_attribute_type(self):
        """Tests that the pass applies canonicalization when an operand is marked as
        constant-like and returns a value from fold() that is one of the expected types."""

        constant1 = self._ConstantOp(PhaseAttr(1.23), PhaseType())
        constant2 = ConstantOp(PhaseAttr(2.54))
        add_op = AddOp(constant1, constant2, PhaseType())
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 2
        op = module.body.ops.first
        assert isinstance(op, ConstantOp)
        assert op.value == PhaseAttr(3.77)
        assert op.result_types[0] == PhaseType()

    @pytest.mark.parametrize(
        "pulse_attribute, type_",
        [
            (PhaseAttr(2.54), PhaseType()),
            (
                SampledWaveformAttr(np.ones(160), _WIDTH_ATTR, _SAMPLE_TIME_ATTR),
                WaveformType(),
            ),
        ],
    )
    def test_folding_with_unsupported_attribute_type(self, pulse_attribute, type_):
        """Tests that the pass does not apply any canonicalization if the constant-like
        operand returns a value from fold() but it is not one of the expected attribute
        types that the folding logic can handle."""
        constant1 = self._ConstantOp(self._MockAttribute(), type_)
        constant2 = ConstantOp(pulse_attribute)
        add_op = AddOp(constant1, constant2, type_)
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 4
        assert module.body.ops.first is constant1
        assert module.body.ops.first.next_op is constant2
        assert module.body.ops.first.next_op.next_op is add_op

    def test_folding_waveforms_with_different_widths_raises_pass_failed_error(self):
        """Tests that if we try to fold two SampledWaveformAttrs with different widths, the
        pass raises a PassFailedException."""
        constant1 = ConstantOp(
            SampledWaveformAttr(
                np.ones(160), TimeAttr(80, TimeUnits.NANOSECOND), _SAMPLE_TIME_ATTR
            )
        )
        constant2 = ConstantOp(
            SampledWaveformAttr(
                2 * np.ones(180), TimeAttr(90, TimeUnits.NANOSECOND), _SAMPLE_TIME_ATTR
            )
        )
        add_op = AddOp(constant1, constant2, WaveformType())
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        with pytest.raises(
            PassFailedException,
            match="Cannot fold two SampledWaveformAttrs with different widths.",
        ):
            CanonicalizePass().apply(Context(), module)

    def test_folding_waveforms_with_different_sample_times_is_ignored(self):
        """Waveforms with the same width but different sample times has no current defined
        optimization path."""

        constant1 = ConstantOp(
            SampledWaveformAttr(
                np.ones(160),
                TimeAttr(80, TimeUnits.NANOSECOND),
                TimeAttr(0.5, TimeUnits.NANOSECOND),
            )
        )
        constant2 = ConstantOp(
            SampledWaveformAttr(
                2 * np.ones(80),
                TimeAttr(80, TimeUnits.NANOSECOND),
                TimeAttr(1.0, TimeUnits.NANOSECOND),
            )
        )
        add_op = AddOp(constant1, constant2, WaveformType())
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 4
        assert module.body.ops.first is constant1
        assert module.body.ops.first.next_op is constant2
        assert module.body.ops.first.next_op.next_op is add_op

    def test_folding_with_unsupported_result_type_is_passed(self):
        """Tests that if we try to fold with a result type that is not one of the expected
        types, the pass does not apply any canonicalization but also does not raise an
        error."""
        constant1 = self._ConstantOp(FloatAttr(2.0, f64), f64)
        constant2 = self._ConstantOp(FloatAttr(3.0, f64), f64)
        add_op = self._MockBinaryOp(constant1.result, constant2.result, f64)
        dummy_op = _DummyOp(add_op.result)
        module = ModuleOp(ops=[constant1, constant2, add_op, dummy_op])
        assert len(module.body.ops) == 4

        CanonicalizePass().apply(Context(), module)
        assert len(module.body.ops) == 4
        assert module.body.ops.first is constant1
        assert module.body.ops.first.next_op is constant2
        assert module.body.ops.first.next_op.next_op is add_op
