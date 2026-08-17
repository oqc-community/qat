# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import (
    BoolAttr,
    ComplexType,
    FlatSymbolRefAttr,
    FloatAttr,
    FunctionType,
    IntAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    f64,
    i64,
)
from xdsl.dialects.complex import ComplexNumberAttr, ConstantOp as ComplexConstantOp
from xdsl.ir import Attribute, Block, Operation, Region
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    param_def,
    result_def,
)
from xdsl.traits import IsolatedFromAbove, SymbolTable
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AcquisitionType,
    AddOp,
    AmplitudeAttr,
    AmplitudeType,
    BlackmanWaveformOp,
    CallKernelOp,
    CallKernelOpUserOpInterface,
    ConstantOp,
    CreateFrameOp,
    DiscriminateOp,
    DiscriminatorPolicyAttr,
    EqualiseAttr,
    EqualiseOp,
    FrameType,
    FrequencyAttr,
    FrequencyType,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    IntegrateOp,
    IQResultType,
    KernelOp,
    MaxTimeOp,
    MixOp,
    ModuloOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PhaseType,
    PulseOp,
    RealThresholdPolicyAttr,
    ReturnOp,
    RoundedSquareWaveformOp,
    SampledWaveformAttr,
    ScaleOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinusoidalWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StateKeyType,
    StateMapOp,
    StopContinuousWaveformOp,
    SubOp,
    SynchronizeOp,
    TimeAttr,
    TimeType,
    WaitOp,
    WaveformType,
    WeightsAttr,
)
from qat.experimental.dialect.pulse.ir.attributes import StateMapDictAttr
from qat.experimental.waveforms.shapes.blackman import BlackmanWaveformShape
from qat.experimental.waveforms.shapes.gaussian import GaussianWaveformShape
from qat.experimental.waveforms.shapes.gaussian_square import GaussianSquareWaveformShape
from qat.experimental.waveforms.shapes.rounded_square import RoundedSquareWaveformShape
from qat.experimental.waveforms.shapes.sech import SechWaveformShape
from qat.experimental.waveforms.shapes.setup_hold import SetupHoldWaveformShape
from qat.experimental.waveforms.shapes.sinusoidal import SinusoidalWaveformShape
from qat.experimental.waveforms.shapes.soft_square import SoftSquareWaveformShape
from qat.experimental.waveforms.shapes.square import SquareWaveformShape


@irdl_op_definition
class _ProducerOp(IRDLOperation):
    name = "test.producer"
    result = result_def()

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])


def _float_constant(value: float):
    return ArithConstantOp(FloatAttr(value, 64), f64).results[0]


def _time_constant(value: float):
    return ConstantOp(TimeAttr(value)).results[0]


def _amplitude_constant(value: float | complex):
    return ConstantOp(AmplitudeAttr(value)).results[0]


def _phase_constant(value: float):
    return ConstantOp(PhaseAttr(value)).results[0]


class TestConstantOp:
    @pytest.mark.parametrize(
        "attr, result",
        [
            (PhaseAttr(np.pi / 2), PhaseType()),
            (FrequencyAttr(5.5e9), FrequencyType()),
            (TimeAttr(160e-9), TimeType()),
            (AmplitudeAttr(0.5 - 0.5j), AmplitudeType()),
            (
                SampledWaveformAttr(
                    np.array([0.0, 0.5, 1.0, 0.5, 0.0]), TimeAttr(5e-9), TimeAttr(1e-9)
                ),
                WaveformType(),
            ),
        ],
    )
    def test_verification_passes(self, attr, result):
        op = ConstantOp(attr)
        assert op.value == attr
        assert op.result.type == result
        op.verify()  # Should not raise an exception

    def test_verification_of_invalid_attr(self):
        op = ConstantOp(FloatAttr(1.0, 32), AmplitudeType())
        with pytest.raises(VerifyException, match="Unexpected attribute"):
            op.verify()

    def test_verification_of_mismatched_attr_and_result_types(self):
        attr = FrequencyAttr(5.5e9)
        op = ConstantOp(FrequencyAttr(5.5e9), PhaseType())
        assert attr.associated_type() != op.result.type
        with pytest.raises(VerifyException, match="Type of value attribute"):
            op.verify()

    def test_verification_of_invalid_result_type(self):
        attr = FrequencyAttr(5.5e9)
        op = ConstantOp(attr, FrameType("default"))
        assert op.result.type == FrameType("default")
        with pytest.raises(VerifyException, match="result 'result' at position 0"):
            op.verify()

    @pytest.mark.parametrize(
        "attr, result",
        [
            (PhaseAttr(np.pi / 2), PhaseType()),
            (FrequencyAttr(5.5e9), FrequencyType()),
            (TimeAttr(160e-9), TimeType()),
            (AmplitudeAttr(0.5 - 0.5j), AmplitudeType()),
        ],
    )
    def test_type(self, attr, result):
        op = ConstantOp(attr)
        assert op.result.type == result

    def test_fold(self):
        attr = PhaseAttr(np.pi / 2)
        op = ConstantOp(attr)
        folded = op.fold()
        assert folded == (attr,)


@pytest.mark.parametrize("op_type", [AddOp, SubOp])
class TestInternalBinaryOps:
    @pytest.mark.parametrize("with_operation", [True, False])
    @pytest.mark.parametrize(
        "operand1, operand2, result_type",
        [
            (PhaseAttr(0.5), PhaseAttr(1.0), PhaseType()),
            (FrequencyAttr(5.5e9), FrequencyAttr(0.1e9), FrequencyType()),
            (TimeAttr(160e-9), TimeAttr(40e-9), TimeType()),
            (AmplitudeAttr(0.5 - 0.5j), AmplitudeAttr(0.25 + 0.25j), AmplitudeType()),
            (
                SampledWaveformAttr(
                    np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9)
                ),
                SampledWaveformAttr(
                    np.array([0.0, 0.25, 0.5]), TimeAttr(3e-9), TimeAttr(1e-9)
                ),
                WaveformType(),
            ),
        ],
    )
    def test_initialization(self, op_type, with_operation, operand1, operand2, result_type):
        """Also demonstrates we can instantiate with an SSA value, or use an operation and
        the result will be selected.

        This won't be tested for every operation.
        """
        constant1 = ConstantOp(operand1)
        constant2 = ConstantOp(operand2)
        if with_operation:
            op = op_type(constant1, constant2, result_type)
        else:
            op = op_type(constant1.results[0], constant2.results[0], result_type)
        op.verify()

    def test_verify_with_different_operand_types(self, op_type):
        constant1 = ConstantOp(PhaseAttr(0.5))
        constant2 = ConstantOp(FrequencyAttr(5.5e9))
        op = op_type(constant1.results[0], constant2.results[0], result_type=PhaseType())
        with pytest.raises(VerifyException, match="Types of lhs and rhs"):
            op.verify()

    def test_verify_with_result_type_mismatch(self, op_type):
        constant1 = ConstantOp(PhaseAttr(0.5))
        constant2 = ConstantOp(PhaseAttr(1.0))
        op = op_type(
            constant1.results[0], constant2.results[0], result_type=FrequencyType()
        )
        with pytest.raises(
            VerifyException, match="Type of result must be the same as type"
        ):
            op.verify()

    def test_invalid_operand_raises_validation_error(self, op_type):
        constant1 = ConstantOp(PhaseAttr(0.5))
        constant2 = ArithConstantOp.from_int_and_width(4, 32)
        op = op_type(constant1.results[0], constant2.results[0], result_type=PhaseType())
        with pytest.raises(VerifyException, match="operand 'rhs'"):
            op.verify()

    def test_invalid_result_type_raises_validation_error(self, op_type):
        constant1 = ConstantOp(PhaseAttr(0.5))
        constant2 = ConstantOp(PhaseAttr(1.0))
        op = op_type(
            constant1.results[0], constant2.results[0], result_type=FrameType("default")
        )
        with pytest.raises(VerifyException, match="result 'result' at position 0"):
            op.verify()


class TestAddOp:
    @pytest.mark.parametrize(
        "lhs, rhs, result",
        [
            (0.5, 1.0, 1.5),
            (5.5e9, 0.1e9, 5.6e9),
            (
                np.asarray([0.0, 0.5, 1.0]),
                np.array([0.0, 0.25, 0.5]),
                np.array([0.0, 0.75, 1.5]),
            ),
        ],
    )
    def test_py_operation(self, lhs, rhs, result):
        assert np.allclose(AddOp.py_operation(lhs, rhs), result)


class TestSubOp:
    @pytest.mark.parametrize(
        "lhs, rhs, result",
        [
            (1.0, 0.5, 0.5),
            (5.5e9, 0.1e9, 5.4e9),
            (
                np.asarray([0.0, 0.5, 1.0]),
                np.array([0.0, 0.25, 0.5]),
                np.array([0.0, 0.25, 0.5]),
            ),
        ],
    )
    def test_py_operation(self, lhs, rhs, result):
        assert np.allclose(SubOp.py_operation(lhs, rhs), result)


class TestMaxTimeOp:
    """Basic tests to check the operation is defined correctly."""

    @pytest.mark.parametrize(
        "times, expected",
        [
            ([TimeAttr(64e-9), TimeAttr(128e-9)], TimeAttr(128e-9)),
            ([TimeAttr(192e-9), TimeAttr(64e-9), TimeAttr(128e-9)], TimeAttr(192e-9)),
            ([TimeAttr(256e-9)], TimeAttr(256e-9)),
        ],
    )
    def test_initialization(self, times, expected):
        constants = [ConstantOp(time) for time in times]
        op = MaxTimeOp(*(constant.results[0] for constant in constants))

        assert list(op.times) == [constant.results[0] for constant in constants]
        assert expected.associated_type is TimeType
        assert op.result.type == TimeType()
        op.verify()

    @pytest.mark.parametrize(
        "attr",
        [
            PhaseAttr(0.5),
            FrequencyAttr(5.5e9),
            AmplitudeAttr(0.5 - 0.5j),
            SampledWaveformAttr(np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9)),
        ],
    )
    def test_invalid_operand_types(self, attr):
        constant1 = ConstantOp(attr)
        constant2 = ConstantOp(attr)
        op = MaxTimeOp(constant1.results[0], constant2.results[0])

        with pytest.raises(VerifyException, match="operand 'times'"):
            op.verify()

    def test_requires_at_least_one_operand(self):
        """Test that MaxTimeOp requires at least one time operand."""
        op = MaxTimeOp()
        with pytest.raises(VerifyException, match="operand 'times'"):
            op.verify()


class TestMixOp:
    def test_initialization(self):
        wf1 = SampledWaveformAttr(np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9))
        wf2 = SampledWaveformAttr(
            np.array([0.0, 0.25, 0.5]), TimeAttr(3e-9), TimeAttr(1e-9)
        )
        constant1 = ConstantOp(wf1)
        constant2 = ConstantOp(wf2)
        op = MixOp(constant1.results[0], constant2.results[0])
        assert op.result.type == WaveformType()
        op.verify()

    def test_with_non_waveform_type(self):
        constant1 = ConstantOp(PhaseAttr(1.0))
        constant2 = ConstantOp(PhaseAttr(0.5))
        op = MixOp(constant1.results[0], constant2.results[0])
        with pytest.raises(VerifyException, match="!pulse.phase"):
            op.verify()

    def test_py_operation(self):
        wf1 = np.asarray([0.0, 0.5, 1.0])
        wf2 = np.asarray([0.0, 0.25, 0.5])
        result = np.asarray([0.0, 0.125, 0.5])
        assert np.allclose(MixOp.py_operation(wf1, wf2), result)


class TestScaleOp:
    @pytest.mark.parametrize(
        "lhs, rhs, result",
        [
            (FloatAttr(2.0, 64), PhaseAttr(np.pi / 2), PhaseType()),
            (IntegerAttr(2, i64), PhaseAttr(np.pi / 2), PhaseType()),
            (FloatAttr(0.5, 64), FrequencyAttr(5.5e9), FrequencyType()),
            (IntegerAttr(2, i64), FrequencyAttr(5.5e9), FrequencyType()),
            (FloatAttr(0.5, 64), TimeAttr(160e-9), TimeType()),
            (IntegerAttr(2, i64), TimeAttr(160e-9), TimeType()),
            (FloatAttr(0.5, 64), AmplitudeAttr(0.5 - 0.5j), AmplitudeType()),
            (IntegerAttr(2, i64), AmplitudeAttr(0.5 - 0.5j), AmplitudeType()),
            (
                ComplexNumberAttr(0.5, -0.5, ComplexType(f64)),
                AmplitudeAttr(0.5 - 0.5j),
                AmplitudeType(),
            ),
            (
                FloatAttr(0.5, 64),
                SampledWaveformAttr(
                    np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9)
                ),
                WaveformType(),
            ),
            (
                IntegerAttr(2, i64),
                SampledWaveformAttr(
                    np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9)
                ),
                WaveformType(),
            ),
            (
                ComplexNumberAttr(0.5, -0.5, ComplexType(f64)),
                SampledWaveformAttr(
                    np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9)
                ),
                WaveformType(),
            ),
        ],
    )
    def test_initialization(self, lhs, rhs, result):
        if isinstance(lhs, FloatAttr):
            constant1 = ArithConstantOp(lhs, f64)
        elif isinstance(lhs, IntegerAttr):
            constant1 = ArithConstantOp(lhs, i64)
        elif isinstance(lhs, ComplexNumberAttr):
            constant1 = ComplexConstantOp(lhs, ComplexType(f64))

        constant2 = ConstantOp(rhs)
        op = ScaleOp(constant1.results[0], constant2.results[0], result)
        assert op.result.type == result
        op.verify()

    def test_invalid_lhs_type_raises(self):
        constant1 = ConstantOp(PhaseAttr(np.pi / 2))
        constant2 = ConstantOp(PhaseAttr(np.pi / 2))
        op = ScaleOp(constant1.results[0], constant2.results[0], PhaseType())
        with pytest.raises(VerifyException, match="operand 'lhs'"):
            op.verify()

    def test_invalid_rhs_type_raises(self):
        constant1 = ArithConstantOp(FloatAttr(2.0, 64), f64)
        constant2 = ArithConstantOp(FloatAttr(0.5, 64), f64)
        op = ScaleOp(constant1.results[0], constant2.results[0], FrameType("default"))
        with pytest.raises(VerifyException, match="operand 'rhs'"):
            op.verify()

    @pytest.mark.parametrize(
        "attr",
        [
            PhaseAttr(np.pi / 2),
            FrequencyAttr(5.5e9),
            TimeAttr(160e-9),
        ],
    )
    def test_complex_lhs_on_not_allowed_rhs_raises(self, attr):
        constant1 = ComplexConstantOp(
            ComplexNumberAttr(0.5, -0.5, ComplexType(f64)), ComplexType(f64)
        )
        constant2 = ConstantOp(attr)
        op = ScaleOp(constant1.results[0], constant2.results[0], attr.associated_type())
        with pytest.raises(VerifyException, match="Complex scaling is only supported"):
            op.verify()

    def test_mismatching_rhs_type_and_result_type_raises(self):
        constant1 = ArithConstantOp(FloatAttr(2.0, 64), f64)
        constant2 = ConstantOp(PhaseAttr(np.pi / 2))
        op = ScaleOp(constant1.results[0], constant2.results[0], FrequencyType())
        with pytest.raises(VerifyException, match="type of operand"):
            op.verify()

    @pytest.mark.parametrize(
        "lhs, rhs, result",
        [
            (2.0, 0.5, 1.0),
            (2, 0.5, 1.0),
            (0.5, 5.5e9, 2.75e9),
            (2, 5.5e9, 11e9),
            (0.5 - 0.5j, 0.5 + 0.5j, 0.5),
            (2, np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 2.0])),
            (0.5, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.25, 0.5])),
            (
                0.5 - 0.5j,
                np.array([0.0, 0.5, 1.0]),
                np.array([0.0, 0.25 - 0.25j, 0.5 - 0.5j]),
            ),
        ],
    )
    def test_py_operation(self, lhs, rhs, result):
        assert np.allclose(ScaleOp.py_operation(lhs, rhs), result)


class TestModuloOp:
    def test_properties(self):
        constant1 = ConstantOp(PhaseAttr(0.5))
        constant2 = ConstantOp(PhaseAttr(np.pi))
        op = ModuloOp(constant1.results[0], constant2.results[0], PhaseType())
        assert op.lhs == constant1.results[0]
        assert op.rhs == constant2.results[0]
        assert op.result.type == PhaseType()
        op.verify()

    @pytest.mark.parametrize(
        "attr",
        [
            FrequencyAttr(5.5e9),
            TimeAttr(160e-9),
            AmplitudeAttr(0.5 - 0.5j),
            SampledWaveformAttr(np.array([0.0, 0.5, 1.0]), TimeAttr(3e-9), TimeAttr(1e-9)),
        ],
    )
    def test_invalid_operand_types(self, attr):
        constant1 = ConstantOp(attr)
        constant2 = ConstantOp(attr)
        op = ModuloOp(constant1.results[0], constant2.results[0], attr.associated_type())
        with pytest.raises(VerifyException, match="operand 'lhs'"):
            op.verify()

    def test_invalid_result_type(self):
        constant1 = ConstantOp(PhaseAttr(0.5))
        constant2 = ConstantOp(PhaseAttr(np.pi))
        op = ModuloOp(constant1.results[0], constant2.results[0], FrameType("default"))
        with pytest.raises(VerifyException, match="result 'result' at position 0"):
            op.verify()

    @pytest.mark.parametrize(
        "lhs, rhs, result",
        [
            (0.5, 1.0, 0.5),
            (2.5 * np.pi, np.pi, 0.5 * np.pi),
            (-2.2 * np.pi, np.pi, 0.8 * np.pi),
        ],
    )
    def test_py_operation(self, lhs, rhs, result):
        assert np.isclose(ModuloOp.py_operation(lhs, rhs), result)


class TestSoftSquareWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "fractional_top_width": _float_constant(0.5),
        "fractional_rise": _float_constant(0.1),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "fractional_top_width": f64,
        "fractional_rise": f64,
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return SoftSquareWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["fractional_top_width"],
            shape_operands["fractional_rise"],
            False,
            _float_constant(0.2),
        )

    def test_initialization(self):
        width = ConstantOp(TimeAttr(800e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)
        fractional_rise = ArithConstantOp(FloatAttr(0.1, 64), f64)

        op = SoftSquareWaveformOp(
            width,
            amplitude,
            fractional_top_width,
            fractional_rise,
            BoolAttr(False, value_type=1),
        )
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.fractional_top_width == fractional_top_width.results[0]
        assert op.fractional_rise == fractional_rise.results[0]
        assert not op.regularize.value.data
        assert op.result.type == WaveformType()
        op.verify()

    def test_initialization_accepts_bool_for_regularize(self):
        width = ConstantOp(TimeAttr(800e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)
        fractional_rise = ArithConstantOp(FloatAttr(0.1, 64), f64)

        op = SoftSquareWaveformOp(
            width,
            amplitude,
            fractional_top_width,
            fractional_rise,
            False,
        )
        assert op.regularize.value.data is False
        op.verify()

    def test_build_shape_returns_expected_shape(self):
        """Builds a soft-square shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, SoftSquareWaveformShape)
        assert shape.fractional_top_width == pytest.approx(0.5)
        assert shape.fractional_rise == pytest.approx(0.1)
        assert shape.regularize is False

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any soft-square shape param cannot be constant-folded."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestSquareWaveformOp:
    @staticmethod
    def _build_waveform_op():
        return SquareWaveformOp(_time_constant(800e-9), _amplitude_constant(1.0))

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))

        op = SquareWaveformOp(width, amplitude)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.result.type == WaveformType()
        op.verify()

    def test_build_shape_always_returns_square_shape(self):
        """SquareWaveformOp has no shape params so build_shape always returns the shape."""
        shape = self._build_waveform_op().build_shape()

        assert isinstance(shape, SquareWaveformShape)

    def test_drag_coefficients_returns_empty_tuple(self):
        op = self._build_waveform_op()

        assert op.drag_coefficients == ()


class TestGaussianSquareWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "fractional_rise": _float_constant(0.2),
        "fractional_top_width": _float_constant(0.5),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "fractional_rise": f64,
        "fractional_top_width": f64,
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return GaussianSquareWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["fractional_rise"],
            shape_operands["fractional_top_width"],
            False,
            _float_constant(0.1),
        )

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_rise = ArithConstantOp(FloatAttr(0.2, 64), f64)
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)
        regularize = BoolAttr(False, value_type=1)

        op = GaussianSquareWaveformOp(
            width,
            amplitude,
            fractional_rise,
            fractional_top_width,
            regularize,
        )
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.fractional_rise == fractional_rise.results[0]
        assert op.fractional_top_width == fractional_top_width.results[0]
        assert op.regularize.value.data is False
        assert op.result.type == WaveformType()
        op.verify()

    def test_initialization_accepts_single_drag_coefficient(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_rise = ArithConstantOp(FloatAttr(0.2, 64), f64)
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)
        drag_coefficient = ArithConstantOp(FloatAttr(0.1, 64), f64)

        op = GaussianSquareWaveformOp(
            width,
            amplitude,
            fractional_rise,
            fractional_top_width,
            False,
            drag_coefficient,
        )
        assert len(op.drag_coefficients) == 1
        assert op.drag_coefficients[0] == drag_coefficient.results[0]
        op.verify()

    def test_initialization_accepts_bool_for_regularize(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_rise = ArithConstantOp(FloatAttr(0.2, 64), f64)
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)

        op = GaussianSquareWaveformOp(
            width,
            amplitude,
            fractional_rise,
            fractional_top_width,
            False,
        )
        assert op.regularize.value.data is False
        op.verify()

    def test_multiple_drag_coefficients_raises_verify_exception_on_verification(self):
        """Multiple DRAG coefficients are not allowed."""

        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_rise = ArithConstantOp(FloatAttr(0.2, 64), f64)
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)
        drag_first = ArithConstantOp(FloatAttr(0.1, 64), f64)
        drag_second = ArithConstantOp(FloatAttr(0.2, 64), f64)

        op = GaussianSquareWaveformOp(
            width,
            amplitude,
            fractional_rise,
            fractional_top_width,
            False,
            drag_first,
            drag_second,
        )
        with pytest.raises(VerifyException, match="supports at most one DRAG coefficient"):
            op.verify()

    def test_build_shape_returns_expected_shape(self):
        """Builds a Gaussian-square shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, GaussianSquareWaveformShape)
        assert shape.fractional_rise == pytest.approx(0.2)
        assert shape.fractional_top_width == pytest.approx(0.5)
        assert shape.regularize is False

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any Gaussian-square shape param is non-constant."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestGaussianWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "fractional_breadth": _float_constant(0.47),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "fractional_breadth": f64,
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return GaussianWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["fractional_breadth"],
            False,
            _float_constant(0.1),
        )

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_breadth = ArithConstantOp(FloatAttr(0.47, 64), f64)

        op = GaussianWaveformOp(
            width,
            amplitude,
            fractional_breadth,
            BoolAttr(False, value_type=1),
        )
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.fractional_breadth == fractional_breadth.results[0]
        assert not op.regularize.value.data
        assert op.result.type == WaveformType()
        op.verify()

    def test_initialization_accepts_bool_for_regularize(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_breadth = ArithConstantOp(FloatAttr(0.47, 64), f64)

        op = GaussianWaveformOp(width, amplitude, fractional_breadth, False)
        assert op.regularize.value.data is False
        op.verify()

    def test_initialization_with_drag_coefficients(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_breadth = ArithConstantOp(FloatAttr(0.47, 64), f64)
        drag_first = ArithConstantOp(FloatAttr(0.1, 64), f64)
        drag_second = ArithConstantOp(FloatAttr(0.2, 64), f64)

        op = GaussianWaveformOp(
            width,
            amplitude,
            fractional_breadth,
            False,
            drag_first,
            drag_second,
        )
        assert len(op.drag_coefficients) == 2
        assert op.drag_coefficients[0].owner.value.value.data == pytest.approx(0.1)
        assert op.drag_coefficients[1].owner.value.value.data == pytest.approx(0.2)
        op.verify()

    def test_build_shape_returns_expected_shape(self):
        """Builds a Gaussian shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, GaussianWaveformShape)
        assert shape.fractional_breadth == pytest.approx(0.47)
        assert shape.regularize is False

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any Gaussian shape param is non-constant."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestBlackmanWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))

        op = BlackmanWaveformOp(width, amplitude)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.result.type == WaveformType()
        op.verify()

    def test_build_shape_always_returns_blackman_shape(self):
        """BlackmanWaveformOp has no shape params so build_shape always returns the
        shape."""
        op = BlackmanWaveformOp(
            _time_constant(800e-9), _amplitude_constant(1.0), _float_constant(0.1)
        )
        shape = op.build_shape()

        assert isinstance(shape, BlackmanWaveformShape)


class TestSetupHoldWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "setup": _float_constant(0.5),
        "fractional_rise": _float_constant(0.1),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "setup": f64,
        "fractional_rise": f64,
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return SetupHoldWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["setup"],
            shape_operands["fractional_rise"],
        )

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        setup = ArithConstantOp(FloatAttr(0.5, 64), f64)
        fractional_rise = ArithConstantOp(FloatAttr(0.1, 64), f64)

        op = SetupHoldWaveformOp(width, amplitude, setup, fractional_rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.setup == setup.results[0]
        assert op.fractional_rise == fractional_rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()

    def test_drag_coefficients_returns_empty_tuple(self):
        op = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS)

        assert op.drag_coefficients == ()

    def test_build_shape_returns_expected_shape(self):
        """Builds a setup-hold shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, SetupHoldWaveformShape)
        assert shape.setup == pytest.approx(0.5)
        assert shape.rise_location == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any setup-hold shape param is non-constant."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestRoundedSquareWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "fractional_top_width": _float_constant(0.5),
        "fractional_rise": _float_constant(0.1),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "fractional_top_width": f64,
        "fractional_rise": f64,
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return RoundedSquareWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["fractional_top_width"],
            shape_operands["fractional_rise"],
            _float_constant(0.2),
        )

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_top_width = ArithConstantOp(FloatAttr(0.5, 64), f64)
        fractional_rise = ArithConstantOp(FloatAttr(0.1, 64), f64)

        op = RoundedSquareWaveformOp(
            width,
            amplitude,
            fractional_top_width,
            fractional_rise,
        )

        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.fractional_top_width == fractional_top_width.results[0]
        assert op.fractional_rise == fractional_rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()

    def test_build_shape_returns_expected_shape(self):
        """Builds a rounded-square shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, RoundedSquareWaveformShape)
        assert shape.fractional_top_width == pytest.approx(0.5)
        assert shape.fractional_rise == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any rounded-square shape param is non-constant."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestSinusoidalWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "number_of_periods": _float_constant(0.5),
        "internal_phase": _phase_constant(1.57),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "number_of_periods": f64,
        "internal_phase": PhaseType(),
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return SinusoidalWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["number_of_periods"],
            shape_operands["internal_phase"],
            _float_constant(0.2),
        )

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        number_of_periods = ArithConstantOp(FloatAttr(0.5, 64), f64)
        internal_phase = ConstantOp(PhaseAttr(1.57))

        op = SinusoidalWaveformOp(width, amplitude, number_of_periods, internal_phase)

        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.number_of_periods == number_of_periods.results[0]
        assert op.internal_phase == internal_phase.results[0]

        assert op.result.type == WaveformType()
        op.verify()

    def test_build_shape_returns_expected_shape(self):
        """Builds a sinusoidal shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, SinusoidalWaveformShape)
        assert shape.number_of_periods == pytest.approx(0.5)
        assert shape.internal_phase == pytest.approx(1.57)

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any sinusoidal shape param is non-constant."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestSechWaveformOp:
    _BUILD_SHAPE_OPERANDS = {
        "fractional_breadth": _float_constant(1.0 / 3.0),
    }
    _BUILD_SHAPE_OPERAND_TYPES = {
        "fractional_breadth": f64,
    }

    @staticmethod
    def _build_waveform_op(shape_operands):
        return SechWaveformOp(
            _time_constant(800e-9),
            _amplitude_constant(1.0),
            shape_operands["fractional_breadth"],
            False,
            _float_constant(0.1),
        )

    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_breadth = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)

        op = SechWaveformOp(
            width,
            amplitude,
            fractional_breadth,
            BoolAttr(False, value_type=1),
        )
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.fractional_breadth == fractional_breadth.results[0]
        assert not op.regularize.value.data
        assert op.result.type == WaveformType()
        op.verify()

    def test_initialization_accepts_bool_for_regularize(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        fractional_breadth = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)

        op = SechWaveformOp(width, amplitude, fractional_breadth, False)
        assert op.regularize.value.data is False
        op.verify()

    def test_build_shape_returns_expected_shape(self):
        """Builds a sech shape when shape operands are constants."""
        shape = self._build_waveform_op(self._BUILD_SHAPE_OPERANDS).build_shape()

        assert isinstance(shape, SechWaveformShape)
        assert shape.fractional_breadth == pytest.approx(1.0 / 3.0)
        assert shape.regularize is False

    @pytest.mark.parametrize(
        "operand_name",
        [pytest.param(name, id=name) for name in _BUILD_SHAPE_OPERAND_TYPES],
    )
    def test_build_shape_returns_none_for_non_constant_shape_params(self, operand_name):
        """Returns ``None`` when any sech shape param is non-constant."""
        non_constant_operands = dict(self._BUILD_SHAPE_OPERANDS)
        non_constant_operands[operand_name] = _ProducerOp(
            self._BUILD_SHAPE_OPERAND_TYPES[operand_name]
        ).results[0]

        assert self._build_waveform_op(non_constant_operands).build_shape() is None


class TestCreateFrameOp:
    def test_minimal_initialization(self):
        """Creating a frame should use port as the identifier."""
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(frequency, StringAttr("drive"))
        assert frame.frequency == frequency.results[0]
        assert frame.port == StringAttr("drive")
        assert frame.result.type == FrameType("drive")

        assert frame.imbalance is None
        assert frame.phase_offset is None

        assert frame.acquire_allowed.value.data
        assert frame.pulse_allowed.value.data
        assert frame.track_phase.value.data
        frame.verify()

    def test_create_frame_with_different_port_sets_parameterized_result_type(self):
        """Creating a frame with a different port should parameterize the frame result
        type."""
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(
            frequency,
            StringAttr("measure"),
        )
        assert frame.port == StringAttr("measure")
        assert frame.result.type == FrameType("measure")
        frame.verify()

    def test_read_port_from_create_frame_result_type(self):
        """The frame port should match the result type port parameter."""
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(
            frequency,
            StringAttr("measure"),
        )
        assert frame.port == frame.result.type.port
        frame.verify()

    def test_with_optionals(self):
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(
            frequency,
            StringAttr("drive"),
            imbalance=FloatAttr(0.9, 64),
            phase_offset=FloatAttr(0.1, 64),
        )
        assert isinstance(frame.imbalance, FloatAttr)
        assert isinstance(frame.phase_offset, FloatAttr)
        assert frame.imbalance.value.data == 0.9
        assert frame.phase_offset.value.data == 0.1
        frame.verify()

    def test_with_non_defaults(self):
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        frame = CreateFrameOp(
            frequency,
            StringAttr("drive"),
            acquire_allowed=BoolAttr(False, value_type=1),
            pulse_allowed=BoolAttr(False, value_type=1),
            track_phase=BoolAttr(False, value_type=1),
        )
        assert not frame.acquire_allowed.value.data
        assert not frame.pulse_allowed.value.data
        assert not frame.track_phase.value.data
        frame.verify()


@pytest.mark.parametrize("op", [PhaseSetOp, PhaseShiftOp])
class TestPhaseOps:
    def test_initialization(self, op):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        phase = ConstantOp(PhaseAttr(1.57))
        phase_op = op(frame.results[0], phase.results[0])
        assert phase_op.frame == frame.results[0]
        assert phase_op.phase == phase.results[0]
        assert phase_op.result.type == FrameType("drive")
        assert "phase" in phase_op.name
        phase_op.verify()

    def test_apply_phase_operation_preserves_parameterized_frame_type(self, op):
        """Applying a phase operation should preserve the input frame parameterization."""
        frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("measure"),
        )
        phase = ConstantOp(PhaseAttr(1.57))
        phase_op = op(frame.results[0], phase.results[0])
        assert phase_op.result.type == FrameType("measure")
        phase_op.verify()


class TestWaitOp:
    def test_initialization(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        time = ConstantOp(TimeAttr(800e-9))
        wait_op = WaitOp(frame.results[0], time.results[0])
        assert wait_op.frame == frame.results[0]
        assert wait_op.duration == time.results[0]
        assert wait_op.result.type == FrameType("drive")
        wait_op.verify()

    def test_wait_operation_preserves_parameterized_frame_type(self):
        """Waiting on a parameterized frame should keep the frame parameterization."""
        frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("measure"),
        )
        time = ConstantOp(TimeAttr(800e-9))
        wait_op = WaitOp(frame.results[0], time.results[0])
        assert wait_op.result.type == FrameType("measure")
        wait_op.verify()


class TestSynchronizeOp:
    @pytest.mark.parametrize("num_frames", [2, 3])
    def test_initialization(self, num_frames):
        frames = [
            CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr(f"drive_{i}"))
            for i in range(num_frames)
        ]
        sync_op = SynchronizeOp(*[chan.result for chan in frames])
        assert sync_op.frames == tuple(chan.result for chan in frames)
        assert len(sync_op.result) == num_frames
        assert len(sync_op.results) == num_frames
        sync_op.verify()

    def test_verification_fails_with_one_frame(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        sync_op = SynchronizeOp(frame.result)
        with pytest.raises(VerifyException, match="At least two frames"):
            sync_op.verify()

    def test_sync_operation_preserves_each_parameterized_frame_type(self):
        """Synchronizing frames should preserve each input frame parameterization."""
        output_frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("drive"),
        )
        input_frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(6.8e9)),
            StringAttr("measure"),
        )
        sync_op = SynchronizeOp(output_frame.result, input_frame.result)
        assert sync_op.results[0].type == FrameType("drive")
        assert sync_op.results[1].type == FrameType("measure")
        sync_op.verify()


class TestPulseOp:
    def test_initialization(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        width = ConstantOp(TimeAttr(800e-9))
        amp = ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(width, amp)
        pulse_op = PulseOp(frame.result, waveform.result)
        assert pulse_op.frame == frame.results[0]
        assert pulse_op.waveform == waveform.result
        assert pulse_op.result.type == FrameType("drive")
        pulse_op.verify()

    def test_apply_pulse_preserves_parameterized_frame_type(self):
        """Playing a pulse should preserve the input frame parameterization."""
        frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("measure"),
        )
        width = ConstantOp(TimeAttr(800e-9))
        amp = ConstantOp(AmplitudeAttr(1.0))
        waveform = SquareWaveformOp(width, amp)
        pulse_op = PulseOp(frame.result, waveform.result)
        assert pulse_op.result.type == FrameType("measure")
        pulse_op.verify()


class TestStartContinuousWaveformOp:
    def test_initialization(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        start_op = StartContinuousWaveformOp(frame.result, amplitude.result)
        assert start_op.frame == frame.results[0]
        assert start_op.amplitude == amplitude.result
        assert start_op.result.type == FrameType("drive")
        start_op.verify()

    def test_start_continuous_waveform_preserves_parameterized_frame_type(self):
        """Starting a continuous waveform should preserve frame parameterization."""
        frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("measure"),
        )
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        start_op = StartContinuousWaveformOp(frame.result, amplitude.result)
        assert start_op.result.type == FrameType("measure")
        start_op.verify()


class TestStopContinuousWaveformOp:
    def test_initialization(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("drive"))
        stop_op = StopContinuousWaveformOp(frame.result)
        assert stop_op.frame == frame.results[0]
        assert stop_op.result.type == FrameType("drive")
        stop_op.verify()

    def test_stop_continuous_waveform_preserves_parameterized_frame_type(self):
        """Stopping a continuous waveform should preserve frame parameterization."""
        frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("measure"),
        )
        stop_op = StopContinuousWaveformOp(frame.result)
        assert stop_op.result.type == FrameType("measure")
        stop_op.verify()


class TestAcquireOp:
    def test_initialization(self):
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("measure"))
        duration = ConstantOp(TimeAttr(400e-9))
        acquire_op = AcquireOp(frame.result, duration.result)
        assert acquire_op.frame == frame.results[0]
        assert acquire_op.duration is duration.result
        assert len(acquire_op.results) == 2
        assert acquire_op.frame_result.type == FrameType("measure")
        assert acquire_op.acquisition_result.type == AcquisitionType()
        assert acquire_op.weights is None
        acquire_op.verify()

    def test_acquire_operation_preserves_parameterized_frame_type(self):
        """Acquiring on a parameterized frame should preserve frame_result type."""
        frame = CreateFrameOp(
            ConstantOp(FrequencyAttr(5.0e9)),
            StringAttr("measure"),
        )
        duration = ConstantOp(TimeAttr(400e-9))
        acquire_op = AcquireOp(frame.result, duration.result)
        assert acquire_op.frame_result.type == FrameType("measure")
        acquire_op.verify()

    def test_with_weights_is_valid(self):
        """Tests that an AcquireOp with weights is valid and the weights are accessible."""
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("measure"))
        duration = ConstantOp(TimeAttr(400e-9))
        weights = np.asarray([0.1, 0.2, 0.3])
        weights_attr = WeightsAttr(weights)
        acquire_op = AcquireOp(frame.result, duration.result, weights=weights_attr)
        assert acquire_op.weights == weights_attr
        acquire_op.verify()

    def test_with_label(self):
        """Tests that an AcquireOp with a label is valid and the label is accessible."""
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("measure"))
        duration = ConstantOp(TimeAttr(400e-9))
        label = StringAttr("acquire_label")
        acquire_op = AcquireOp(frame.result, duration.result, label=label)
        assert acquire_op.label == label
        acquire_op.verify()


class TestIntegrateOp:
    """Tests the integration operation with initialization and verification."""

    def test_initialization_with_result_does_not_raise_validation_error(self):
        """Tests when the result is directly passed to the integrate operation."""

        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5.0e9)), StringAttr("measure"))
        duration = ConstantOp(TimeAttr(400e-9))
        acquire_op = AcquireOp(frame.result, duration.result)
        integrate_op = IntegrateOp(acquire_op.acquisition_result)
        assert integrate_op.acquisition == acquire_op.acquisition_result
        assert integrate_op.result.type == IQResultType()
        integrate_op.verify()


class TestEqualiseOp:
    """The equalise operation applies an affine transformation to an IQ value with the
    intention of normalising the distributions to have favourable properties.

    These tests check the initialisation of this operation passes verification.
    """

    def test_initialisation_passes_verification(self):
        """Initialises an operation that should be valid, and runs verification to ensure it
        does not fail."""
        block = Block(arg_types=[IQResultType()])
        iq_value = block.args[0]
        affine = EqualiseAttr(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
        op = EqualiseOp(iq_value, affine)
        assert op.value == iq_value
        assert op.affine_transform == affine
        assert op.result.type == IQResultType()
        op.verify()

    def test_initialisation_from_operation_uses_iq_result_operand(self):
        """Passing an operation should resolve the matching IQ result operand."""

        block = Block(arg_types=[AcquisitionType()])
        acquisition = block.args[0]
        integrate_op = IntegrateOp(acquisition)
        affine = EqualiseAttr(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)

        op = EqualiseOp(integrate_op, affine)

        assert op.value == integrate_op.result
        op.verify()


class TestDiscriminateOp:
    """The discriminate operation maps an IQ value to a discriminated integer state
    according to a given policy.

    These tests test the creation of those operations, ensuring the results type matches the
    provided policy, and that the operation passes verification.
    """

    @irdl_attr_definition
    class _MockDiscriminatorPolicy(DiscriminatorPolicyAttr):
        """A configurable mock discriminator policy for testing."""

        name = "pulse.discriminate_test_mock_policy"
        POLICY_NAME = "mock"
        min_val: IntAttr = param_def()
        max_val: IntAttr = param_def()

        def __init__(self, min_state: int, max_state: int):
            return super().__init__(IntAttr(min_state), IntAttr(max_state))

        @property
        def state_range(self) -> tuple[int, int]:
            return (self.min_val.data, self.max_val.data)

    @pytest.mark.parametrize(
        "num_states, invalid_state", [(2, False), (3, False), (3, True)]
    )
    def test_initialisation_with_mock_policy_has_desired_results_type(
        self, num_states, invalid_state
    ):
        """Creates a mock policy that allows for a dynamic number of states, optionally
        allowing for the invalid state map ``-1``, and tests its integration with this
        operation.

        Tests the results type has the correct range, and that the op passes verification.
        """
        min_state = -1 if invalid_state else 0
        max_state = num_states - 1

        policy = self._MockDiscriminatorPolicy(min_state, max_state)
        block = Block(arg_types=[IQResultType()])
        iq_value = block.args[0]

        op = DiscriminateOp(iq_value, policy)
        assert op.result.type == StateKeyType(min_state, max_state)
        op.verify()

    def test_policy_that_is_not_state_discrimination_fails_verification(self):
        """The policy attribute should be a state discrimination policy type.

        If that's not the case, verification should fail. This tests that.
        """
        block = Block(arg_types=[IQResultType()])
        iq_value = block.args[0]

        policy = RealThresholdPolicyAttr(0.5)
        op = DiscriminateOp(iq_value, policy)

        # Replace the policy property with a non-DiscriminatorPolicyAttr attribute
        op.properties["policy"] = IntAttr(1)

        with pytest.raises(VerifyException, match="policy"):
            op.verify()

    def test_non_matching_result_type_and_policy_fails_verification(self):
        """Verification should reject IR where the result type disagrees with the policy."""

        block = Block(arg_types=[IQResultType()])
        iq_value = block.args[0]
        policy = self._MockDiscriminatorPolicy(-1, 2)
        op = DiscriminateOp(iq_value, policy)

        op.properties["policy"] = self._MockDiscriminatorPolicy(0, 1)

        with pytest.raises(VerifyException, match="result state type must match"):
            op.verify()


class TestStateMapOp:
    """The state map op maps a StateKeyType onto a binary integer.

    This tests that operation is instantiated correctly with legal values, and verification
    passes, and that verification fails when the state map does not match the allowed
    integer state keys.

    To facilitate testing, the test class includes a mock operation that returns a state
    type with custom minimum and maximum integer state keys.
    """

    @irdl_op_definition
    class _MockOp(IRDLOperation):
        """A minimal mock op that produces a StateKeyType SSA result for testing."""

        name = "pulse.state_map_test_mock_op"
        res = result_def(AnyAttr())

        def __init__(self, state_type: StateKeyType):
            return super().__init__(result_types=[state_type])

    def test_initialisation_with_matching_value_and_mapping_passes_verification(self):
        """Creates a valid operation and runs verification, testing it passes."""
        state_type = StateKeyType(0, 1)
        mock_op = self._MockOp(state_type)
        op = StateMapOp(mock_op.res, {0: 0, 1: 1})
        assert isinstance(op.mapping, StateMapDictAttr)
        op.verify()

    def test_initialisation_with_state_map_dict_attr(self):
        """Tests that the state map can be initialised with a StateMapDictAttr directly."""
        state_type = StateKeyType(0, 1)
        mock_op = self._MockOp(state_type)
        mapping = StateMapDictAttr({0: 0, 1: 1})
        op = StateMapOp(mock_op.res, mapping)
        assert op.mapping is mapping
        op.verify()

    def test_operation_with_non_matching_value_and_mapping_fails_verification(self):
        """Creates an operation where the state type does not match the values in the state
        map, and checks for a verify error with the correct error message."""
        state_type = StateKeyType(0, 1)
        mock_op = self._MockOp(state_type)
        # Mapping includes a key (2) not in the state type range (0..1)
        op = StateMapOp(mock_op.res, {0: 0, 1: 1, 2: 0})
        with pytest.raises(
            VerifyException,
            match="does not contain a mapping for every allowed state",
        ):
            op.verify()

    def test_operation_with_empty_mapping_fails_verification(self):
        """An empty mapping should fail verification with a clear error message."""

        state_type = StateKeyType(0, 1)
        mock_op = self._MockOp(state_type)
        op = StateMapOp(mock_op.res, {})
        with pytest.raises(VerifyException, match="state map cannot be empty"):
            op.verify()

    def test_state_map_with_non_i1_values_raises_error(self):
        """The state map must have attributes which are integers with one bitwidth.

        This tests that when this isn't the case, a suitable error is raised.
        """
        state_type = StateKeyType(0, 1)
        mock_op = self._MockOp(state_type)
        # Value 2 is not a valid i1 (binary) value
        op = StateMapOp(mock_op.res, {0: 0, 1: 2})
        with pytest.raises(VerifyException, match="binary"):
            op.verify()


def _build_region(ops: list[Operation], arg_types: list[Attribute] | None = None) -> Region:
    """Builds a single-block region with optional block arguments."""

    block = Block(arg_types=arg_types or [])
    block.add_ops(ops)
    return Region(block)


def _build_module(*ops: Operation) -> ModuleOp:
    """Builds a module operation containing the given top-level ops."""

    return ModuleOp(ops=list(ops))


def _build_kernel(
    name: str | StringAttr,
    function_type: FunctionType | tuple[list[Attribute], list[Attribute]],
    region: Region | type[Region.DEFAULT] = Region.DEFAULT,
) -> KernelOp:
    """Builds a kernel operation with a given signature and region."""

    return KernelOp(name=name, function_type=function_type, region=region)


class TestKernelOp:
    """Tests the instantiation and verification of the kernel operation."""

    def _build_body(
        self,
        ops: list[Operation],
        arg_types: list[Attribute] | None = None,
    ) -> Region:
        """Builds a single-block body for kernel tests."""

        block = Block(arg_types=arg_types or [])
        block.add_ops(ops)

        return Region(block)

    def test_kernel_has_isolated_from_above_trait(self):
        """Tests the kernel advertises IsolatedFromAbove semantics via its traits."""

        kernel = _build_kernel("my_kernel", ([], []), Region())
        assert kernel.has_trait(IsolatedFromAbove)

    def test_kernel_is_recorded_in_symbol_table_of_module(self):
        """The kernel should be accessible through the symbol table."""

        kernel = _build_kernel("my_kernel", ([], []), Region())
        module = _build_module(kernel)
        found = SymbolTable.lookup_symbol(module, FlatSymbolRefAttr("my_kernel"))
        assert found is kernel

    def test_instantiation_from_string_builds_attribute(self):
        """Tests that the kernel can be instantiated from a string and the attribute is
        built correctly."""

        kernel = _build_kernel("my_kernel", ([], []), Region())
        assert kernel.sym_name == StringAttr("my_kernel")
        kernel.verify()

    def test_instantiation_from_string_attribute(self):
        """Tests that the kernel can be instantiated from a string attribute and the
        attribute is built correctly."""

        symbol = StringAttr("my_kernel")
        kernel = _build_kernel(symbol, ([], []), Region())
        assert kernel.sym_name is symbol
        kernel.verify()

    def test_from_function_type_attribute(self):
        """Tests that the kernel can be instantiated from a function type attribute and the
        attribute is built correctly."""

        function_type = FunctionType.from_lists([PhaseType()], [FrequencyType()])
        kernel = _build_kernel("my_kernel", function_type, Region())
        assert kernel.function_type is function_type
        kernel.verify()

    def test_function_type_from_tuple(self):
        """Tests that the kernel can be instantiated from a tuple of input and output types
        and the attribute is built correctly."""

        kernel = _build_kernel("my_kernel", ([PhaseType()], [PhaseType()]), Region())
        assert isinstance(kernel.function_type, FunctionType)
        assert list(kernel.function_type.inputs) == [PhaseType()]
        assert list(kernel.function_type.outputs) == [PhaseType()]
        kernel.verify()

    def test_frame_inputs_raises_verify_exception(self):
        """Tests that the kernel cannot have frame inputs."""

        kernel = _build_kernel("my_kernel", ([FrameType("drive")], []), Region())
        with pytest.raises(VerifyException, match="Passing a frame as an argument"):
            kernel.verify()

    def test_frame_returns_raises_verify_exception(self):
        """Tests that the kernel cannot have frame returns."""

        kernel = _build_kernel("my_kernel", ([], [FrameType("drive")]), Region())
        with pytest.raises(VerifyException, match="Returning a frame from a kernel"):
            kernel.verify()

    def test_region_with_no_blocks_passes_verification(self):
        """Tests that a kernel with no blocks passes verification, which symbolically
        represents a function defined elsewhere."""

        kernel = _build_kernel("my_kernel", ([PhaseType()], [FrequencyType()]), Region())
        kernel.verify()

    def test_region_with_different_block_args_to_func_args_raises_verify_exception(self):
        """Tests that a kernel with a block with arguments that do not match the function
        type raises a verify exception."""

        block = Block(arg_types=[FrequencyType()])
        block.add_op(ReturnOp(block.args[0]))
        body = Region(block)
        kernel = _build_kernel("my_kernel", ([PhaseType()], [FrequencyType()]), body)
        with pytest.raises(VerifyException, match="types of the block arguments"):
            kernel.verify()

    def test_with_valid_return_types_and_arguments_passes_verification(self):
        """Tests that a kernel with a block with arguments that match the function type and
        a return operation with matching types passes verification."""

        block = Block(arg_types=[PhaseType()])
        block.add_op(ReturnOp(block.args[0]))
        body = Region(block)
        kernel = _build_kernel("my_kernel", ([PhaseType()], [PhaseType()]), body)
        kernel.verify()


class TestReturnOp:
    """Tests verification and initialisation of the return operation."""

    def test_non_kernel_parent_raises_verify_exception(self):
        """Tests that a return operation with a non-kernel parent raises a verify
        exception."""

        value = ConstantOp(PhaseAttr(0.0))
        return_op = ReturnOp(value.result)
        parent = func.FuncOp("main", ((), ()), _build_region([value, return_op]))
        assert parent.body.block.last_op is return_op
        with pytest.raises(VerifyException, match="expects parent op 'pulse.kernel'"):
            return_op.verify()

    def test_operations_after_in_block_raises_verify_exception(self):
        """Tests that a return operation with operations after it in the block raises a
        verify exception, as this is a terminator operation."""

        value = ConstantOp(PhaseAttr(0.0))
        return_op = ReturnOp(value.result)
        trailing_op = ConstantOp(PhaseAttr(0.0))
        body = _build_region([value, return_op, trailing_op])
        kernel = _build_kernel("my_kernel", ([], [PhaseType()]), body)

        with pytest.raises(VerifyException, match="must be the last operation"):
            kernel.verify()

    def test_return_types_that_are_different_to_parent_raises_verify_exception(self):
        """Tests that a return operation with types that are different to the parent kernel
        raises a verify exception."""

        frequency = ConstantOp(FrequencyAttr(5.0e9))
        return_op = ReturnOp(frequency.result)
        body = _build_region([frequency, return_op])
        kernel = _build_kernel("my_kernel", ([], [PhaseType()]), body)

        with pytest.raises(VerifyException, match="return types of the return operation"):
            kernel.verify()

    def test_return_types_matching_parent_passes_verify_impl(self):
        """A return op with matching parent function outputs should pass verify_."""

        phase = ConstantOp(PhaseAttr(0.0))
        return_op = ReturnOp(phase.result)
        body = _build_region([phase, return_op], arg_types=[PhaseType()])
        _build_kernel("my_kernel", ([PhaseType()], [PhaseType()]), body)

        return_op.verify()


class TestCallKernelOp:
    """Tests the instantiation and verification of the call kernel operation."""

    def test_instantiation_with_string_callee(self):
        """Tests that the call kernel operation can be instantiated with a string callee and
        the attribute is built correctly."""

        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp("my_kernel", [argument.result], [PhaseType()])

        assert call.callee == FlatSymbolRefAttr("my_kernel")
        assert call.arguments == (argument.result,)
        assert list(call.results.types) == [PhaseType()]

    def test_instantiation_with_symbol_ref_attr_callee(self):
        """Tests that the call kernel operation can be instantiated with a symbol ref attr
        callee and the attribute is built correctly."""

        argument = ConstantOp(PhaseAttr(0.25))
        callee = FlatSymbolRefAttr("my_kernel")
        call = CallKernelOp(callee, [argument.result], [PhaseType()])
        assert call.callee is callee
        assert call.arguments == (argument.result,)
        assert list(call.results.types) == [PhaseType()]

    def test_unfound_symbol_raises_verify_exception(self):
        """Tests that the call kernel operation cannot be verified if the callee is not
        found in the symbol table."""

        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp("missing_kernel", [argument.result], [PhaseType()])
        _build_module(argument, call)

        with pytest.raises(VerifyException, match="no symbol was found"):
            call.verify()

    def test_non_kernel_callee_raises_verify_exception(self):
        """Tests that the call kernel operation cannot be instantiated with a callee that is
        not a kernel and raises a verify exception."""

        non_kernel = func.FuncOp("my_kernel", ((PhaseType(),), (PhaseType(),)), Region())
        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp("my_kernel", [argument.result], [PhaseType()])
        _build_module(non_kernel, argument, call)

        with pytest.raises(VerifyException, match="must reference a KernelOp"):
            call.verify()

    def test_mismatching_number_of_arguments_raises_verify_exception(self):
        """Tests that the call kernel operation cannot be verified if the number of
        arguments does not match the number of callee arguments."""

        kernel = _build_kernel(
            "my_kernel",
            ([PhaseType(), FrequencyType()], [PhaseType()]),
            Region(),
        )
        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp("my_kernel", [argument.result], [PhaseType()])
        _build_module(kernel, argument, call)

        with pytest.raises(VerifyException, match="same number of arguments"):
            call.verify()

    def test_mismatching_number_of_results_raises_verify_exception(self):
        """Tests that the call kernel operation cannot be verified if the number of results
        does not match the number of callee results."""

        kernel = _build_kernel(
            "my_kernel",
            ([PhaseType()], [PhaseType(), FrequencyType()]),
            Region(),
        )
        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp("my_kernel", [argument.result], [PhaseType()])
        _build_module(kernel, argument, call)

        with pytest.raises(VerifyException, match="same number of results"):
            call.verify()

    def test_mismatching_argument_types_raises_verify_exception(self):
        """Tests that the call kernel operation cannot be verified if the argument types do
        not match the callee argument types."""

        kernel = _build_kernel("my_kernel", ([PhaseType()], [PhaseType()]), Region())
        wrong_argument = ConstantOp(FrequencyAttr(5.0e9))
        call = CallKernelOp("my_kernel", [wrong_argument.result], [PhaseType()])
        _build_module(kernel, wrong_argument, call)

        with pytest.raises(VerifyException, match="same argument types"):
            call.verify()

    def test_mismatching_result_types_raises_verify_exception(self):
        """Tests that the call kernel operation cannot be verified if the result types do
        not match the callee result types."""

        kernel = _build_kernel("my_kernel", ([PhaseType()], [PhaseType()]), Region())
        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp("my_kernel", [argument.result], [FrequencyType()])
        _build_module(kernel, argument, call)

        with pytest.raises(VerifyException, match="same result types"):
            CallKernelOpUserOpInterface().verify(call)

    def test_matching_multiple_result_types_passes_verification(self):
        """Matching multiple outputs should verify without raising."""

        kernel = _build_kernel(
            "my_kernel",
            ([PhaseType()], [PhaseType(), FrequencyType()]),
            Region(),
        )
        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp(
            "my_kernel",
            [argument.result],
            [[PhaseType(), FrequencyType()]],
        )
        _build_module(kernel, argument, call)
        call.verify()

    def test_mismatching_second_result_type_raises_verify_exception(self):
        """If the second result type mismatches, verification should fail on index 1."""

        kernel = _build_kernel(
            "my_kernel",
            ([PhaseType()], [PhaseType(), FrequencyType()]),
            Region(),
        )
        argument = ConstantOp(PhaseAttr(0.25))
        call = CallKernelOp(
            "my_kernel",
            [argument.result],
            [[PhaseType(), AmplitudeType()]],
        )
        _build_module(kernel, argument, call)

        with pytest.raises(VerifyException, match="result 1"):
            call.verify()
