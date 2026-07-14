# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import (
    BoolAttr,
    ComplexType,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    f64,
    i64,
)
from xdsl.dialects.complex import ComplexNumberAttr, ConstantOp as ComplexConstantOp
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AcquisitionType,
    AddOp,
    AmplitudeAttr,
    AmplitudeType,
    BlackmanWaveformOp,
    ConstantOp,
    CosWaveformOp,
    CreateFrameOp,
    DragGaussianWaveformOp,
    ExtraSoftSquareWaveformOp,
    FrameType,
    FrequencyAttr,
    FrequencyType,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    GaussianZeroEdgeWaveformOp,
    IntegrateOp,
    IQResultType,
    MaxTimeOp,
    MixOp,
    ModuloOp,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PhaseType,
    PulseOp,
    RoundedSquareWaveformOp,
    SampledWaveformAttr,
    ScaleOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinWaveformOp,
    SofterGaussianWaveformOp,
    SofterSquareWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SubOp,
    SynchronizeOp,
    TimeAttr,
    TimeType,
    WaitOp,
    WaveformType,
    WeightsAttr,
)


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
    def test_initialization(self):
        width = ConstantOp(TimeAttr(800e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)

        op = SoftSquareWaveformOp(width, amplitude, rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.rise == rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestSofterSquareWaveformOp:
    def test_initialization(self):
        width = ConstantOp(TimeAttr(800e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        std_dev = ConstantOp(TimeAttr(200e-9))
        rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)

        op = SofterSquareWaveformOp(width, amplitude, std_dev, rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.std_dev == std_dev.results[0]
        assert op.rise == rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestExtraSoftSquareWaveformOp:
    def test_initialization(self):
        width = ConstantOp(TimeAttr(800e-9))
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        std_dev = ConstantOp(TimeAttr(200e-9))
        rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)

        op = ExtraSoftSquareWaveformOp(width, amplitude, std_dev, rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.std_dev == std_dev.results[0]
        assert op.rise == rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestSquareWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))

        op = SquareWaveformOp(width, amplitude)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestGaussianSquareWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        std_dev = ConstantOp(TimeAttr(200e-9))
        square_width = ConstantOp(TimeAttr(400e-9))
        zero_at_edges = BoolAttr(False, value_type=1)

        op = GaussianSquareWaveformOp(
            width, amplitude, std_dev, square_width, zero_at_edges
        )
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.std_dev == std_dev.results[0]
        assert op.square_width == square_width.results[0]
        assert op.zero_at_edges.value.data is False
        assert op.result.type == WaveformType()
        op.verify()


class TestGaussianWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)

        op = GaussianWaveformOp(width, amplitude, rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.rise == rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestSofterGaussianWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)
        op = SofterGaussianWaveformOp(width, amplitude, rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.rise == rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestBlackmanWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))

        op = BlackmanWaveformOp(width, amplitude)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestSetupHoldWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        amp_setup = ConstantOp(AmplitudeAttr(1.0))
        rise = ConstantOp(TimeAttr(200e-9))

        op = SetupHoldWaveformOp(width, amplitude, amp_setup, rise)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.amp_setup == amp_setup.results[0]
        assert op.rise == rise.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestRoundedSquareWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        rise = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)
        std_dev = ConstantOp(TimeAttr(200e-9))

        op = RoundedSquareWaveformOp(width, amplitude, rise, std_dev)

        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.rise == rise.results[0]
        assert op.std_dev == std_dev.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestDragGaussianWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(
            TimeAttr(800e-9),
        )
        std_dev = ConstantOp(TimeAttr(200e-9))
        beta = ArithConstantOp(FloatAttr(1.0 / 3.0, 64), f64)
        zero_at_edges = BoolAttr(False, value_type=1)

        op = DragGaussianWaveformOp(width, amplitude, std_dev, beta, zero_at_edges)

        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.std_dev == std_dev.results[0]
        assert op.beta == beta.results[0]
        assert op.zero_at_edges.value.data is False

        assert op.result.type == WaveformType()
        op.verify()


class TestGaussianZeroEdgeWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(
            TimeAttr(800e-9),
        )
        std_dev = ConstantOp(TimeAttr(200e-9))
        zero_at_edges = BoolAttr(False, 1)

        op = GaussianZeroEdgeWaveformOp(width, amplitude, std_dev, zero_at_edges)

        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.std_dev == std_dev.results[0]
        assert not op.zero_at_edges.value.data

        assert op.result.type == WaveformType()
        op.verify()


class TestCosWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(
            TimeAttr(800e-9),
        )
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        internal_phase = ConstantOp(PhaseAttr(1.57))

        op = CosWaveformOp(width, amplitude, frequency, internal_phase)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.frequency == frequency.results[0]
        assert op.internal_phase == internal_phase.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestSinWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(
            TimeAttr(800e-9),
        )
        frequency = ConstantOp(FrequencyAttr(5.0e9))
        internal_phase = ConstantOp(PhaseAttr(1.57))

        op = SinWaveformOp(width, amplitude, frequency, internal_phase)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.frequency == frequency.results[0]
        assert op.internal_phase == internal_phase.results[0]
        assert op.result.type == WaveformType()
        op.verify()


class TestSechWaveformOp:
    def test_initialization(self):
        amplitude = ConstantOp(AmplitudeAttr(1.0))
        width = ConstantOp(TimeAttr(800e-9))
        std_dev = ConstantOp(TimeAttr(200e-9))

        op = SechWaveformOp(width, amplitude, std_dev)
        assert op.width == width.results[0]
        assert op.amplitude == amplitude.results[0]
        assert op.std_dev == std_dev.results[0]
        assert op.result.type == WaveformType()
        op.verify()


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
